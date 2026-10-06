"""Run a selected local GGUF model as a terminal chat."""

from __future__ import annotations

import os
import select
import subprocess
import sys
import termios
import time
import tty
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llama_cpp import Llama, llama_cpp
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

if TYPE_CHECKING:
    from src.services import BaseModelService


# Edit these settings to tune the terminal chat.
N_CTX = 131072
N_GPU_LAYERS = -1
FLASH_ATTN = True
N_BATCH = 64
N_UBATCH = 64
TEMPERATURE = 0.7
TOP_P = 0.95
MAX_NEW_TOKENS = 8192
ENABLE_THINKING = True
THINKING_TOKEN_BUDGET = 1024
REFRESH_PER_SECOND = 4

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"


def require_gpu_if_requested() -> None:
    if N_GPU_LAYERS == 0:
        return
    if not llama_cpp.llama_supports_gpu_offload():
        raise RuntimeError(
            "CPU-only llama-cpp-python build installed. Run `uv sync` to install "
            "the CUDA wheel locked by this project. Set N_GPU_LAYERS=0 only if CPU mode is intended."
        )
    try:
        subprocess.run(
            ["nvidia-smi", "-L"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("CUDA build is installed, but no NVIDIA GPU is visible to this process.") from exc


@dataclass(frozen=True)
class Turn:
    messages: list[dict[str, str]]
    prompt_tokens: list[int]


@dataclass(frozen=True)
class TurnResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    elapsed_s: float
    decode_tps: float
    context_used: int
    finish_reason: str | None


def discover_models() -> list[Path]:
    return sorted(MODELS_DIR.rglob("*.gguf"), key=lambda path: str(path).lower())


def select_model(models: list[Path], console: Console) -> Path:
    if not models:
        raise FileNotFoundError(f"No GGUF models found under {MODELS_DIR}")

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        for index, model in enumerate(models, 1):
            console.print(f"{index}. {model.relative_to(ROOT)}")
        choice = int(console.input("Model number: "))
        if not 1 <= choice <= len(models):
            raise ValueError("Model number is out of range")
        return models[choice - 1]

    selected = 0
    console.print("Use Up/Down arrows to select a model and press Enter:\n")
    for index, model in enumerate(models):
        marker = ">" if index == selected else " "
        console.print(f"{marker} {model.relative_to(ROOT)}", highlight=False)

    input_fd = sys.stdin.fileno()
    original_settings = termios.tcgetattr(input_fd)
    try:
        tty.setcbreak(input_fd)
        while True:
            key = os.read(input_fd, 1)
            if key in (b"\r", b"\n"):
                console.print()
                return models[selected]
            if key == b"\x03":
                raise KeyboardInterrupt
            if key == b"\x1b" and select.select([input_fd], [], [], 0.05)[0]:
                sequence = os.read(input_fd, 1)
                if sequence == b"[" and select.select([input_fd], [], [], 0.05)[0]:
                    direction = os.read(input_fd, 1)
                    if direction == b"A":
                        selected = (selected - 1) % len(models)
                    elif direction == b"B":
                        selected = (selected + 1) % len(models)
                    else:
                        continue
                else:
                    continue
            else:
                continue

            sys.stdout.write(f"\x1b[{len(models)}A")
            for index, model in enumerate(models):
                marker = ">" if index == selected else " "
                sys.stdout.write(f"\x1b[2K\r{marker} {model.relative_to(ROOT)}\n")
            sys.stdout.flush()
    finally:
        termios.tcsetattr(input_fd, termios.TCSADRAIN, original_settings)


def service_for_model(model: Path) -> BaseModelService:
    # src.config reads MODEL_PATH when src.services is imported.
    os.environ["MODEL_PATH"] = str(model)
    from src.services import GemmaService, QwenService

    model_type = model.name.lower()
    template = next(
        (parent / "chat_template.jinja" for parent in model.parents if (parent / "chat_template.jinja").is_file()),
        None,
    )
    if template is None:
        raise FileNotFoundError(f"No chat_template.jinja found for {model}")
    if "gemma" in model_type:
        return GemmaService(str(template))
    if "qwen" in model_type:
        return QwenService(str(template))
    raise ValueError(f"Unsupported model family: {model.name}")


def prepare_turn(
    llm: Llama,
    service: BaseModelService,
    history: list[dict[str, str]],
    user_text: str,
) -> Turn:
    messages = [*history, {"role": "user", "content": user_text}]
    prompt_messages = messages
    if ENABLE_THINKING:
        prompt_messages = [
            {
                "role": "system",
                "content": (
                    f"Aim to finish your thinking within {THINKING_TOKEN_BUDGET} tokens, "
                    "then provide your final answer."
                ),
            },
            *messages,
        ]
    prompt = service.render_prompt(prompt_messages, enable_thinking=ENABLE_THINKING)
    # Passing token IDs to create_completion ensures the preflight count is the
    # exact prompt sent to the model, including any model-required BOS token.
    prompt_tokens = llm.tokenize(prompt.encode("utf-8"), add_bos=True, special=True)
    return Turn(messages=messages, prompt_tokens=prompt_tokens)


def fits_context(prompt_count: int, context_size: int, max_new_tokens: int) -> bool:
    # llama-cpp-python requires the prompt itself to be shorter than n_ctx.
    return prompt_count + max_new_tokens < context_size


def decode_speed(tokens: int, elapsed_s: float) -> float:
    # The first token's wait includes prompt evaluation. Measure decode speed
    # from the first emitted token onward.
    return max(tokens - 1, 0) / elapsed_s if elapsed_s > 0 else 0.0


def render_live(text: str, generated: int, context_tokens: int, capacity: int, tps: float, console: Console) -> Group:
    stats = Text(
        f"Generated: {generated} tokens   Context: {context_tokens:,}/{capacity:,}   Speed: {tps:.1f} tok/s"
    )
    stats_panel = Panel(stats, title="Live stats")
    stats_rows = len(console.render_lines(stats_panel))
    reply_rows = max(console.height - stats_rows - 2, 1)
    # Keep the live render within the terminal height. Rich otherwise crops its
    # bottom, hiding the newest tokens and the stats as the reply grows.
    source = (text or "Waiting for first token…")[-max(console.width * console.height * 2, 1):]
    lines = Text(source).wrap(console, max(console.width - 4, 1), overflow="fold")
    truncated = len(source) < len(text) or len(lines) > reply_rows
    preview = "\n".join(line.plain for line in lines[-reply_rows:])
    title = "Assistant (latest)" if truncated else "Assistant"
    return Group(Panel(Text(preview), title=title), stats_panel)


def generate_turn(
    llm: Llama,
    service: BaseModelService,
    turn: Turn,
    console: Console,
) -> TurnResult:
    prompt_count = len(turn.prompt_tokens)
    context_size = llm.n_ctx()
    pieces: list[str] = []
    first_token_at: float | None = None
    final_usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    generated = 0

    stream = llm.create_completion(
        prompt=turn.prompt_tokens,
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        stop=service.get_stop_tokens(),
        stream=True,
    )
    try:
        with Live(
            render_live("", 0, prompt_count, context_size, 0.0, console),
            console=console,
            refresh_per_second=REFRESH_PER_SECOND,
            transient=True,
        ) as live:
            for chunk in stream:
                choice = chunk["choices"][0]
                piece = choice.get("text", "")
                if piece:
                    pieces.append(piece)
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                if chunk.get("usage"):
                    final_usage = chunk["usage"]
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]

                generated = max(generated, llm.n_tokens - prompt_count)
                if final_usage:
                    generated = final_usage["completion_tokens"]
                elapsed = time.perf_counter() - first_token_at if first_token_at else 0.0
                live.update(render_live("".join(pieces), generated, prompt_count, context_size, decode_speed(generated, elapsed), console))
    finally:
        close = getattr(stream, "close", None)
        if close:
            close()

    finished_at = time.perf_counter()
    completion_count = final_usage["completion_tokens"] if final_usage else generated
    elapsed = finished_at - first_token_at if first_token_at else 0.0
    return TurnResult(
        text="".join(pieces),
        prompt_tokens=prompt_count,
        completion_tokens=completion_count,
        elapsed_s=elapsed,
        decode_tps=decode_speed(completion_count, elapsed),
        context_used=prompt_count,
        finish_reason=finish_reason,
    )


def run_chat(llm: Llama, service: BaseModelService, console: Console) -> None:
    history: list[dict[str, str]] = []
    console.print(f"Ready. Context: {llm.n_ctx():,} tokens. Commands: /reset, /exit")
    while True:
        try:
            user_text = console.input("\n[bold cyan]You> [/]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nBye.")
            return

        if user_text == "/exit":
            console.print("Bye.")
            return
        if user_text == "/reset":
            history.clear()
            llm.reset()
            console.print("Conversation cleared.")
            continue
        if not user_text:
            continue

        turn = prepare_turn(llm, service, history, user_text)
        if not fits_context(len(turn.prompt_tokens), llm.n_ctx(), MAX_NEW_TOKENS):
            console.print(
                f"This turn needs {len(turn.prompt_tokens) + MAX_NEW_TOKENS:,} tokens "
                f"including reply space; capacity is {llm.n_ctx():,}. Enter /reset to clear history."
            )
            continue

        try:
            result = generate_turn(llm, service, turn, console)
        except KeyboardInterrupt:
            llm.reset()
            console.print("\nGeneration canceled. Conversation unchanged.")
            continue
        except Exception as exc:
            llm.reset()
            console.print(f"Generation failed: {exc}")
            continue

        history = [*turn.messages, {"role": "assistant", "content": result.text}]
        console.print(Panel(Text(result.text or "(empty reply)"), title="Assistant"))
        console.print(
            f"[dim]Generated: {result.completion_tokens} tokens · "
            f"Context: {result.context_used:,}/{llm.n_ctx():,} · "
            f"{result.decode_tps:.1f} tok/s · "
            f"decode {result.elapsed_s:.2f}s[/]"
        )
        if result.finish_reason == "length":
            console.print("[yellow]Reply reached the configured token limit.[/]")


def main() -> int:
    console = Console()
    try:
        require_gpu_if_requested()
        model = select_model(discover_models(), console)
        service = service_for_model(model)
        console.print(f"Loading {model.relative_to(ROOT)} …")
        llm = Llama(
            model_path=str(model),
            n_ctx=N_CTX,
            n_gpu_layers=N_GPU_LAYERS,
            flash_attn=FLASH_ATTN,
            n_batch=N_BATCH,
            n_ubatch=N_UBATCH,
            logits_all=False,
            verbose=False,
        )
        run_chat(llm, service, console)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        console.print(f"[red]{exc}[/]")
        return 1
    except KeyboardInterrupt:
        console.print("\nBye.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
