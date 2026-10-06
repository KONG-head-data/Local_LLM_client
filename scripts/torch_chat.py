"""Chat with a locally downloaded Transformers safetensors checkpoint."""

from __future__ import annotations

import json
import os
import select
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text


# Edit these settings to tune the terminal chat.
CONTEXT_LIMIT = 8192
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.95
ENABLE_THINKING = True
THINKING_TOKEN_BUDGET = 1024
REFRESH_PER_SECOND = 4

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


@dataclass(frozen=True)
class TurnResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    elapsed_s: float
    decode_tps: float


def discover_models() -> list[Path]:
    """Return folders with config, tokenizer, and all indexed weight shards."""
    models = []
    for config in MODELS_DIR.rglob("config.json"):
        folder = config.parent
        if not (folder / "tokenizer_config.json").is_file():
            continue
        index = folder / "model.safetensors.index.json"
        if index.is_file():
            try:
                shards = set(json.loads(index.read_text())["weight_map"].values())
            except (OSError, ValueError, KeyError, TypeError):
                continue
            if not shards or not all((folder / shard).is_file() for shard in shards):
                continue
        elif not (folder / "model.safetensors").is_file():
            continue
        models.append(folder)
    return sorted(models, key=lambda path: str(path).lower())


def select_model(models: list[Path], console: Console) -> Path:
    if not models:
        raise FileNotFoundError(f"No complete safetensors models found under {MODELS_DIR}")

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
        console.print(f"{'>' if index == selected else ' '} {model.relative_to(ROOT)}", highlight=False)

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
            if key != b"\x1b" or not select.select([input_fd], [], [], 0.05)[0]:
                continue
            if os.read(input_fd, 1) != b"[" or not select.select([input_fd], [], [], 0.05)[0]:
                continue
            direction = os.read(input_fd, 1)
            if direction == b"A":
                selected = (selected - 1) % len(models)
            elif direction == b"B":
                selected = (selected + 1) % len(models)
            else:
                continue
            sys.stdout.write(f"\x1b[{len(models)}A")
            for index, model in enumerate(models):
                sys.stdout.write(f"\x1b[2K\r{'>' if index == selected else ' '} {model.relative_to(ROOT)}\n")
            sys.stdout.flush()
    finally:
        termios.tcsetattr(input_fd, termios.TCSADRAIN, original_settings)


def load_model(folder: Path) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForMultimodalLM, AutoProcessor, AutoTokenizer

    config = json.loads((folder / "config.json").read_text())
    multimodal = "vision_config" in config or config.get("model_type") == "qwen3_5"
    if multimodal:
        processor = AutoProcessor.from_pretrained(folder, local_files_only=True)
        model_class = AutoModelForMultimodalLM
    else:
        processor = AutoTokenizer.from_pretrained(folder, local_files_only=True)
        model_class = AutoModelForCausalLM
    model = model_class.from_pretrained(
        folder,
        local_files_only=True,
        use_safetensors=True,
        dtype="auto",
        device_map="auto",
    )
    model.eval()
    console_device = getattr(model, "hf_device_map", None) or str(model.device)
    print(f"Loaded on: {console_device}; CUDA available: {torch.cuda.is_available()}")
    return model, processor


def prepare_inputs(processor: Any, history: list[dict[str, str]], user_text: str) -> tuple[list[dict[str, str]], Any]:
    messages = [*history, {"role": "user", "content": user_text}]
    prompt_messages = messages
    if ENABLE_THINKING:
        prompt_messages = [
            {
                "role": "system",
                "content": f"Aim to finish your thinking within {THINKING_TOKEN_BUDGET} tokens, then provide your final answer.",
            },
            *messages,
        ]
    inputs = processor.apply_chat_template(
        prompt_messages,
        add_generation_prompt=True,
        enable_thinking=ENABLE_THINKING,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    return messages, inputs


def render_live(text: str, generated: int, prompt_tokens: int, tps: float, console: Console) -> Group:
    stats = Panel(
        Text(f"Generated: {generated} tokens   Context: {prompt_tokens:,}/{CONTEXT_LIMIT:,}   Speed: {tps:.1f} tok/s"),
        title="Live stats",
    )
    reply_rows = max(console.height - len(console.render_lines(stats)) - 2, 1)
    source = (text or "Waiting for first token…")[-max(console.width * console.height * 2, 1):]
    lines = Text(source).wrap(console, max(console.width - 4, 1), overflow="fold")
    truncated = len(source) < len(text) or len(lines) > reply_rows
    preview = "\n".join(line.plain for line in lines[-reply_rows:])
    return Group(Panel(Text(preview), title="Assistant (latest)" if truncated else "Assistant"), stats)


def generate_turn(model: Any, processor: Any, inputs: Any, console: Console) -> TurnResult:
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

    tokenizer = getattr(processor, "tokenizer", processor)
    prompt_tokens = inputs["input_ids"].shape[-1]
    stop_event = threading.Event()

    class StopRequested(StoppingCriteria):
        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
            return stop_event.is_set()

    class CountingStreamer(TextIteratorStreamer):
        completion_tokens = 0

        def put(self, value: Any) -> None:
            if self.next_tokens_are_prompt:
                super().put(value)
                return
            self.completion_tokens += value.numel()
            super().put(value)

    streamer = CountingStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=0.2)
    errors: list[BaseException] = []
    pieces: list[str] = []
    first_token_at: float | None = None

    def worker() -> None:
        try:
            with torch.inference_mode():
                model.generate(
                    **inputs.to(model.device),
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=TEMPERATURE > 0,
                    temperature=TEMPERATURE if TEMPERATURE > 0 else None,
                    top_p=TOP_P if TEMPERATURE > 0 else None,
                    streamer=streamer,
                    stopping_criteria=StoppingCriteriaList([StopRequested()]),
                )
        except BaseException as exc:
            errors.append(exc)
            streamer.end()

    thread = threading.Thread(target=worker)
    thread.start()
    try:
        with Live(render_live("", 0, prompt_tokens, 0, console), console=console,
                  refresh_per_second=REFRESH_PER_SECOND, transient=True) as live:
            while True:
                try:
                    piece = next(streamer)
                except Empty:
                    if not thread.is_alive() and errors:
                        break
                    continue
                except StopIteration:
                    break
                if piece:
                    pieces.append(piece)
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                elapsed = time.perf_counter() - first_token_at if first_token_at else 0.0
                speed = max(streamer.completion_tokens - 1, 0) / elapsed if elapsed > 0 else 0.0
                live.update(render_live("".join(pieces), streamer.completion_tokens, prompt_tokens, speed, console))
    except KeyboardInterrupt:
        stop_event.set()
        thread.join()
        raise
    thread.join()
    if errors:
        raise RuntimeError(str(errors[0])) from errors[0]
    elapsed = time.perf_counter() - first_token_at if first_token_at else 0.0
    count = streamer.completion_tokens
    return TurnResult("".join(pieces), prompt_tokens, count, elapsed,
                      max(count - 1, 0) / elapsed if elapsed > 0 else 0.0)


def run_chat(model: Any, processor: Any, console: Console) -> None:
    history: list[dict[str, str]] = []
    console.print(f"Ready. Context limit: {CONTEXT_LIMIT:,} tokens. Commands: /reset, /exit")
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
            console.print("Conversation cleared.")
            continue
        if not user_text:
            continue

        messages, inputs = prepare_inputs(processor, history, user_text)
        prompt_tokens = inputs["input_ids"].shape[-1]
        if prompt_tokens + MAX_NEW_TOKENS >= CONTEXT_LIMIT:
            console.print(
                f"This turn needs {prompt_tokens + MAX_NEW_TOKENS:,} tokens including reply space; "
                f"capacity is {CONTEXT_LIMIT:,}. Enter /reset to clear history."
            )
            continue
        try:
            result = generate_turn(model, processor, inputs, console)
        except KeyboardInterrupt:
            console.print("\nGeneration canceled. Conversation unchanged.")
            continue
        except Exception as exc:
            console.print(f"Generation failed: {exc}")
            continue
        history = [*messages, {"role": "assistant", "content": result.text}]
        console.print(Panel(Text(result.text or "(empty reply)"), title="Assistant"))
        console.print(
            f"[dim]Generated: {result.completion_tokens} tokens · "
            f"Context: {result.prompt_tokens:,}/{CONTEXT_LIMIT:,} · "
            f"{result.decode_tps:.1f} tok/s · decode {result.elapsed_s:.2f}s[/]"
        )
        if result.completion_tokens >= MAX_NEW_TOKENS:
            console.print("[yellow]Reply reached the configured token limit.[/]")


def main() -> int:
    console = Console()
    try:
        models = discover_models()
        model_path = select_model(models, console)
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            import accelerate  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Torch chat needs torch, transformers, and accelerate. "
                "See the Safetensors terminal chat section in README.md."
            ) from exc
        console.print(f"Loading {model_path.relative_to(ROOT)} …")
        model, processor = load_model(model_path)
        run_chat(model, processor, console)
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        console.print(f"[red]{exc}[/]")
        return 1
    except KeyboardInterrupt:
        console.print("\nBye.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
