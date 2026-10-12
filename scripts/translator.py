"""Translate text in a terminal with the local NLLB GGUF via Malaga."""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import struct
import subprocess
import tempfile
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

import requests
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models/NLLB-200-distil-600M/nllb-200-distilled-600M-q4_k_m.gguf"
HOST = "127.0.0.1"
STARTUP_TIMEOUT_S = 120
REQUEST_TIMEOUT_S = 180
SOURCE_DEFAULT = "eng_Latn"
TARGET_DEFAULT = "vie_Latn"
LANGUAGE_CODE = re.compile(r"[a-z]{3}_[A-Z][a-z]{3}\Z")


@dataclass(frozen=True)
class Translation:
    text: str
    input_tokens: int
    output_tokens: int
    elapsed_s: float

    @property
    def output_tps(self) -> float:
        return self.output_tokens / self.elapsed_s if self.elapsed_s > 0 else 0.0


def _read_u32(file: BinaryIO) -> int:
    data = file.read(4)
    if len(data) != 4:
        raise ValueError("Truncated GGUF metadata")
    return struct.unpack("<I", data)[0]


def _read_u64(file: BinaryIO) -> int:
    data = file.read(8)
    if len(data) != 8:
        raise ValueError("Truncated GGUF metadata")
    return struct.unpack("<Q", data)[0]


def _read_string(file: BinaryIO) -> str:
    size = _read_u64(file)
    if size > 100_000_000:
        raise ValueError("GGUF metadata string is too large")
    data = file.read(size)
    if len(data) != size:
        raise ValueError("Truncated GGUF metadata string")
    return data.decode("utf-8")


def load_tokenizer(model: Path) -> tuple[Tokenizer, set[str]]:
    """Read the Hugging Face tokenizer stored in this Malaga GGUF."""
    with model.open("rb") as file:
        if file.read(4) != b"GGUF":
            raise ValueError(f"Not a GGUF model: {model}")
        version = _read_u32(file)
        if version not in (2, 3):
            raise ValueError(f"Unsupported GGUF version {version}")
        _read_u64(file)  # Tensor count.
        metadata_count = _read_u64(file)
        for _ in range(metadata_count):
            key = _read_string(file)
            value_type = _read_u32(file)
            if value_type != 8:
                # The embedded tokenizer is preceded only by string fields in
                # the Malaga model. Fail clearly if that format changes.
                raise ValueError(f"Unexpected GGUF metadata type before tokenizer: {key}")
            value = _read_string(file)
            if key == "tokenizer.huggingface.json":
                data = json.loads(value)
                codes = {
                    item["content"]
                    for item in data["added_tokens"]
                    if LANGUAGE_CODE.fullmatch(item["content"])
                }
                if not codes:
                    raise ValueError("No NLLB language codes found in the model tokenizer")
                return Tokenizer.from_str(value), codes
    raise ValueError("The GGUF has no embedded Hugging Face tokenizer")


def find_malaga() -> str:
    configured = os.environ.get("MALAGA_BIN")
    local_binary = ROOT / "build/malaga"
    if configured:
        found = shutil.which(configured)
    elif local_binary.is_file() and os.access(local_binary, os.X_OK):
        found = str(local_binary)
    else:
        found = shutil.which("malaga")
    if found is None:
        raise FileNotFoundError(
            "Malaga executable not found. Build https://github.com/rzafiamy/malaga "
            "and set MALAGA_BIN to its executable path."
        )
    return found


def choose_language(console: Console, label: str, current: str, codes: set[str]) -> str:
    while True:
        answer = console.input(f"[bold cyan]{label} [{current}]> [/]").strip()
        if answer == "?":
            console.print("Supported NLLB codes: " + ", ".join(sorted(codes)), markup=False)
            continue
        code = answer or current
        if code in codes:
            return code
        console.print(f"[yellow]Unknown language code: {code}. Enter ? to list valid codes.[/]")


def choose_pair(console: Console, codes: set[str], source: str, target: str) -> tuple[str, str]:
    console.print("Choose NLLB language codes. Enter ? to list them.")
    source = choose_language(console, "Source", source, codes)
    target = choose_language(console, "Target", target, codes)
    return source, target


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind((HOST, 0))
        return sock.getsockname()[1]


def _log_tail(log: BinaryIO) -> str:
    log.seek(0, os.SEEK_END)
    log.seek(max(log.tell() - 4000, 0))
    return log.read().decode("utf-8", errors="replace").strip()


@contextmanager
def malaga_server(
    binary: str, model: Path, source: str, target: str
) -> Iterator[tuple[requests.Session, str, str]]:
    port = _free_port()
    base_url = f"http://{HOST}:{port}"
    requested_device = os.environ.get("MALAGA_DEVICE", "cuda").lower()
    command = [
        binary, "serve", "--model", str(model), "--device", requested_device,
        "--host", HOST, "--port", str(port), "--default-source", source, "--default-target", target,
    ]
    session = requests.Session()
    session.trust_env = False  # Loopback must not be routed through a proxy.
    try:
        with tempfile.TemporaryFile(mode="w+b") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + STARTUP_TIMEOUT_S
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError(f"Malaga exited during startup:\n{_log_tail(log)}")
                    try:
                        response = session.get(f"{base_url}/health", timeout=1)
                        if response.ok and response.json().get("status") == "ok":
                            startup_log = _log_tail(log)
                            loaded = re.search(
                                r"loaded .+ on (.+?) in \d+(?:\.\d+)?(?:ns|µs|ms|s)", startup_log
                            )
                            if loaded is None:
                                raise RuntimeError(f"Malaga is ready but its device could not be verified:\n{startup_log}")
                            actual_device = loaded.group(1)
                            if requested_device == "cuda" and not actual_device.startswith("Cuda"):
                                raise RuntimeError(
                                    f"CUDA requested, but Malaga loaded on {actual_device}.\n{startup_log}"
                                )
                            yield session, base_url, actual_device
                            return
                    except (requests.RequestException, ValueError):
                        pass
                    time.sleep(0.2)
                raise RuntimeError(f"Malaga did not become ready within {STARTUP_TIMEOUT_S}s:\n{_log_tail(log)}")
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
    finally:
        session.close()


def translate(
    session: requests.Session, base_url: str, tokenizer: Tokenizer, text: str, source: str, target: str
) -> Translation:
    started = time.perf_counter()
    response = session.post(
        f"{base_url}/v1/translate",
        json={"text": text, "source": source, "target": target},
        timeout=REQUEST_TIMEOUT_S,
    )
    elapsed_s = time.perf_counter() - started
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(f"Malaga rejected the translation: {response.text[:300]}") from exc
    try:
        translated = response.json()["translation"]
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError("Malaga returned an invalid translation response") from exc
    if not isinstance(translated, str):
        raise RuntimeError("Malaga returned a translation that is not text")
    return Translation(
        text=translated,
        input_tokens=len(tokenizer.encode(text, add_special_tokens=False).ids),
        output_tokens=len(tokenizer.encode(translated, add_special_tokens=False).ids),
        elapsed_s=elapsed_s,
    )


def run_translator(
    session: requests.Session, base_url: str, tokenizer: Tokenizer, codes: set[str],
    source: str, target: str, console: Console,
) -> None:
    console.print(f"Ready: {source} → {target}. Commands: /pair, /exit")
    while True:
        try:
            text = console.input("\n[bold cyan]Text> [/]")
        except (EOFError, KeyboardInterrupt):
            console.print("\nBye.")
            return
        command = text.strip()
        if command == "/exit":
            console.print("Bye.")
            return
        if command == "/pair":
            try:
                source, target = choose_pair(console, codes, source, target)
            except (EOFError, KeyboardInterrupt):
                console.print("\nBye.")
                return
            console.print(f"Language pair: {source} → {target}")
            continue
        if not command:
            continue
        try:
            with console.status("Translating…"):
                result = translate(session, base_url, tokenizer, text, source, target)
        except (requests.RequestException, RuntimeError) as exc:
            console.print(f"[red]Translation failed: {exc}[/]")
            continue
        except KeyboardInterrupt:
            console.print("\nTranslation interrupted.")
            continue
        console.print(Panel(Text(result.text or "(empty translation)"), title=f"{source} → {target}"))
        console.print(
            f"[dim]Input text: {result.input_tokens} tokens · "
            f"Output text: {result.output_tokens} tokens · "
            f"Wall time: {result.elapsed_s:.2f}s · "
            f"Estimated output: {result.output_tps:.1f} tok/s[/]"
        )


def main() -> int:
    console = Console()
    try:
        if not MODEL.is_file():
            raise FileNotFoundError(f"Model not found: {MODEL}")
        tokenizer, codes = load_tokenizer(MODEL)
        binary = find_malaga()
        source, target = choose_pair(console, codes, SOURCE_DEFAULT, TARGET_DEFAULT)
        with ExitStack() as stack:
            with console.status("Loading NLLB model in Malaga…"):
                session, base_url, device = stack.enter_context(malaga_server(binary, MODEL, source, target))
            console.print(f"Model loaded on: {device}")
            run_translator(session, base_url, tokenizer, codes, source, target, console)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        console.print(f"[red]{exc}[/]")
        return 1
    except (EOFError, KeyboardInterrupt):
        console.print("\nBye.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
