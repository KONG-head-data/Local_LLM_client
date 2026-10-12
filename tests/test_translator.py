from io import StringIO

import requests
import pytest
from rich.console import Console

from scripts import translator


def test_local_gguf_contains_tokenizer_and_language_codes():
    tokenizer, codes = translator.load_tokenizer(translator.MODEL)

    assert {"eng_Latn", "vie_Latn", "fra_Latn"} <= codes
    assert len(tokenizer.encode("Hello, world.", add_special_tokens=False).ids) == 4


def test_choose_pair_accepts_defaults_and_retries_invalid_code(monkeypatch):
    output = StringIO()
    console = Console(file=output, force_terminal=False)
    answers = iter(["bad", "fra_Latn", ""])
    monkeypatch.setattr(console, "input", lambda _prompt: next(answers))

    pair = translator.choose_pair(
        console, {"eng_Latn", "vie_Latn", "fra_Latn"}, "eng_Latn", "vie_Latn"
    )

    assert pair == ("fra_Latn", "vie_Latn")
    assert "Unknown language code: bad" in output.getvalue()


def test_find_malaga_uses_local_build_without_environment_variable(monkeypatch, tmp_path):
    binary = tmp_path / "build/malaga"
    binary.parent.mkdir()
    binary.write_text("stub")
    binary.chmod(0o755)
    monkeypatch.setattr(translator, "ROOT", tmp_path)
    monkeypatch.delenv("MALAGA_BIN", raising=False)

    assert translator.find_malaga() == str(binary)


def test_translate_sends_selected_pair_and_reports_text_metrics():
    tokenizer, _ = translator.load_tokenizer(translator.MODEL)

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"translation": "Bonjour le monde."}

    class FakeSession:
        def __init__(self):
            self.sent = None

        def post(self, url, **kwargs):
            self.sent = (url, kwargs)
            return FakeResponse()

    session = FakeSession()
    result = translator.translate(
        session, "http://127.0.0.1:4567", tokenizer, "Hello world.", "eng_Latn", "fra_Latn"
    )

    assert session.sent[0].endswith("/v1/translate")
    assert session.sent[1]["json"] == {
        "text": "Hello world.", "source": "eng_Latn", "target": "fra_Latn"
    }
    assert result.text == "Bonjour le monde."
    assert result.input_tokens == len(tokenizer.encode("Hello world.", add_special_tokens=False).ids)
    assert result.output_tokens == len(tokenizer.encode(result.text, add_special_tokens=False).ids)
    assert result.elapsed_s >= 0
    assert result.output_tps >= 0


def test_malaga_server_waits_for_health_and_stops_child(monkeypatch):
    events = []

    class FakeResponse:
        ok = True

        def json(self):
            return {"status": "ok"}

    class FakeSession:
        trust_env = True

        def get(self, url, timeout):
            events.append(("health", url, timeout))
            return FakeResponse()

        def close(self):
            events.append("session closed")

    class FakeProcess:
        def __init__(self, command, **kwargs):
            events.append(("start", command))
            kwargs["stdout"].write(b"INFO malaga: loaded model on Cuda(0) in 1.2s\n")
            kwargs["stdout"].flush()
            self.running = True

        def poll(self):
            return None if self.running else 0

        def terminate(self):
            self.running = False
            events.append("terminated")

        def wait(self, timeout):
            events.append(("wait", timeout))

    monkeypatch.setattr(translator.requests, "Session", FakeSession)
    monkeypatch.setattr(translator.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(translator, "_free_port", lambda: 4567)
    monkeypatch.delenv("MALAGA_DEVICE", raising=False)

    with translator.malaga_server("malaga", translator.MODEL, "eng_Latn", "vie_Latn") as (session, url, device):
        assert url == "http://127.0.0.1:4567"
        assert session.trust_env is False
        assert device == "Cuda(0)"

    assert events[0][1][-4:] == ["--default-source", "eng_Latn", "--default-target", "vie_Latn"]
    assert events[0][1][events[0][1].index("--device") + 1] == "cuda"
    assert events[1] == ("health", "http://127.0.0.1:4567/health", 1)
    assert events[-3:] == ["terminated", ("wait", 5), "session closed"]


def test_malaga_server_rejects_cpu_when_cuda_requested(monkeypatch):
    class FakeResponse:
        ok = True

        def json(self):
            return {"status": "ok"}

    class FakeSession:
        def get(self, *_args, **_kwargs):
            return FakeResponse()

        def close(self):
            pass

    class FakeProcess:
        def __init__(self, _command, **kwargs):
            kwargs["stdout"].write(b"INFO malaga: loaded model on Cpu in 1.2s\n")
            kwargs["stdout"].flush()
            self.running = True

        def poll(self):
            return None if self.running else 0

        def terminate(self):
            self.running = False

        def wait(self, timeout):
            pass

    monkeypatch.setattr(translator.requests, "Session", FakeSession)
    monkeypatch.setattr(translator.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(translator, "_free_port", lambda: 4567)
    monkeypatch.delenv("MALAGA_DEVICE", raising=False)

    with pytest.raises(RuntimeError, match="CUDA requested, but Malaga loaded on Cpu"):
        with translator.malaga_server("malaga", translator.MODEL, "eng_Latn", "vie_Latn"):
            pass


def test_translation_http_error_shows_server_response():
    class FakeSession:
        def post(self, *_args, **_kwargs):
            response = requests.Response()
            response.status_code = 400
            response._content = b"unsupported language"
            return response

    class FakeTokenizer:
        pass

    try:
        translator.translate(FakeSession(), "http://127.0.0.1:4567", FakeTokenizer(), "x", "bad", "fra_Latn")
    except RuntimeError as exc:
        assert "unsupported language" in str(exc)
    else:
        raise AssertionError("Expected a translation error")
