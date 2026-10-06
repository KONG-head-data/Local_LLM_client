from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import chat
import pytest
from rich.console import Console
from rich.live_render import LiveRender


class FakeModel:
    def __init__(self):
        self.n_tokens = 0
        self.calls = []
        self.resets = 0

    def n_ctx(self):
        return 100

    def tokenize(self, text, add_bos, special):
        assert add_bos and special
        return list(range(len(text) // 10 + 1))

    def create_completion(self, **kwargs):
        self.calls.append(kwargs)
        self.n_tokens = len(kwargs["prompt"]) + 2
        yield {"choices": [{"text": "Hello", "finish_reason": None}]}
        self.n_tokens += 3
        yield {"choices": [{"text": " there", "finish_reason": None}]}
        yield {
            "choices": [{"text": "", "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": len(kwargs["prompt"]),
                "completion_tokens": 6,
                "total_tokens": len(kwargs["prompt"]) + 6,
            },
        }

    def reset(self):
        self.resets += 1
        self.n_tokens = 0


class FakeService:
    def render_prompt(self, messages, enable_thinking=False):
        return "\n".join(message["content"] for message in messages)

    def get_stop_tokens(self):
        return ["<stop>"]


def test_gpu_mode_rejects_cpu_only_llama_build(monkeypatch):
    monkeypatch.setattr(chat, "N_GPU_LAYERS", -1)
    monkeypatch.setattr(chat, "llama_cpp", SimpleNamespace(llama_supports_gpu_offload=lambda: False), raising=False)
    with pytest.raises(RuntimeError, match="CPU-only"):
        chat.require_gpu_if_requested()


def test_model_picker_accepts_number_when_input_is_not_a_terminal(monkeypatch):
    class FakeConsole:
        def print(self, _message):
            pass

        def input(self, _prompt):
            return "2"

    monkeypatch.setattr(chat.sys.stdin, "isatty", lambda: False)
    models = [chat.ROOT / "models/first.gguf", chat.ROOT / "models/second.gguf"]
    assert chat.select_model(models, FakeConsole()) == models[1]


def test_selected_model_uses_its_own_template():
    model = Path("models/Qwen-3.5-4B/weight/Qwen3.5-4B-Q4_K_M.gguf").resolve()
    service = chat.service_for_model(model)
    assert "Qwen-3.5-4B/chat_template.jinja" in service.template_path


def test_prepare_turn_and_context_limit(monkeypatch):
    monkeypatch.setattr(chat, "ENABLE_THINKING", False)
    model = FakeModel()
    history = [{"role": "user", "content": "Earlier"}]
    turn = chat.prepare_turn(model, FakeService(), history, "Now")

    assert turn.messages == [*history, {"role": "user", "content": "Now"}]
    assert turn.prompt_tokens == list(range(len("Earlier\nNow".encode()) // 10 + 1))
    assert chat.fits_context(87, 100, 12)
    assert not chat.fits_context(88, 100, 12)
    assert history == [{"role": "user", "content": "Earlier"}]


@pytest.mark.parametrize(
    "model_name",
    ["Qwen-3.5-4B", "Gemma"],
)
def test_prepare_turn_passes_thinking_setting_to_model_template(monkeypatch, model_name):
    service = chat.service_for_model(chat.ROOT / "models" / model_name / f"{model_name}.gguf")
    prompts = []
    model = FakeModel()
    original_tokenize = model.tokenize

    def capture_prompt(text, add_bos, special):
        prompts.append(text.decode("utf-8"))
        return original_tokenize(text, add_bos, special)

    model.tokenize = capture_prompt
    monkeypatch.setattr(chat, "THINKING_TOKEN_BUDGET", 1024)

    for enabled in (False, True):
        monkeypatch.setattr(chat, "ENABLE_THINKING", enabled)
        turn = chat.prepare_turn(model, service, [], "Hello")
        assert turn.messages == [{"role": "user", "content": "Hello"}]

    guidance = "Aim to finish your thinking within 1024 tokens, then provide your final answer."
    assert guidance not in prompts[0]
    assert guidance in prompts[1]

    if model_name.startswith("Qwen"):
        assert prompts[0].endswith("<think>\n\n</think>\n\n")
        assert prompts[1].endswith("<think>\n")
    else:
        assert "<|think|>" not in prompts[0]
        assert "<|think|>" in prompts[1]


def test_streaming_stats_use_model_tokens_and_final_usage():
    model = FakeModel()
    console = Console(file=StringIO(), force_terminal=False)
    turn = chat.Turn(messages=[], prompt_tokens=[1, 2, 3])

    result = chat.generate_turn(model, FakeService(), turn, console)

    assert result.text == "Hello there"
    assert result.prompt_tokens == 3
    assert result.completion_tokens == 6
    assert result.context_used == 3
    assert result.finish_reason == "stop"
    assert result.decode_tps >= 0
    assert model.calls[0]["prompt"] == [1, 2, 3]
    assert model.calls[0]["stream"] is True
    assert model.calls[0]["stop"] == ["<stop>"]


@pytest.mark.parametrize(
    ("width", "height", "reply"),
    [
        (50, 10, "\n".join(f"line {index:02}" for index in range(30)) + "\nLATEST_MARKER"),
        (30, 8, "long wrapped words " * 30 + "LATEST_MARKER"),
    ],
)
def test_long_live_reply_shows_latest_text_and_stats(monkeypatch, width, height, reply):
    class LongReplyModel(FakeModel):
        def create_completion(self, **kwargs):
            self.n_tokens = len(kwargs["prompt"]) + 300
            yield {"choices": [{"text": reply, "finish_reason": "stop"}]}

    class CapturingLive:
        def __init__(self, renderable, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def update(self, renderable):
            live_view = LiveRender(renderable)
            visible.append("".join(segment.text for segment in console.render(live_view)))

    console = Console(width=width, height=height, force_terminal=False)
    visible = []
    monkeypatch.setattr(chat, "Live", CapturingLive)

    result = chat.generate_turn(LongReplyModel(), FakeService(), chat.Turn([], [1, 2, 3]), console)

    assert result.text == reply
    assert "LATEST_MARKER" in visible[-1]
    assert "Generated: 300 tokens" in visible[-1]
    assert "Context: 3/100" in visible[-1]
    assert len(visible[-1].splitlines()) <= height


def test_full_context_waits_for_explicit_reset(monkeypatch):
    class FakeConsole:
        def __init__(self):
            self.inputs = iter(["A long message", "/reset", "/exit"])
            self.output = []

        def input(self, _prompt):
            return next(self.inputs)

        def print(self, message):
            self.output.append(message)

    model = FakeModel()
    console = FakeConsole()
    monkeypatch.setattr(chat, "MAX_NEW_TOKENS", 99)

    chat.run_chat(model, FakeService(), console)

    assert model.calls == []
    assert model.resets == 1
    assert any("Enter /reset" in message for message in console.output)
    assert "Conversation cleared." in console.output
