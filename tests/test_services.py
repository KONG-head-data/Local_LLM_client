import pytest
import os
from pathlib import Path
from src.services import ServiceFactory, GemmaService, QwenService

@pytest.fixture
def project_root():
    return str(Path(__file__).parent.parent)

def test_service_factory_gemma(project_root):
    service = ServiceFactory.get_service("gemma", project_root)
    assert isinstance(service, GemmaService)
    assert "Gemma" in service.template_path

def test_service_factory_qwen(project_root):
    # This test assumes at least one Qwen template exists in models/
    try:
        service = ServiceFactory.get_service("qwen", project_root)
        assert isinstance(service, QwenService)
        assert "Qwen" in service.template_path
    except FileNotFoundError:
        pytest.skip("Qwen template not found in models directory")

def test_gemma_render_prompt(project_root):
    service = ServiceFactory.get_service("gemma", project_root)
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ]
    prompt = service.render_prompt(messages)
    
    # Check if assistant was converted to model for Gemma
    assert "<|turn>user\nHello<turn|>" in prompt
    assert "<|turn>model\nHi there!<turn|>" in prompt
    assert "<|turn>model\n" in prompt # Generation prompt

def test_gemma_parse_response(monkeypatch):
    # Mock open to avoid FileNotFoundError and support context manager
    class MockFile:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return "template content"
    
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: MockFile())
    
    service = GemmaService(template_path="dummy")
    
    text = "<|channel>thought\nI should help the user.\n<channel|>Hello! <|tool_call|>call:get_weather{location:\"London\"}<tool_call|>"
    parsed = service.parse_response(text)
    
    assert parsed["thinking"] == "I should help the user."
    assert parsed["content"] == "Hello!"
    assert len(parsed["tool_calls"]) == 1
    assert parsed["tool_calls"][0]["name"] == "get_weather"
    assert parsed["tool_calls"][0]["arguments"] == {"location": "London"}

def test_qwen_parse_response():
    # Mocking QwenService since we might not have the file easily accessible for unit test
    class MockQwenService(QwenService):
        def __init__(self):
            pass
            
    service = MockQwenService()
    text = "<think>\nThinking process\n</think>\nI will call a tool.\n<tool_call>\n<function=calculate_sum>\n<parameter=a>\n5\n</parameter>\n<parameter=b>\n10\n</parameter>\n</function>\n</tool_call>"
    parsed = service.parse_response(text)
    
    assert parsed["thinking"] == "Thinking process"
    assert parsed["content"] == "I will call a tool."
    assert len(parsed["tool_calls"]) == 1
    assert parsed["tool_calls"][0]["name"] == "calculate_sum"
    assert parsed["tool_calls"][0]["arguments"] == {"a": 5, "b": 10}

def test_gemma_stop_tokens(project_root):
    service = ServiceFactory.get_service("gemma", project_root)
    stops = service.get_stop_tokens()
    assert "<end_of_turn>" in stops
    assert "<turn|>" in stops
