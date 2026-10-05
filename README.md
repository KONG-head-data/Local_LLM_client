# Local LLM client

## Terminal chat

Install the project dependencies, then run:

```bash
uv sync
uv run python chat.py
```

Choose a GGUF model with the arrow keys and Enter. Type messages to chat, `/reset` to clear the conversation, or `/exit` to quit. A live panel shows generated tokens per second and the context used out of the configured limit. The final line for each reply shows its output token count and decode speed. The context count includes the current prompt and generated reply; it is recalculated for each turn.

Edit the constants near the top of `chat.py` to change the context size, GPU offload, batch sizes, sampling, or maximum reply length. Set `ENABLE_THINKING = True` to request reasoning from a model whose chat template supports it; the terminal displays any reasoning the model emits as part of its raw reply. The default is `False`. `uv sync` builds `llama-cpp-python` with CUDA for the RTX 3060 (compute capability 8.6), so the CUDA toolkit and NVIDIA driver must be installed. Chat checks that the CUDA backend and an NVIDIA GPU are visible before loading a model; if you intentionally want CPU mode, set `N_GPU_LAYERS = 0`. If the full prompt plus reserved reply tokens will not fit, the chat waits for `/reset` instead of dropping earlier messages.

The existing `./start.sh` command still starts the local HTTP server.
