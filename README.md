# Local LLM client

## Download a model

Edit `REPO_ID`, `REVISION`, and `MODEL_FOLDER` near the top of
`download_model.py`, then run:

```bash
uv run python download_model.py
```

The script discovers every file in the repository and downloads it to
`models/<MODEL_FOLDER>/`, preserving subfolders. It does not use a Hugging Face
cache. Existing files with the expected size are skipped. The current example
uses a Qwen 3.5 4B Q4_K_M repository; replace `REPO_ID` with the repository you
want. Set `HF_TOKEN` in the script only if the repository requires
authentication. No command-line parameters are needed.

## Terminal chat

Install the project dependencies, then run:

```bash
uv sync
uv run python chat.py
```

Choose a GGUF model with the arrow keys and Enter. Type messages to chat, `/reset` to clear the conversation, or `/exit` to quit. The live panel and final line show two separate token counts: `Context` is the prompt sent at the start of the turn, and `Generated` is the new reply. Context is recalculated for each turn and does not include the current reply's generated tokens.

Edit the constants near the top of `chat.py` to change the context size, GPU offload, batch sizes, sampling, or maximum reply length. `ENABLE_THINKING = True` requests reasoning from a model whose chat template supports it, and `THINKING_TOKEN_BUDGET = 1024` asks the model to finish thinking within roughly that many tokens. This budget is guidance, not an enforced cutoff; `MAX_NEW_TOKENS` remains the hard limit for the whole reply. The terminal displays any reasoning the model emits as part of its raw reply. `uv sync` builds `llama-cpp-python` with CUDA for the RTX 3060 (compute capability 8.6), so the CUDA toolkit and NVIDIA driver must be installed. Chat checks that the CUDA backend and an NVIDIA GPU are visible before loading a model; if you intentionally want CPU mode, set `N_GPU_LAYERS = 0`. If the full prompt plus reserved reply tokens will not fit, the chat waits for `/reset` instead of dropping earlier messages.

The existing `./start.sh` command still starts the local HTTP server.

## Safetensors terminal chat

For a local Transformers checkpoint with `config.json`, tokenizer files, and complete
`.safetensors` weights under `models/`, run:

```bash
uv run python scripts/torch_chat.py
```

Choose a model folder, then chat with `/reset` and `/exit` as in `chat.py`.
The script keeps the full conversation until `/reset`, streams the reply, and shows
separate prompt context and newly generated token counts. Change the constants at
the top of `scripts/torch_chat.py` to adjust the context limit, sampling, reply
length, and thinking request. It loads local files only and uses Transformers'
automatic device placement; the reported device map shows whether layers are on
CUDA or CPU. Install a CUDA-enabled PyTorch build appropriate for your driver if
you want GPU execution.

## TensorRT-LLM environment

TensorRT-LLM 0.21.0 is installed in `.venv-tensorrt-llm/` on DRIVE_A. This
version's support matrix includes the RTX 3060's SM 8.6 architecture. The
separate environment keeps its PyTorch and CUDA Python versions independent of
the project's regular `.venv`.

To recreate it, run from the project directory:

```bash
uv venv .venv-tensorrt-llm --python 3.12
UV_CACHE_DIR=/mnt/DRIVE_A/UV_CACHE_DIR uv pip install \
  --python .venv-tensorrt-llm/bin/python \
  -r requirements-tensorrt-llm.txt
```

Verify the installation and GPU visibility with:

```bash
.venv-tensorrt-llm/bin/python -c 'import torch, tensorrt_llm; print(tensorrt_llm.__version__, torch.cuda.get_device_name(0))'
```

Use this interpreter for TensorRT-LLM scripts. TensorRT-LLM takes supported
Hugging Face model checkpoints; the GGUF terminal chat uses `chat.py`.
