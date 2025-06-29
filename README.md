# OINK 🐖 — Over‑provisioned Inference ‘N’ Kit

A tiny **OpenAI‑compatible** server that blatantly hogs your M‑series Mac’s resources in the name of *speed*. With a single Python file (`oink.py`) you get:

* **< 1 s first‑token latency** on an M3 Ultra (with 4‑bit 7 B models).
* The standard `/v1/chat/completions` (streaming + non‑stream) and `/v1/models` endpoints.
* Zero external services: MLX runs straight on Metal; FastAPI + Uvicorn serve HTTP.

> “Why tiptoe when you can OINK?”

---

## Table of Contents

1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Quick start](#quick-start)
4. [Configuration & tuning](#configuration--tuning)
5. [API examples](#api-examples)
6. [Benchmarks](#benchmarks)
7. [Troubleshooting](#troubleshooting)
8. [Roadmap](#roadmap)
9. [License](#license)

---

## Features

| ⚡️ Feature                | Details                                                              |
| ------------------------- | -------------------------------------------------------------------- |
| **OpenAI spec**           | `/v1/models` and `/v1/chat/completions` (*streaming* & *non‑stream*) |
| **One‑file server**       | `oink.py` — 140 ish lines, nothing else                              |
| **MLX backend**           | Loads any HF or local checkpoint that `mlx_lm` can convert/run       |
| **Ahead‑of‑time compile** | `compile=True` + warm‑up jit shrinks per‑request latency             |
| **Quantization ready**    | 4‑bit QLoRA models run in < 5 GB VRAM                                |
| **RAM‑lock helper**       | Script snippet bumps macOS “wired‑memory” limit so nothing swaps     |

---

## Prerequisites

* **macOS 14.4+** (Sonoma) with an Apple‑silicon GPU (M1/M2/M3).
  Tested on **Mac Studio M3 Ultra 512 GB**.
* **Python 3.11+** — newer interpreters cut Metal shader compile time.
* **Xcode Command‑Line Tools** (`xcode-select --install`).

---

## Quick start

```bash
# 1 / Clone & activate venv
$ git clone https://github.com/DavidRagone/oink.git
$ cd oink && python3 -m venv venv
$ source venv/bin/activate && pip install -U pip

# 2 / Install deps (FastAPI + Uvicorn + MLX)
$ pip install "fastapi>=0.111" "uvicorn[standard]>=0.29" "pydantic>=2.7" mlx_lm

# 3 / Choose a model (HF repo or local path)
$ export MODEL_ID=mlx-community/Mistral-7B-Instruct-v0.3-4bit
# or, for example, use Qwen-1.5-7B:
$ export MODEL_ID=mlx-community/Qwen3-30B-A3B-8bit

# 4 / Launch the pig 🚀
$ python oink.py --model $MODEL_ID --host 0.0.0.0 --port 8000
```

Expected log:

```text
[server] Loading model '…' … (first run can take ~10 s)
[server] Model ready → starting API
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

---

## Configuration & tuning

| Option                 | CLI flag / env                                         | Effect                                                                |
| ---------------------- | ------------------------------------------------------ | --------------------------------------------------------------------- |
| **Model path / repo**  | `--model` or `MODEL_ID`                                | Any checkpoint `mlx_lm` understands (HF, local)                       |
| **Host / port**        | `--host`, `--port`                                     | Default `0.0.0.0:8000`                                                |
| **macOS wired memory** | `python oink.py --configure-macos`                     | Runs MLX’s helper script to raise the page‑lock limit to 64 GB        |
| **Workers**            | use `uvicorn` directly: `uvicorn oink:app --workers 4` | Each worker loads its own model copy—512 GB Mac Studio can handle 4–6 |

---

## API examples

### Curl (streaming)

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "'$MODEL_ID'",
    "stream": true,
    "messages": [{"role":"user","content":"Tell me a joke"}]
  }'
```

### OpenAI Python SDK

```python
import openai, os
openai.api_key  = os.getenv("OPENAI_API_KEY", "sk-local")
openai.base_url = "http://localhost:8000/v1"

chat = openai.ChatCompletion.create(
    model=os.environ["MODEL_ID"],
    messages=[{"role": "user", "content": "Ping?"}],
    stream=True,
)
for chunk in chat:
    print(chunk.choices[0].delta.content, end="", flush=True)
```

---

## Benchmarks

| Model (4‑bit)       | VRAM   | Tokens/s | 1st‑token (70 tok prompt) |
| ------------------- | ------ | -------- | ------------------------- |
| Mistral‑7B‑Instruct | 3.9 GB | 110–130  | \~0.35 s                  |
| Qwen‑1.5‑7B         | 4.3 GB | 100–120  | \~0.38 s                  |
| Phi‑3‑mini‑4.2B     | 2.1 GB | 170+     | \~0.25 s                  |

*Numbers from Mac Studio M3 Ultra, macOS 15.5, MLX 0.26. Your mileage may vary.*

---

## Troubleshooting

| Symptom                               | Remedy                                                              |
| ------------------------------------- | ------------------------------------------------------------------- |
| `ModuleNotFoundError: mlx`            | Ensure `pip install mlx_lm` (not just `mlx`).                       |
| Metal “unsupported GPU family”        | Update to macOS 14.4+; M3 GPUs need it.                             |
| `OSError: address already in use`     | Change `--port` or kill the existing process (`lsof -i :8000`).     |
| High first‑token latency after reboot | First call recompiles Metal shaders; subsequent calls will be fast. |

---

## Roadmap / Nice‑to‑haves

* [ ] Embeddings endpoint (`/v1/embeddings`)
* [ ] Function‑calling / tool spec passthrough
* [ ] Simple LoRA hot‑swap via `?adapter=…` query
* [ ] Dockerfile (just for Intel‑Linux bragging rights)

PRs welcome—OINK is deliberately simple but extensibility patches are gladly reviewed.

---

## License

MIT © 2025 David Ragone.
Feel free to fork, extend, and, of course, **OINK** up those CPU cycles.
