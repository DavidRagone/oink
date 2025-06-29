# OINK 🐖 — Over‑provisioned Inference ‘N’ Kit

> NOTE: This project is in **early development**. Expect breaking changes and
> limited documentation.

> NOTE: This project relies heavily on LLM-driven development. If that makes you
> uncomfortable, you may want to look elsewhere.

A tiny **OpenAI‑compatible** server that blatantly hogs your M‑series Mac’s resources in the name of *speed*. With a single Python file (`oink.py`) you get:

* **< 1 s first‑token latency** on an M3 Ultra (with 4‑bit 7 B models).
* The standard `/v1/chat/completions` (streaming + non‑stream) and `/v1/models` endpoints.
* Zero external services: MLX runs straight on Metal; FastAPI + Uvicorn serve HTTP.

> “Why tiptoe when you can OINK?”

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick start](#quick-start)
3. [Configuration & tuning](#configuration--tuning)
4. [API examples](#api-examples)
5. [License](#license)

---

## Prerequisites

* **macOS 14.4+** (Sonoma) with an Apple‑silicon GPU (M1/M2/M3).
  Tested on **Mac Studio M3 Ultra 512 GB**.
* **Python 3.11+**

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
# or, for example, use Qwen 8B
$ export MODEL_ID=lmstudio-community/Qwen3-32B-MLX-8bit

# 4 / Launch the pig 🚀
$ python oink.py --model $MODEL_ID
# with  --host and --port to bind to a specific address/port
$ python oink.py --model $MODEL_ID --host 0.0.0.0 --port 8000
```

Expected log:

```text
[server] Loading model '…' … (first run can take ~10 s)
[server] Model ready → starting API
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

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

## License

MIT © 2025 David Ragone.
Feel free to fork, extend, and, of course, **OINK** up those CPU cycles.
