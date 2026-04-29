import os

from env import load_dotenv

load_dotenv()

# Which backend the provider layer talks to. Override via env:
#     MYAH_PROVIDER=anthropic python main.py
# Supported: ollama | openai-compat | openai | anthropic | deepseek
MODEL_PROVIDER = os.environ.get("MYAH_PROVIDER", "openai-compat")

# ── Ollama ────────────────────────────────────────────────────────────────────
OLLAMA_MODEL = "qwen/qwen3.5-9b"
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# ── OpenAI-compatible HTTP (llama.cpp server, LM Studio, vLLM, OpenRouter, …)
OPENAI_COMPAT_MODEL = os.environ.get("OPENAI_COMPAT_MODEL", "google/gemma-4-e4b")
OPENAI_COMPAT_BASE_URL = os.environ.get("OPENAI_COMPAT_BASE_URL", "http://127.0.0.1:1234/v1")
# API key read inside the factory from OPENAI_COMPAT_API_KEY (empty is fine for
# local servers that don't check auth).

# ── Hosted first-party providers ──────────────────────────────────────────────
# These read their own <NAME>_MODEL / <NAME>_API_KEY / <NAME>_BASE_URL env
# vars directly in providers/__init__.py; config.py holds no defaults so we
# don't duplicate the provider presets. Set the relevant env vars and flip
# MYAH_PROVIDER to pick one up:
#     MYAH_PROVIDER=openai    OPENAI_API_KEY=...    OPENAI_MODEL=gpt-4.1-mini
#     MYAH_PROVIDER=anthropic ANTHROPIC_API_KEY=... ANTHROPIC_MODEL=claude-sonnet-4-6
#     MYAH_PROVIDER=deepseek  DEEPSEEK_API_KEY=...  DEEPSEEK_MODEL=deepseek-chat

# Active model name, derived from provider. Both /context and harness_info
# already read this, so the slash/tool surfaces stay provider-agnostic.
# For the hosted providers we resolve against the env var at import time so
# /context shows something sensible before the first turn runs.
if MODEL_PROVIDER == "ollama":
    MODEL_NAME = OLLAMA_MODEL
elif MODEL_PROVIDER == "openai-compat":
    MODEL_NAME = OPENAI_COMPAT_MODEL
elif MODEL_PROVIDER == "openai":
    MODEL_NAME = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
elif MODEL_PROVIDER == "anthropic":
    MODEL_NAME = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
elif MODEL_PROVIDER == "deepseek":
    MODEL_NAME = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
elif MODEL_PROVIDER == "google":
    MODEL_NAME = os.environ.get("GOOGLE_MODEL", "google/gemma-4-e4b")
else:
    MODEL_NAME = OLLAMA_MODEL  # fall back; the factory will raise on first use

# Context window budget — set explicitly so we know the real limit.
# qwen2.5:7b default is 4096; bump to 8192 if your Ollama build supports it.
# For openai-compat this is advisory — most servers ignore num_ctx in the body.
NUM_CTX = 32768

# Extra delay per streamed chunk, in milliseconds. Set to 0 to stream at the
# provider's native rate. A small delay (~8ms) slows output ~30% so it's easier
# to read as it arrives; tune to taste.
STREAM_DELAY_MS = 100

# Cap on tool result size (in characters) before it's appended to the message
# history. Anything bigger gets sliced and a marker is appended. Prevents a
# single `read_file` on a 5MB log from blowing the context window.
TOOL_RESULT_MAX_BYTES = 10_000

# Upper bound on tokens the model may emit in a single response. Sent as
# `max_tokens` in the openai-compat payload. LM Studio and some other local
# servers default this low (512–2048) and silently cut responses mid-sentence;
# explicitly setting it lets the harness own the ceiling. Providers that don't
# take max_tokens (Ollama) ignore it.
MAX_COMPLETION_TOKENS = int(os.environ.get("MYAH_MAX_COMPLETION_TOKENS", "4096"))

# Hard ceiling on the number of provider iterations a single `run_agent`
# call may perform before the loop gives up. A small model that gets stuck
# in a tool-call-tool-call loop would otherwise run until the user hits
# Ctrl-C or the context window explodes. On hit, run_agent returns a
# synthetic assistant message explaining the halt.
MAX_AGENT_ITERATIONS = int(os.environ.get("MYAH_MAX_ITERS", "50"))

# Completion token reserve. trim_history caps the prompt budget at
# num_ctx - RESERVED_COMPLETION_TOKENS so the model always has headroom
# to generate an answer. Prevents "prompt fits but response is cut off".
RESERVED_COMPLETION_TOKENS = 1024

# Window size for the spinning-call guard: if the model emits the same
# (tool_name, args) tuple this many *consecutive* times, the loop halts
# before executing again. Sliding-window — a legitimate re-read of the
# same file inside a longer trajectory does not trigger.
SPIN_WINDOW = 3
