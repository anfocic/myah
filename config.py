"""Configuration layer — defaults, file overrides, env overrides.

Three layers (later wins):
  1. Hardcoded defaults (in repl.config_loader)
  2. JSON config files (~/.myah/config.json, .myah/config.json, .myah/config.local.json)
  3. Environment variables (MYAH_*, OLLAMA_*, OPENAI_*, etc.)

All existing `from config import X` sites continue to work unchanged.
The file-based config system is opt-in — if no config files exist,
behavior is identical to pre-config-system Myah.
"""
import os

from env import load_dotenv
from repl.config_loader import get_config

load_dotenv()

# ── File-based base layer ────────────────────────────────────────────────────
_file_cfg = get_config()

_provider_cfg = _file_cfg.get("provider", {})
_context_cfg = _file_cfg.get("context", {})
_behavior_cfg = _file_cfg.get("behavior", {})
_paths_cfg = _file_cfg.get("paths", {})

# ── Provider ─────────────────────────────────────────────────────────────────
MODEL_PROVIDER = os.environ.get("MYAH_PROVIDER", _provider_cfg.get("default", "openai-compat"))

OLLAMA_MODEL = _provider_cfg.get("models", {}).get("ollama", "qwen/qwen3.5-9b")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

OPENAI_COMPAT_MODEL = os.environ.get(
    "OPENAI_COMPAT_MODEL",
    _provider_cfg.get("models", {}).get("openai-compat", "google/gemma-4-e4b"),
)
OPENAI_COMPAT_BASE_URL = os.environ.get(
    "OPENAI_COMPAT_BASE_URL", "http://127.0.0.1:1234/v1"
)

# Active model name derived from the resolved provider.
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
elif MODEL_PROVIDER == "opencode":
    MODEL_NAME = os.environ.get("OPENCODE_MODEL", "opencode/default")
else:
    MODEL_NAME = OLLAMA_MODEL  # fall back; the factory will raise on first use

# ── Context ──────────────────────────────────────────────────────────────────
NUM_CTX = _context_cfg.get("num_ctx", 32768)
RESERVED_COMPLETION_TOKENS = _context_cfg.get("reserved_completion_tokens", 1024)
TOOL_RESULT_MAX_BYTES = _context_cfg.get("tool_result_max_bytes", 10_000)

# ── Behavior ─────────────────────────────────────────────────────────────────
STREAM_DELAY_MS = _behavior_cfg.get("stream_delay_ms", 100)
MAX_COMPLETION_TOKENS = int(
    os.environ.get("MYAH_MAX_COMPLETION_TOKENS", str(_behavior_cfg.get("max_completion_tokens", 4096)))
)
MAX_AGENT_ITERATIONS = int(
    os.environ.get("MYAH_MAX_ITERS", str(_behavior_cfg.get("max_iterations", 50)))
)
SPIN_WINDOW = 3  # hardcoded — changing this breaks the spinning-call guard tests

# ── Paths ────────────────────────────────────────────────────────────────────
SESSION_FILE = os.path.expanduser(_paths_cfg.get("session_file", "~/.mia_session.json"))
INPUT_HISTORY_FILE = os.path.expanduser(_paths_cfg.get("input_history", "~/.mia_input_history"))
