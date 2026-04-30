"""Configuration layer — defaults, file overrides, env overrides.

Three layers (later wins):
  1. Hardcoded defaults (in repl.config_loader)
  2. JSON config files (~/.myah/config.json, .myah/config.json, .myah/config.local.json)
  3. Environment variables (MYAH_*, OLLAMA_*, OPENAI_*, etc.)

All existing `from config import X` sites continue to work unchanged.
The file-based config system is opt-in — if no config files exist,
behavior is identical to pre-config-system Myah.

Note on `/config reload`: the slash command re-reads the JSON layers
into the config cache (so `/config` displays the latest values), but
module-level constants here are bound at import time and won't change
mid-session. Restart to pick up file edits in actual harness behavior.
"""
import os

from env import load_dotenv
from repl.config_loader import env_override, get_config

load_dotenv()

# ── File-based base layer ────────────────────────────────────────────────────
_file_cfg = get_config()

_provider_cfg = _file_cfg.get("provider", {})
_models_cfg = _provider_cfg.get("models", {})
_context_cfg = _file_cfg.get("context", {})
_behavior_cfg = _file_cfg.get("behavior", {})
_paths_cfg = _file_cfg.get("paths", {})

# ── Provider ─────────────────────────────────────────────────────────────────
MODEL_PROVIDER = env_override(
    "MYAH_PROVIDER", "provider.default",
    _provider_cfg.get("default", "openai-compat"),
)

OLLAMA_MODEL = _models_cfg.get("ollama", "qwen/qwen3.5-9b")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

OPENAI_COMPAT_MODEL = env_override(
    "OPENAI_COMPAT_MODEL", "provider.models.openai-compat",
    _models_cfg.get("openai-compat", "google/gemma-4-e4b"),
)
OPENAI_COMPAT_BASE_URL = os.environ.get(
    "OPENAI_COMPAT_BASE_URL", "http://127.0.0.1:1234/v1"
)

# Active model name derived from the resolved provider. Hosted providers
# all consult `provider.models.<name>` first, then their dedicated env var,
# so file config and env overrides stay in sync across the matrix.
if MODEL_PROVIDER == "ollama":
    MODEL_NAME = OLLAMA_MODEL
elif MODEL_PROVIDER == "openai-compat":
    MODEL_NAME = OPENAI_COMPAT_MODEL
elif MODEL_PROVIDER == "openai":
    MODEL_NAME = env_override(
        "OPENAI_MODEL", "provider.models.openai",
        _models_cfg.get("openai", "gpt-4.1-mini"),
    )
elif MODEL_PROVIDER == "anthropic":
    MODEL_NAME = env_override(
        "ANTHROPIC_MODEL", "provider.models.anthropic",
        _models_cfg.get("anthropic", "claude-sonnet-4-6"),
    )
elif MODEL_PROVIDER == "deepseek":
    MODEL_NAME = env_override(
        "DEEPSEEK_MODEL", "provider.models.deepseek",
        _models_cfg.get("deepseek", "deepseek-chat"),
    )
elif MODEL_PROVIDER == "google":
    MODEL_NAME = env_override(
        "GOOGLE_MODEL", "provider.models.google",
        _models_cfg.get("google", "gemma-4-e4b"),
    )
elif MODEL_PROVIDER == "opencode":
    MODEL_NAME = env_override(
        "OPENCODE_MODEL", "provider.models.opencode",
        _models_cfg.get("opencode", "opencode/default"),
    )
else:
    MODEL_NAME = OLLAMA_MODEL  # fall back; the factory will raise on first use

# ── Context ──────────────────────────────────────────────────────────────────
NUM_CTX = _context_cfg.get("num_ctx", 32768)
RESERVED_COMPLETION_TOKENS = _context_cfg.get("reserved_completion_tokens", 1024)
TOOL_RESULT_MAX_BYTES = _context_cfg.get("tool_result_max_bytes", 10_000)

# ── Behavior ─────────────────────────────────────────────────────────────────
MAX_COMPLETION_TOKENS = int(env_override(
    "MYAH_MAX_COMPLETION_TOKENS", "behavior.max_completion_tokens",
    _behavior_cfg.get("max_completion_tokens", 4096),
))
MAX_AGENT_ITERATIONS = int(env_override(
    "MYAH_MAX_ITERS", "behavior.max_iterations",
    _behavior_cfg.get("max_iterations", 50),
))
SPIN_WINDOW = 3  # hardcoded — changing this breaks the spinning-call guard tests

# ── Paths ────────────────────────────────────────────────────────────────────
SESSION_FILE = os.path.expanduser(_paths_cfg.get("session_file", "~/.mia_session.json"))
INPUT_HISTORY_FILE = os.path.expanduser(_paths_cfg.get("input_history", "~/.mia_input_history"))
