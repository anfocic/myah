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
_ui_cfg = _file_cfg.get("ui", {})

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
        _models_cfg.get("opencode", "kimi-k2.6"),
    )
elif MODEL_PROVIDER == "openrouter":
    MODEL_NAME = env_override(
        "OPENROUTER_MODEL", "provider.models.openrouter",
        _models_cfg.get("openrouter", "nousresearch/hermes-3-llama-3.1-405b"),
    )
else:
    MODEL_NAME = OLLAMA_MODEL  # fall back; the factory will raise on first use

# ── Context ──────────────────────────────────────────────────────────────────
NUM_CTX = _context_cfg.get("num_ctx", 32768)


def get_context_size() -> int:
    """Return the active provider's context window size.

    Falls back to NUM_CTX when no provider has been initialized yet
    (early import, test fixtures that bypass `set_active_provider`).
    Hosted providers report their own native window (e.g. OpenAI = 128K,
    Anthropic = 200K); local providers use the user-configured NUM_CTX.

    Narrow exception list on purpose: a real bug in the provider should
    propagate, not silently degrade context-management decisions to the
    NUM_CTX default."""
    try:
        from providers import get_active_provider
    except ImportError:
        # `providers` failed to import — typically the test process tearing
        # down its sys.modules, or a broken install. Fall back rather than
        # crash the REPL footer.
        return NUM_CTX
    try:
        return get_active_provider().context_size
    except (RuntimeError, AttributeError):
        # RuntimeError: no provider active yet (lazy-init path before
        # set_active_provider has run). AttributeError: a provider that
        # somehow lacks the `context_size` attr (test stub, or an adapter
        # added without updating the Protocol).
        return NUM_CTX


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

# Transient provider failures (network unreachable, timeout, HTTP 429 / 5xx)
# get a bounded retry budget per turn. Higher counts mostly help recover from
# rate-limit bursts on hosted providers; local servers either work or don't.
MAX_PROVIDER_RETRIES = int(env_override(
    "MYAH_MAX_PROVIDER_RETRIES", "behavior.max_provider_retries",
    _behavior_cfg.get("max_provider_retries", 3),
))
# Exponential backoff base: sleep(attempt) = BASE * (2 ** (attempt - 1)) + jitter.
# 1.0 means 1s / 2s / 4s — gentle enough for transient 5xx, short enough that
# the user notices the wait but doesn't bail.
PROVIDER_RETRY_BASE_S = float(env_override(
    "MYAH_PROVIDER_RETRY_BASE_S", "behavior.provider_retry_base_s",
    _behavior_cfg.get("provider_retry_base_s", 1.0),
))

# ── UI ───────────────────────────────────────────────────────────────────────
# Phosphor accent hue. The whole TUI reads its dominant color from this one
# knob — masthead, section brackets, prompt floor, tool glyphs. Themes
# (dark/light/hc) are intentionally *not* a knob: rich's named ANSI colors
# already track the user's terminal palette, so "max compat" is free.
_VALID_ACCENTS = {"green", "amber", "cyan"}
PHOSPHOR_ACCENT = env_override(
    "MYAH_ACCENT", "ui.accent", _ui_cfg.get("accent", "green"),
)
if PHOSPHOR_ACCENT not in _VALID_ACCENTS:
    PHOSPHOR_ACCENT = "green"

# ── Paths ────────────────────────────────────────────────────────────────────
SESSION_FILE = os.path.expanduser(_paths_cfg.get("session_file", "~/.mia_session.json"))
INPUT_HISTORY_FILE = os.path.expanduser(_paths_cfg.get("input_history", "~/.mia_input_history"))

# ── Personal vault ───────────────────────────────────────────────────────────
# The user's Obsidian vault that Mia reads and writes via the `note_*` tools.
# Distinct from `vault/` in this repo, which is the project's dev knowledge
# base reached through `vault_search`.
_vault_cfg = _file_cfg.get("vault", {})
MIA_VAULT_PATH = os.path.expanduser(
    env_override("MIA_VAULT_PATH", "vault.path", _vault_cfg.get("path", "~/MiaVault"))
)
MIA_DAILY_DIR = _vault_cfg.get("daily_dir", "daily")
