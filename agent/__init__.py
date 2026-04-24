"""Mia's agent loop layer.

Each module owns one concern:

- `tokens`         — token-count estimate + tool-result size cap
- `status`         — spinner status-line + JSONL logging per turn
- `context`        — trim/compact/microcompact/summarize — §6/§7/§33/§35
- `system_prompt`  — base persona + env block + CLAUDE.md + plan-mode rules
- `loop`           — `run_agent`, the core loop + parallel tool execution

`READ_ONLY_TOOLS` lives here (not in `loop` or `system_prompt`) because
both of those import it — top-level breaks the cycle."""
# Tools the plan-mode gate lets through unchanged. Everything else gets
# short-circuited so the model can investigate (glob/grep/read) while
# planning, but can't mutate state until plan mode is turned off.
READ_ONLY_TOOLS = frozenset(
    {"glob", "grep", "read_file", "get_current_time", "harness_info", "web_search"}
)


# Public surface — re-export so existing `from agent import X` callers
# keep working without caring about the new internal layout.
from agent.context import (
    COMPACT_KEEP_LAST,
    ELIDED_PREFIX,
    MICROCOMPACT_CTX_THRESHOLD,
    MICROCOMPACT_KEEP_RECENT,
    apply_summary,
    compact_history,
    microcompact,
    summarize_dropped,
    trim_history,
)
from agent.loop import _debug_dump_messages, _run_tools_parallel, run_agent
from agent.status import LOG_FILE, log_response, status_line
from agent.system_prompt import build_system_prompt
from agent.tokens import estimate_tokens, truncate_tool_result

__all__ = [
    "READ_ONLY_TOOLS",
    "COMPACT_KEEP_LAST",
    "ELIDED_PREFIX",
    "LOG_FILE",
    "MICROCOMPACT_CTX_THRESHOLD",
    "MICROCOMPACT_KEEP_RECENT",
    "apply_summary",
    "build_system_prompt",
    "compact_history",
    "estimate_tokens",
    "log_response",
    "microcompact",
    "run_agent",
    "status_line",
    "summarize_dropped",
    "trim_history",
    "truncate_tool_result",
    # Private helpers re-exported so tests can reach them via `from agent
    # import ...`. Not part of the stable surface, hence the _ prefix.
    "_debug_dump_messages",
    "_run_tools_parallel",
]
