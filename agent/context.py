"""Context-window management — the four mechanisms that keep the prompt
under budget:

- `trim_history`  (§6) — reactive: drops oldest turns when ctx_used > high.
- `compact_history` (§33) — proactive: keeps last N turns, rest is caller's
  problem (typically fed to summarize_dropped + apply_summary).
- `microcompact`  (§35) — intra-turn: elides older tool-result bodies when
  a tool-heavy turn piles up 10+ results.
- `summarize_dropped` (§7) — the LLM call that turns dropped turns into a
  2-3 sentence system note; `apply_summary` prepends it.

Each layer catches a different failure mode; per CONCEPTS §35 you only
notice you need the next one when the previous starts to leak."""
from agent.tokens import count_tokens
from config import RESERVED_COMPLETION_TOKENS, get_context_size
from providers import ProviderError, get_active_provider


def trim_history(
    history: list, ctx_used: int, num_ctx: int,
    high: float = 0.8, target: float = 0.5,
    tools: list | None = None, model_name: str | None = None,
) -> tuple[list, list]:
    """If ctx is over `high`, drop oldest user/assistant pairs until history
    fits under `target`. Returns (new_history, dropped_messages).

    The effective target is reduced by RESERVED_COMPLETION_TOKENS so the
    model always has headroom to generate a response.

    Pass `tools` and `model_name` so the inner count matches what the
    gate-check `ctx_used` measured — without them, the loop undercounts
    by the tool-schema budget (~3-5K tokens) and stops too early."""
    if ctx_used <= high * num_ctx:
        return history, []

    target_tokens = int(target * num_ctx) - RESERVED_COMPLETION_TOKENS
    if target_tokens < 0:
        target_tokens = 0
    dropped: list = []
    while len(history) >= 2 and count_tokens(history, tools=tools, model_name=model_name) > target_tokens:
        dropped.extend(history[:2])
        history = history[2:]
    return history, dropped


COMPACT_KEEP_LAST = 2  # turns retained after manual /compact — one turn
                       # of continuity + the current user message's runway


def compact_history(
    history: list, keep_last: int = COMPACT_KEEP_LAST,
) -> tuple[list, list]:
    """Manual compact: keep the last `keep_last` user/assistant pairs, drop
    the rest. Caller is expected to summarize the dropped turns and re-insert
    the summary. Distinct from `trim_history` — that's reactive (fires on
    pressure); this is proactive (fires when the user says so)."""
    keep_msgs = keep_last * 2
    if len(history) <= keep_msgs:
        return history, []
    dropped = list(history[:-keep_msgs]) if keep_msgs else list(history)
    new_history = list(history[-keep_msgs:]) if keep_msgs else []
    return new_history, dropped


def apply_summary(history: list, dropped: list) -> list:
    """Summarize `dropped` and prepend the summary as a system note to
    `history`. If summarization fails or returns empty, `history` is
    returned unchanged so the caller can decide how to announce the outcome.
    Used by both manual /compact and auto-trim — one canonical shape for
    post-compaction history."""
    summary = summarize_dropped(dropped)
    if not summary:
        return history
    return [
        {
            "role": "system",
            "content": f"Summary of earlier conversation: {summary}",
        },
        *history,
    ]


MICROCOMPACT_KEEP_RECENT = 3
MICROCOMPACT_CTX_THRESHOLD = 0.6
ELIDED_PREFIX = "[tool result elided"


def microcompact(messages: list, keep_recent: int = MICROCOMPACT_KEEP_RECENT) -> int:
    """Elide old tool-result bodies in the live `messages` list. Mutates
    in-place; returns the number of messages rewritten. Tool messages stay
    in the array (the assistant's tool_calls still reference them), only
    their `content` gets replaced with a tiny stub.

    Intra-turn counterpart to `trim_history`: history-level trim can't reach
    tool results because history only holds user/assistant pairs (§3). A
    turn that does 10 read_files accumulates 10 fat tool messages that only
    microcompact can shrink."""
    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    if len(tool_indices) <= keep_recent:
        return 0
    to_elide = tool_indices[:-keep_recent]
    n = 0
    for i in to_elide:
        original = messages[i].get("content") or ""
        if original.startswith(ELIDED_PREFIX):
            continue  # already elided — don't re-stamp
        messages[i] = {
            "role": "tool",
            "content": f"{ELIDED_PREFIX} — {len(original)} chars]",
        }
        n += 1
    return n


def _user_text(content) -> str:
    """Best-effort text extraction from a user message's content. Handles
    plain strings and the list-of-blocks shape used by image attachments
    (returns concatenated text blocks only, image blocks ignored)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text") or "")
        return " ".join(p for p in parts if p)
    return ""


def _extractive_summary(dropped: list) -> str:
    """Build a cheap local summary from dropped turns when the LLM call
    fails. Captures first sentence of each user message so context isn't
    completely lost.

    History only persists `{role, content}` (tool_calls are intra-turn
    scratch, never appended), so there's no point trying to extract tool
    names from `dropped` — they were never there to begin with."""
    user_snippets: list[str] = []
    for m in dropped:
        if m.get("role") != "user":
            continue
        text = _user_text(m.get("content", ""))
        # First sentence: up to first period, or first 80 chars.
        sentence = text.split(".")[0].strip()
        if len(sentence) > 80:
            sentence = sentence[:77] + "..."
        if sentence:
            user_snippets.append(sentence)
    if not user_snippets:
        return ""
    return "User asked about " + "; ".join(user_snippets) + "."


def summarize_dropped(dropped: list) -> str:
    """Compress dropped turns into a terse note via the active provider.
    Falls back to an extractive local summary on ProviderError so context
    is never completely lost."""
    if not dropped:
        return ""
    transcript = "\n".join(
        f"{m['role']}: {m.get('content', '')}" for m in dropped
    )
    try:
        content, _ = get_active_provider().chat(
            messages=[
                {
                    "role": "system",
                    "content": "Summarize the following conversation turns in 2-3 terse sentences. Capture user intent, key facts, and any tool results. No filler.",
                },
                {"role": "user", "content": transcript},
            ],
            num_ctx=get_context_size(),
        )
    except ProviderError:
        return _extractive_summary(dropped)
    return content
