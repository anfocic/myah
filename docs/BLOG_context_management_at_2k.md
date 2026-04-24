# Watching context management trigger in a 2K-token budget

> Draft — fill in screenshots and turn-by-turn observations after running the exercise.
>
> **Target:** ~400 words of prose + screenshots. Polish after the run.

<!-- date: 2026-04-21 -->

## Why this post

Most agent-harness writeups describe context management as an abstract concern ("the model has a fixed context window; old turns need to go somewhere"). Mia has four concrete mechanisms for this — `trim_history`, `microcompact`, `apply_summary`, `/compact` — but under a 16K budget running a short session, none of them ever fire. So I decided to force them to fire, watch them do it, and write down exactly what I saw.

The trick was to shrink the budget until the mechanisms engage within a realistic number of turns.

## Setup

<!-- TODO: confirm actual values after running -->

- **Model:** `google/gemma-4-e4b` via LM Studio (OpenAI-compat at `:1234`)
- **`NUM_CTX`:** 2048 tokens (normally 16384)
- **`CLAUDE.md`:** moved aside to `CLAUDE.md.bak` so the ~1.7K-token project context doesn't dominate the system prompt and starve history of budget
- **System prompt size post-setup:** ~400 tokens (persona + env block only)
- **Trim threshold:** 80% = 1,638 tokens. Target after trim: 50% = 1,024 tokens.
- **Microcompact threshold:** 60% = 1,229 tokens, plus ≥4 tool messages in the current turn.

I instrumented `main.py` and `agent/loop.py` so the console prints `ctx before → after` when either mechanism fires — otherwise the screenshot would just show "dropped 2 turns" without the numbers that matter.

## What I saw, turn by turn

<!-- TODO: fill in from the actual run. Run `/profile` between turns and paste the total. -->

| Turn | Prompt (summary) | ctx after | What fired | Notes |
|------|------------------|-----------|------------|-------|
| 1 | "Summarize what this project does in 200 words" | ?? | — | baseline — system prompt ~400 + small user + response |
| 2 | "What does `run_agent` do?" | ?? | — | |
| 3 | "/compact vs trim_history?" | ?? | — | |
| 4 | "Explain microcompact in detail" | ?? | — | |
| 5 | "glob vs grep in this project" | ?? | — | |
| 6 | "Why synthesize tool_call_ids in openai-compat?" | ?? | — | expected trim somewhere around here |
| 7 | "What breaks if I remove patch_stdout?" | ?? | — | |
| 8 | "Recap our conversation so far" | ?? | — | tests whether summary substitution is working |

## The moment `trim_history` fired

<!-- TODO: paste screenshot or terminal block here -->

```
[terminal paste goes here — should include the dim-yellow line:
 ↳ trim_history fired: ctx was XXXX (> threshold 1638 = 80% of NUM_CTX=2048);
   dropped N turn(s), summarized into context; will re-settle after next provider call
]
```

What actually happened, in order:

1. Turn N finished. `run_agent` returned `ctx_used = XXXX` from LM Studio's reported usage.
2. `main.py` called `trim_history(history, ctx_used, NUM_CTX)`. The function saw `ctx_used > 0.8 * NUM_CTX` and entered the while-loop.
3. It repeatedly dropped oldest user/assistant pairs from `history` until `estimate_tokens(history) <= 0.5 * NUM_CTX`. Returned `(new_history, dropped)`.
4. `main.py` saw `dropped` was non-empty, called `summarize_dropped()` — which is a blocking LLM call against the same provider, asking it to compress the dropped turns into 2–3 sentences.
5. That summary got prepended as a synthetic `system` message. The instrumented print fired.
6. Next user message will settle the real token count against the new (shorter) history.

## The moment `microcompact` fired

<!-- TODO: fill in if Gemma's tool calling works; otherwise note "did not reproduce" -->

<!-- If microcompact did not fire, document why:
  - Gemma 4 E4B did not reliably emit multiple tool_calls in a single turn through LM Studio's OpenAI-compat shim
  - OR budget got trimmed by trim_history before tool-heavy turn landed
  - OR the specific prompt didn't induce multi-file reads
-->

## What the code actually does

- `trim_history` → `agent/context.py:19` — reactive, fires post-turn in `main.py:130`
- `apply_summary` → `agent/context.py:55` — summarization via `summarize_dropped` (another provider call)
- `microcompact` → `agent/context.py:78` — intra-turn, fires at top of `run_agent`'s while-loop (`agent/loop.py:59`)
- The full pedagogical breakdown: `docs/BUILD_NOTES.md` §6 (trim), §7 (summarize), §33 (/compact), §35 (microcompact)

## Surprises

<!-- TODO: write 2–4 observations. Seed ideas: -->

- **Char/4 token estimator diverged ~40% from the real tokenizer.** `/profile` (char/4) said X tokens; `/context` (real usage from LM Studio) said Y. That's the delta between `len(content)//4` and whatever Gemma's actual BPE tokenizer did. Worth calibrating.
- **System prompt size is load-bearing.** Leaving CLAUDE.md in at `NUM_CTX=2048` makes the harness unusable — the persistent system prompt alone exceeds the budget. `trim_history` can only drop history turns, not the system prompt. Worth noting: at small budgets, the system-prompt architecture choice (what goes in CLAUDE.md, what's fetched on-demand) matters more than any context-management mechanism.
- **`trim_history` → `summarize_dropped` is a blocking LLM round-trip.** The user sees a "Summarizing dropped turns..." status line; the next turn starts only after that summary call completes. Not free.
- <!-- TODO: anything else you noticed -->

## Takeaways

<!-- TODO: 2–3 honest statements, ideally with a "this changed my mental model" flavor -->

1. **Context management is a layered system, not a single switch.** Each mechanism catches a different failure mode. You only notice you need the next layer when the previous one starts to leak.
2. <!-- TODO -->
3. <!-- TODO -->

## Reproducing this

```bash
# Shrink system prompt
mv CLAUDE.md CLAUDE.md.bak

# In config.py: set NUM_CTX = 2048
# Run:
python main.py

# Hold the 8-turn conversation above. Paste /profile before each turn.
# Screenshot the dim-yellow trim/microcompact lines when they fire.

# Cleanup
mv CLAUDE.md.bak CLAUDE.md
# In config.py: restore NUM_CTX = 16384
```

Total session time: ~70 minutes including writeup.

---

<!-- When finalising:
     - Remove all <!-- TODO --> markers
     - Ensure prose flows (not just tables + bullets)
     - 400 words in the body; let screenshots carry visual weight
     - Link to this post from docs/BLOG.md's TOC if you keep one
-->
