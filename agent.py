# agent.py
import ollama

from config import MODEL_NAME, NUM_CTX


def estimate_tokens(messages: list) -> int:
    """Rough token count: ~1 token per 4 characters of message content."""
    total = sum(len(m.get("content") or "") for m in messages)
    return total // 4


def trim_history(
    history: list, ctx_used: int, num_ctx: int,
    high: float = 0.8, target: float = 0.5,
) -> tuple[list, list]:
    """If ctx is over `high`, drop oldest user/assistant pairs until history
    fits under `target`. Returns (new_history, dropped_messages)."""
    if ctx_used <= high * num_ctx:
        return history, []

    target_tokens = int(target * num_ctx)
    dropped: list = []
    while len(history) >= 2 and estimate_tokens(history) > target_tokens:
        dropped.extend(history[:2])
        history = history[2:]
    return history, dropped


def summarize_dropped(dropped: list) -> str:
    """Compress dropped turns into a terse note via the same model."""
    if not dropped:
        return ""
    transcript = "\n".join(
        f"{m['role']}: {m.get('content', '')}" for m in dropped
    )
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "Summarize the following conversation turns in 2-3 terse sentences. Capture user intent, key facts, and any tool results. No filler.",
            },
            {"role": "user", "content": transcript},
        ],
        options={"num_ctx": NUM_CTX},
    )
    return (response.message.content or "").strip()


def run_agent(user_input: str, tools: list, execute_tool, history: list = []):
    messages = (
        [
            {
                "role": "system",
                "content": f"""You are Mia, a personal assistant.
            You are running on the {MODEL_NAME} model served locally via Ollama.
            You were NOT built by OpenAI or Anthropic. If asked who you are or what model
            you are, answer truthfully based on the line above. Do not claim any other origin.

            Rules:
            - Always use tools when the task requires it
            - After using a tool, always respond with a short confirmation message
            - Never return an empty response
            - For tasks needing multiple steps, do them one at a time""",
            }
        ]
        + history
        + [{"role": "user", "content": user_input}]
    )

    ctx_used = estimate_tokens(messages)

    while True:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            tools=tools,
            options={"num_ctx": NUM_CTX},
        )

        # Ollama returns the real prompt token count; fall back to estimate
        ctx_used = getattr(response, "prompt_eval_count", None) or estimate_tokens(messages)

        message = response.message

        if message.tool_calls:
            messages.append({"role": "assistant", "content": message.content or ""})
            for tool_call in message.tool_calls:
                result = execute_tool(
                    tool_call.function.name, tool_call.function.arguments
                )
                messages.append({"role": "tool", "content": str(result)})
        else:
            if not message.content:
                message.content = "Done."
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": message.content})
            return message.content, history, ctx_used
