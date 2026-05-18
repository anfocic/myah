"""End-to-end `run_agent` test with a scripted FakeProvider.

Exercises the full agentic loop: model emits content chunks, optionally
tool calls, optionally another turn. No live provider required — the
FakeProvider replays canned turns so the test is deterministic and fast.

This is the test you reach for when refactoring the loop itself —
per-feature unit tests stay green while you're quietly breaking the
stitching between context-management / tool-dispatch / streaming."""
from collections.abc import Callable
from dataclasses import dataclass

import pytest

from agent import run_agent
from providers import ProviderError, Usage, set_active_provider
from providers.base import StreamChunk, ToolCall


@dataclass
class ScriptedTurn:
    """One model turn: a sequence of content chunks plus optionally a list
    of tool calls the model emits at the end. `usage` is what the provider
    would report on `done=True`."""
    content_chunks: list[str]
    tool_calls: list[ToolCall] | None = None
    prompt_tokens: int = 100
    completion_tokens: int = 20
    # Reasoning chunks surfaced on a separate stream channel (LM Studio's
    # `reasoning_content`). Yielded before `content_chunks` so the ordering
    # matches what real reasoning-capable servers do (think first, then
    # emit the visible reply).
    reasoning_chunks: list[str] | None = None


class FakeProvider:
    """Replays scripted turns. On each `stream_chat` call, pops the next
    turn off the script and yields its chunks followed by a final
    `done=True` chunk with usage."""
    name = "fake"
    model = "fake-model-v1"
    context_size = 32768

    def __init__(self, script: list[ScriptedTurn]):
        self._script = list(script)

    def stream_chat(self, messages, tools, num_ctx):
        if not self._script:
            raise AssertionError(
                "FakeProvider ran out of scripted turns — the loop called "
                "stream_chat more times than the test expected"
            )
        turn = self._script.pop(0)
        for chunk in turn.reasoning_chunks or []:
            yield StreamChunk(reasoning_delta=chunk)
        for chunk in turn.content_chunks:
            yield StreamChunk(content_delta=chunk)
        # Tool calls arrive in a single chunk (matches Ollama shape; OpenAI
        # adapter buffers deltas internally and emits one completed call).
        if turn.tool_calls:
            yield StreamChunk(tool_calls=turn.tool_calls)
        yield StreamChunk(
            done=True,
            usage=Usage(
                prompt_tokens=turn.prompt_tokens,
                completion_tokens=turn.completion_tokens,
            ),
        )

    def chat(self, messages, num_ctx):
        raise NotImplementedError("scripted FakeProvider only supports streaming")

    def count_tokens(self, messages, tools=None):
        # Stub to satisfy the Provider protocol — tests that care about
        # counting set their own fake before calling. Kept as an explicit
        # NotImplementedError so any accidental use surfaces loudly.
        raise NotImplementedError("scripted FakeProvider does not count tokens")


@pytest.fixture
def install_provider():
    """Install a FakeProvider for the duration of one test, restoring the
    previous active provider on teardown. Pytest fixtures generate a clean
    test every run even if the previous test left `set_active_provider`
    pointing at a fake."""
    from providers import get_active_provider
    original = get_active_provider()

    def _install(script: list[ScriptedTurn]) -> FakeProvider:
        p = FakeProvider(script)
        set_active_provider(p)
        return p

    yield _install
    set_active_provider(original)


def _noop_execute_tool(name: str, args: dict) -> str:
    return f"fake tool result for {name}({args})"


def _execute_tool_factory(responses: dict[str, str]) -> Callable:
    """Build an execute_tool that returns canned strings per tool name."""
    def execute(name: str, args: dict) -> str:
        return responses.get(name, f"no canned response for {name}")
    return execute


def test_single_turn_no_tools(install_provider):
    """Baseline: model responds with content, no tool calls, one turn."""
    install_provider([
        ScriptedTurn(content_chunks=["Hello ", "world!"]),
    ])

    response, history, ctx_used, stats = run_agent(
        user_input="hi",
        tools=[],
        execute_tool=_noop_execute_tool,
        history=[],
        # permission_check=None is fine — no tools means the gate never fires
    )

    assert response == "Hello world!"
    assert len(history) == 2  # user + assistant
    assert history[0] == {"role": "user", "content": "hi"}
    assert history[1] == {"role": "assistant", "content": "Hello world!"}
    assert ctx_used == 100  # prompt_tokens from Usage
    assert stats["completion_tokens"] == 20


def test_tool_use_then_final_answer(install_provider):
    """The classic two-turn tool-use flow: model calls read_file, harness
    executes it, result goes back, model produces final answer."""
    install_provider([
        # Turn 1: model emits a tool call (no content)
        ScriptedTurn(
            content_chunks=[],
            tool_calls=[ToolCall(name="read_file", arguments={"path": "x.py"})],
        ),
        # Turn 2: model responds with content based on tool result
        ScriptedTurn(
            content_chunks=["The file says hello."],
        ),
    ])

    # Permit the tool call; execute_tool returns a canned string
    def allow(name, args):
        return True

    response, history, ctx_used, stats = run_agent(
        user_input="read x.py",
        tools=[],
        execute_tool=_execute_tool_factory({"read_file": "contents of x.py"}),
        history=[],
        permission_check=allow,
    )

    assert response == "The file says hello."
    # History only contains the final user/assistant pair — tool messages
    # are intermediate work, not conversation (§3).
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "The file says hello."


def test_permission_denial_is_returned_as_tool_result(install_provider):
    """When permission_check returns False, the tool result sent back to
    the model is the denial message, not an exception. The loop proceeds
    to a second turn normally."""
    install_provider([
        ScriptedTurn(
            content_chunks=[],
            tool_calls=[ToolCall(name="write_file", arguments={"path": "x", "content": "y"})],
        ),
        ScriptedTurn(
            content_chunks=["Understood, I won't write."],
        ),
    ])

    executed: list[str] = []

    def tracking_execute(name, args):
        executed.append(name)
        return "ran"

    def deny_all(name, args):
        return False

    response, history, _, _ = run_agent(
        user_input="write a file",
        tools=[],
        execute_tool=tracking_execute,
        history=[],
        permission_check=deny_all,
    )

    # The tool was NOT executed — the denial short-circuited it
    assert executed == []
    # But the loop still progressed to a second turn and got a final answer
    assert response == "Understood, I won't write."


def test_multiple_tool_calls_in_one_turn(install_provider):
    """Model can emit multiple tool_calls in a single turn; they're
    executed (in parallel under the hood, §25) and all results sent back."""
    install_provider([
        ScriptedTurn(
            content_chunks=[],
            tool_calls=[
                ToolCall(name="glob", arguments={"pattern": "*.py"}),
                ToolCall(name="grep", arguments={"pattern": "TODO"}),
            ],
        ),
        ScriptedTurn(
            content_chunks=["Scanned."],
        ),
    ])

    executed: list[str] = []

    def tracking_execute(name, args):
        executed.append(name)
        return f"{name} done"

    def allow(name, args):
        return True

    response, history, _, _ = run_agent(
        user_input="scan",
        tools=[],
        execute_tool=tracking_execute,
        history=[],
        permission_check=allow,
    )

    # Both tools ran (order may vary since execution is parallel)
    assert set(executed) == {"glob", "grep"}
    assert response == "Scanned."


def test_empty_content_final_turn_gets_default(install_provider):
    """If the model produces an empty final reply (no content, no tool
    calls), the loop substitutes "Done." so history never holds an empty
    assistant message."""
    install_provider([
        ScriptedTurn(content_chunks=[]),
    ])

    response, history, _, _ = run_agent(
        user_input="ok",
        tools=[],
        execute_tool=_noop_execute_tool,
        history=[],
    )

    assert response == "Done."
    assert history[1]["content"] == "Done."


def test_reasoning_stream_is_kept_out_of_content_and_history(install_provider):
    """Reasoning deltas must never leak into the assistant's visible reply
    or into the message history. If they did, every subsequent turn would
    carry stale chain-of-thought that drags model quality down (and on
    qwen3 specifically, echoing old `<think>` traces back into the context
    confuses the template). The loop's contract: reasoning is for
    observability only — rendered live and logged, but stripped from
    content and history."""
    install_provider([
        ScriptedTurn(
            reasoning_chunks=["I should ", "greet the user."],
            content_chunks=["Hello!"],
        ),
    ])

    response, history, _ctx, _stats = run_agent(
        user_input="hi",
        tools=[],
        execute_tool=_noop_execute_tool,
        history=[],
    )

    assert response == "Hello!"
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "Hello!"
    assert "should" not in history[1]["content"]
    assert "greet" not in history[1]["content"]


def test_reasoning_without_content_still_defaults_to_done(install_provider):
    """Edge case: some qwen3 turns emit reasoning and then hit max_tokens
    before writing any `content`. The loop's existing empty-content
    fallback (defaults to 'Done.') must still apply — we can't surface
    the reasoning as the reply because it would pollute history, and
    returning "" would break callers that assume a non-empty final
    message (e.g. the REPL prints it)."""
    install_provider([
        ScriptedTurn(
            reasoning_chunks=["thinking hard", " and harder"],
            content_chunks=[],
        ),
    ])

    response, history, _ctx, _stats = run_agent(
        user_input="ok",
        tools=[],
        execute_tool=_noop_execute_tool,
        history=[],
    )

    assert response == "Done."
    assert history[1]["content"] == "Done."
    assert "thinking" not in history[1]["content"]


def test_reasoning_total_is_per_call_not_module_global(install_provider):
    """A previous turn's reasoning must not bleed into the next run_agent
    call's stats. If reasoning_total lived at module scope instead of
    local-to-run_agent, every invocation would grow the joined trace, and
    eval reports would carry reasoning from unrelated earlier tasks."""
    install_provider([
        ScriptedTurn(reasoning_chunks=["first-call reasoning"], content_chunks=["A"]),
        ScriptedTurn(reasoning_chunks=["second-call reasoning"], content_chunks=["B"]),
    ])

    _, _, _, stats_one = run_agent(
        user_input="one",
        tools=[],
        execute_tool=_noop_execute_tool,
        history=[],
    )
    _, _, _, stats_two = run_agent(
        user_input="two",
        tools=[],
        execute_tool=_noop_execute_tool,
        history=[],
    )

    assert "first-call reasoning" in stats_one["reasoning"]
    assert "first-call reasoning" not in stats_two["reasoning"]
    assert stats_two["reasoning"] == "second-call reasoning"


def test_provider_error_is_reported_in_stats():
    """Provider failures should remain machine-readable for callers like evals.

    The REPL can still render the human-facing error and keep going, but an
    eval runner must not confuse transport failure with a valid empty model
    response.
    """
    from providers import get_active_provider

    class FailingProvider:
        name = "fake-failing"
        model = "fake-failing-v1"

        def stream_chat(self, messages, tools, num_ctx):
            raise ProviderError("simulated outage")
            yield  # pragma: no cover - makes this a generator

        def chat(self, messages, num_ctx):
            raise NotImplementedError

        def count_tokens(self, messages, tools=None):
            return 0

    original = get_active_provider()
    set_active_provider(FailingProvider())
    try:
        response, history, _ctx, stats = run_agent(
            user_input="hi",
            tools=[],
            execute_tool=_noop_execute_tool,
            history=[],
        )
    finally:
        set_active_provider(original)

    assert response == ""
    assert history == []
    assert stats["provider_error"] == "fake-failing: simulated outage"


# ---------- loop guards: iteration cap + spinning detection ----------

def _looping_tool_script(n: int, call: ToolCall) -> list[ScriptedTurn]:
    """Script `n` turns, each a bare tool call to `call`. Used by the
    iter-cap test: the model never surrenders, so only the cap halts it."""
    return [
        ScriptedTurn(content_chunks=[], tool_calls=[call])
        for _ in range(n)
    ]


def test_loop_halts_at_iteration_cap(install_provider, monkeypatch):
    """If the model never emits content (always tool-only turns), the loop
    must not run forever. The cap is a hard upper bound; on hit, the loop
    returns with `stats['halt_reason'] == 'iter_cap'` and a synthetic
    assistant message explaining the halt."""
    # Shrink the cap for the test so we don't need 50 scripted turns.
    import agent.loop as loop_mod
    monkeypatch.setattr(loop_mod, "MAX_AGENT_ITERATIONS", 3)

    # Alternate between two distinct calls so the spin guard never fires;
    # the only thing that can halt this script is the iteration cap.
    alt_calls = [
        ToolCall(name="read_file", arguments={"path": "a.py"}),
        ToolCall(name="read_file", arguments={"path": "b.py"}),
    ]
    install_provider([
        ScriptedTurn(content_chunks=[], tool_calls=[alt_calls[i % 2]])
        for i in range(5)
    ])

    response, history, _ctx, stats = run_agent(
        user_input="read it",
        tools=[],
        execute_tool=_execute_tool_factory({"read_file": "contents"}),
        history=[],
        permission_check=lambda n, a: True,
    )

    assert stats["halt_reason"] == "iter_cap"
    assert response.startswith("[halted: iter_cap]")
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == response


def test_loop_halts_on_spinning(install_provider):
    """Three consecutive identical tool calls should trigger the spin guard
    before the fourth runs. The synthetic assistant message names the
    repeated call so the user knows which call stuck."""
    repeated = ToolCall(name="read_file", arguments={"path": "missing.py"})
    # Script 5 identical turns so there's no ambiguity about the cap firing.
    install_provider(_looping_tool_script(5, repeated))

    response, history, _ctx, stats = run_agent(
        user_input="read it",
        tools=[],
        execute_tool=_execute_tool_factory({"read_file": "file not found"}),
        history=[],
        permission_check=lambda n, a: True,
    )

    assert stats["halt_reason"] == "spinning"
    assert response.startswith("[halted: spinning]")
    assert "read_file" in response
    assert len(history) == 2


def test_loop_does_not_halt_on_legitimate_repeats(install_provider):
    """Re-reading the same file inside a longer trajectory must NOT trigger
    the spin guard. Only three *consecutive* identical calls do."""
    call_a = ToolCall(name="read_file", arguments={"path": "a.py"})
    call_b = ToolCall(name="read_file", arguments={"path": "b.py"})
    install_provider([
        ScriptedTurn(content_chunks=[], tool_calls=[call_a]),
        ScriptedTurn(content_chunks=[], tool_calls=[call_b]),
        ScriptedTurn(content_chunks=[], tool_calls=[call_a]),
        ScriptedTurn(content_chunks=["Done reading."]),
    ])

    response, _history, _ctx, stats = run_agent(
        user_input="read a and b",
        tools=[],
        execute_tool=_execute_tool_factory({"read_file": "ok"}),
        history=[],
        permission_check=lambda n, a: True,
    )

    assert response == "Done reading."
    assert stats.get("halt_reason") is None


# ---------- _stream_provider_turn unit tests ----------
#
# Hit `_stream_provider_turn` directly so the four chunk channels + three
# exit paths can be exercised without tool dispatch or message assembly.
# See vault/wiki/plans/plan-stream-session-extraction.md for the
# motivation.

import time as _time  # noqa: E402  intentional shadow-free alias for local use

from agent.loop import _stream_provider_turn  # noqa: E402


def _consume(script: list[ScriptedTurn]):
    """Build a FakeProvider on `script` and run one turn through the
    extracted stream helper. Returns the `TurnResult`."""
    provider = FakeProvider(script)
    return _stream_provider_turn(
        provider,
        messages=[{"role": "user", "content": "x"}],
        tools=[],
        num_ctx=4096,
        console=None,
        start_time=_time.time(),
    )


def test_stream_processes_content_deltas_into_single_string():
    result = _consume([ScriptedTurn(content_chunks=["Hello ", "there!"])])
    assert result.content == "Hello there!"
    assert result.tool_calls == []
    assert result.provider_error is None


def test_stream_separates_reasoning_from_content():
    """Both streams accumulate independently and land on distinct fields."""
    result = _consume([
        ScriptedTurn(
            reasoning_chunks=["step 1 ", "step 2"],
            content_chunks=["visible reply"],
        ),
    ])
    assert result.reasoning == "step 1 step 2"
    assert result.content == "visible reply"
    # Reasoning is never smeared into content.
    assert "step" not in result.content


def test_stream_buffers_tool_calls_until_done():
    call = ToolCall(name="read_file", arguments={"path": "x.py"})
    result = _consume([
        ScriptedTurn(content_chunks=[], tool_calls=[call]),
    ])
    assert result.tool_calls == [call]
    assert result.content == ""


def test_stream_captures_usage_from_done_chunk():
    result = _consume([
        ScriptedTurn(
            content_chunks=["hi"],
            prompt_tokens=77,
            completion_tokens=3,
        ),
    ])
    assert result.usage is not None
    assert result.usage.prompt_tokens == 77
    assert result.usage.completion_tokens == 3


def test_stream_provider_error_is_returned_not_raised():
    """ProviderError is caught and surfaced on the TurnResult so the
    caller can build the provider_error stats entry without a try/except."""
    class FailingProvider:
        name = "fake-failing"
        model = "fake-failing-v1"

        def stream_chat(self, messages, tools, num_ctx):
            raise ProviderError("boom")
            yield  # pragma: no cover

        def chat(self, messages, num_ctx):
            raise NotImplementedError

        def count_tokens(self, messages, tools=None):
            return 0

    result = _stream_provider_turn(
        FailingProvider(),
        messages=[{"role": "user", "content": "x"}],
        tools=[],
        num_ctx=4096,
        console=None,
        start_time=_time.time(),
    )
    assert isinstance(result.provider_error, ProviderError)
    assert str(result.provider_error) == "boom"
    assert result.content == ""
    assert result.tool_calls == []


def test_stream_keyboard_interrupt_propagates():
    """Ctrl-C during stream is NOT caught by the helper — it re-raises so
    the REPL's outer handler sees it and can clean up the prompt."""
    class InterruptingProvider:
        name = "fake-interrupt"
        model = "fake-v1"

        def stream_chat(self, messages, tools, num_ctx):
            yield StreamChunk(content_delta="partial")
            raise KeyboardInterrupt

        def chat(self, messages, num_ctx):
            raise NotImplementedError

        def count_tokens(self, messages, tools=None):
            return 0

    with pytest.raises(KeyboardInterrupt):
        _stream_provider_turn(
            InterruptingProvider(),
            messages=[{"role": "user", "content": "x"}],
            tools=[],
            num_ctx=4096,
            console=None,
            start_time=_time.time(),
        )


def test_stream_ttft_measured_on_first_content_not_reasoning():
    """ttft_ms should clock the first *visible* token, not the first
    reasoning delta. Reasoning is upstream thinking, not user-facing
    latency — conflating them would make reasoning-capable models look
    artificially fast on metrics they have no business winning."""
    start = _time.time() - 0.050  # pretend we started 50ms ago
    provider = FakeProvider([
        ScriptedTurn(
            reasoning_chunks=["thinking first"],
            content_chunks=["then reply"],
        ),
    ])
    result = _stream_provider_turn(
        provider,
        messages=[{"role": "user", "content": "x"}],
        tools=[],
        num_ctx=4096,
        console=None,
        start_time=start,
    )
    # ttft is measured relative to start_time, must be positive and
    # roughly ≥50ms since start was shifted backward.
    assert result.ttft_ms is not None
    assert result.ttft_ms >= 50


# ---------- step retries: transient provider errors ----------

def _build_retrying_provider(failures: list[ProviderError], success_content: str):
    """Build a provider that raises the given ProviderErrors on the first N
    stream_chat calls, then yields a normal turn returning `success_content`."""
    class RetryingProvider:
        name = "fake-retry"
        model = "fake-retry-v1"
        context_size = 32768

        def __init__(self):
            self._pending_failures = list(failures)
            self.call_count = 0

        def stream_chat(self, messages, tools, num_ctx):
            self.call_count += 1
            if self._pending_failures:
                err = self._pending_failures.pop(0)
                raise err
                yield  # pragma: no cover - keep it a generator
            yield StreamChunk(content_delta=success_content)
            yield StreamChunk(
                done=True,
                usage=Usage(prompt_tokens=10, completion_tokens=5),
            )

        def chat(self, messages, num_ctx):
            raise NotImplementedError

        def count_tokens(self, messages, tools=None):
            return 0

    return RetryingProvider()


def _install_provider_instance(p):
    """Swap the active provider to an arbitrary instance for one test."""
    from providers import get_active_provider
    original = get_active_provider()
    set_active_provider(p)
    return original


def test_retryable_provider_error_triggers_retry_and_succeeds(monkeypatch):
    """A retryable ProviderError (transient network/5xx) on the first turn
    should be retried; the second attempt that succeeds must produce a
    normal final response and surface the retry count in stats."""
    monkeypatch.setattr("agent.loop._sleep_for_retry", lambda _attempt: None)

    err = ProviderError("unreachable", retryable=True)
    provider = _build_retrying_provider([err], success_content="after retry")
    original = _install_provider_instance(provider)
    try:
        response, history, _ctx, stats = run_agent(
            user_input="hi",
            tools=[],
            execute_tool=_noop_execute_tool,
            history=[],
        )
    finally:
        set_active_provider(original)

    assert response == "after retry"
    assert provider.call_count == 2
    assert stats.get("provider_retries") == 1
    assert "provider_error" not in stats


def test_non_retryable_provider_error_is_not_retried(monkeypatch):
    """A ProviderError without retryable=True (auth, malformed payload,
    parse error) should surface immediately — no backoff loop."""
    sleeps: list[int] = []
    monkeypatch.setattr("agent.loop._sleep_for_retry", lambda attempt: sleeps.append(attempt))

    err = ProviderError("HTTP 401: bad key")  # default retryable=False
    provider = _build_retrying_provider([err, err], success_content="never reached")
    original = _install_provider_instance(provider)
    try:
        response, _history, _ctx, stats = run_agent(
            user_input="hi",
            tools=[],
            execute_tool=_noop_execute_tool,
            history=[],
        )
    finally:
        set_active_provider(original)

    assert response == ""
    assert provider.call_count == 1
    assert sleeps == []
    assert stats.get("provider_retries") in (None, 0)
    assert "provider_error" in stats


def test_retry_budget_exhausted_surfaces_final_error(monkeypatch):
    """When every retry attempt also raises, we give up after the cap and
    return the last provider_error in stats — not an empty success."""
    monkeypatch.setattr("agent.loop._sleep_for_retry", lambda _attempt: None)
    monkeypatch.setattr("agent.loop.MAX_PROVIDER_RETRIES", 2)

    err = ProviderError("unreachable", retryable=True)
    # 3 attempts (initial + 2 retries) all fail
    provider = _build_retrying_provider([err, err, err], success_content="unreached")
    original = _install_provider_instance(provider)
    try:
        response, _history, _ctx, stats = run_agent(
            user_input="hi",
            tools=[],
            execute_tool=_noop_execute_tool,
            history=[],
        )
    finally:
        set_active_provider(original)

    assert response == ""
    assert provider.call_count == 3  # initial + 2 retries
    assert stats.get("provider_retries") == 2
    assert "provider_error" in stats


def test_retry_skipped_when_partial_content_already_streamed(monkeypatch):
    """If the model already streamed visible content before the stream
    broke, we cannot safely retry — duplicate output would land in the
    user's terminal. Surface the error instead."""
    monkeypatch.setattr("agent.loop._sleep_for_retry", lambda _attempt: None)

    class PartialThenError:
        name = "fake-partial"
        model = "fake-partial-v1"
        context_size = 32768

        def __init__(self):
            self.call_count = 0

        def stream_chat(self, messages, tools, num_ctx):
            self.call_count += 1
            yield StreamChunk(content_delta="halfway there ")
            raise ProviderError("dropped mid-stream", retryable=True)

        def chat(self, messages, num_ctx):
            raise NotImplementedError

        def count_tokens(self, messages, tools=None):
            return 0

    provider = PartialThenError()
    original = _install_provider_instance(provider)
    try:
        response, _history, _ctx, stats = run_agent(
            user_input="hi",
            tools=[],
            execute_tool=_noop_execute_tool,
            history=[],
        )
    finally:
        set_active_provider(original)

    assert response == ""
    assert provider.call_count == 1
    assert stats.get("provider_retries") in (None, 0)
    assert "provider_error" in stats


# ---------- idempotency cache for read-only tools ----------

def _counting_execute_tool():
    """Build an execute_tool that records every call and returns a result
    derived from the args so tests can compare cached vs fresh output."""
    calls: list[tuple[str, dict]] = []

    def execute(name: str, args: dict) -> str:
        calls.append((name, dict(args)))
        return f"{name}:{args}"

    return execute, calls


def test_read_only_tool_repeat_call_hits_cache(install_provider):
    """When a read-only tool is called twice with identical args within
    the same run_agent call, the second call should hit the per-call
    cache and skip the actual execution."""
    install_provider([
        ScriptedTurn(content_chunks=[], tool_calls=[
            ToolCall("read_file", {"path": "a.py"})
        ]),
        ScriptedTurn(content_chunks=[], tool_calls=[
            ToolCall("read_file", {"path": "a.py"})
        ]),
        ScriptedTurn(content_chunks=["done"]),
    ])

    execute, calls = _counting_execute_tool()
    response, _history, _ctx, _stats = run_agent(
        user_input="read it twice",
        tools=[{"type": "function", "function": {"name": "read_file", "parameters": {}}}],
        execute_tool=execute,
        history=[],
    )

    assert response == "done"
    assert len(calls) == 1


def test_read_only_tool_different_args_does_not_hit_cache(install_provider):
    """The cache key includes the args — a second call with different
    args must execute fresh, not return the prior cached result."""
    install_provider([
        ScriptedTurn(content_chunks=[], tool_calls=[
            ToolCall("read_file", {"path": "a.py"})
        ]),
        ScriptedTurn(content_chunks=[], tool_calls=[
            ToolCall("read_file", {"path": "b.py"})
        ]),
        ScriptedTurn(content_chunks=["done"]),
    ])

    execute, calls = _counting_execute_tool()
    run_agent(
        user_input="read two files",
        tools=[{"type": "function", "function": {"name": "read_file", "parameters": {}}}],
        execute_tool=execute,
        history=[],
    )

    assert len(calls) == 2


def test_mutating_tool_is_never_cached(install_provider):
    """Mutating tools (write_file, bash, edit_file) must always re-execute —
    the world changes between calls and a stale cached result would lie
    to the model."""
    install_provider([
        ScriptedTurn(content_chunks=[], tool_calls=[
            ToolCall("write_file", {"path": "x.py", "content": "v1"})
        ]),
        ScriptedTurn(content_chunks=[], tool_calls=[
            ToolCall("write_file", {"path": "x.py", "content": "v1"})
        ]),
        ScriptedTurn(content_chunks=["done"]),
    ])

    execute, calls = _counting_execute_tool()
    # Permission gate must approve write_file for this test (subagent
    # default would block it).
    run_agent(
        user_input="write twice",
        tools=[{"type": "function", "function": {"name": "write_file", "parameters": {}}}],
        execute_tool=execute,
        history=[],
        permission_check=lambda name, args, meta=None: True,
    )

    assert len(calls) == 2


def test_mutating_call_invalidates_read_cache(install_provider):
    """A successful mutating call clears the cache: a read after a write
    must reflect the new state of the world, not the pre-write read."""
    install_provider([
        ScriptedTurn(content_chunks=[], tool_calls=[
            ToolCall("read_file", {"path": "a.py"})
        ]),
        ScriptedTurn(content_chunks=[], tool_calls=[
            ToolCall("write_file", {"path": "a.py", "content": "v2"})
        ]),
        ScriptedTurn(content_chunks=[], tool_calls=[
            ToolCall("read_file", {"path": "a.py"})
        ]),
        ScriptedTurn(content_chunks=["done"]),
    ])

    execute, calls = _counting_execute_tool()
    run_agent(
        user_input="read-write-read",
        tools=[
            {"type": "function", "function": {"name": "read_file", "parameters": {}}},
            {"type": "function", "function": {"name": "write_file", "parameters": {}}},
        ],
        execute_tool=execute,
        history=[],
        permission_check=lambda name, args, meta=None: True,
    )

    # 3 actual executions: first read, the write, then the post-write read.
    assert len(calls) == 3
    assert [c[0] for c in calls] == ["read_file", "write_file", "read_file"]


def test_failed_read_only_call_is_not_cached(install_provider):
    """If a read-only tool raises, the failure must NOT be cached. A retry
    with the same args should execute fresh (the failure may have been
    transient, and a cached exception string would prevent recovery)."""
    install_provider([
        ScriptedTurn(content_chunks=[], tool_calls=[
            ToolCall("read_file", {"path": "a.py"})
        ]),
        ScriptedTurn(content_chunks=[], tool_calls=[
            ToolCall("read_file", {"path": "a.py"})
        ]),
        ScriptedTurn(content_chunks=["done"]),
    ])

    call_count = [0]

    def execute(name: str, args: dict) -> str:
        call_count[0] += 1
        if call_count[0] == 1:
            raise OSError("transient")
        return "ok"

    run_agent(
        user_input="retry on fail",
        tools=[{"type": "function", "function": {"name": "read_file", "parameters": {}}}],
        execute_tool=execute,
        history=[],
    )

    assert call_count[0] == 2


def test_cache_hit_surfaces_in_tool_end_meta(install_provider):
    """The on_tool_end callback receives `meta['cache_hit']=True` for cached
    calls so the UI can render a `(cached)` marker without re-implementing
    the cache lookup."""
    install_provider([
        ScriptedTurn(content_chunks=[], tool_calls=[
            ToolCall("read_file", {"path": "a.py"})
        ]),
        ScriptedTurn(content_chunks=[], tool_calls=[
            ToolCall("read_file", {"path": "a.py"})
        ]),
        ScriptedTurn(content_chunks=["done"]),
    ])

    captured: list[dict] = []

    def end_hook(name, args, result, ok, meta=None):
        captured.append({"name": name, "ok": ok, "meta": meta})

    execute, _calls = _counting_execute_tool()
    run_agent(
        user_input="double read",
        tools=[{"type": "function", "function": {"name": "read_file", "parameters": {}}}],
        execute_tool=execute,
        history=[],
        on_tool_end=end_hook,
    )

    assert captured[0]["meta"].get("cache_hit") in (None, False)
    assert captured[1]["meta"].get("cache_hit") is True


# ---------- pre/post-tool hooks ----------

def test_pre_tool_hook_can_rewrite_args(install_provider):
    """A pre_tool_hook returning (True, new_args, None) replaces the args
    the tool receives — useful for path normalization, env injection,
    redaction, etc."""
    install_provider([
        ScriptedTurn(content_chunks=[], tool_calls=[
            ToolCall("read_file", {"path": "RAW"})
        ]),
        ScriptedTurn(content_chunks=["done"]),
    ])

    captured_args: list[dict] = []

    def execute(name: str, args: dict) -> str:
        captured_args.append(dict(args))
        return "ok"

    def pre(name: str, args: dict):
        if name == "read_file":
            return True, {"path": args["path"].lower()}, None
        return True, args, None

    run_agent(
        user_input="rewrite path",
        tools=[{"type": "function", "function": {"name": "read_file", "parameters": {}}}],
        execute_tool=execute,
        history=[],
        pre_tool_hook=pre,
    )

    assert captured_args == [{"path": "raw"}]


def test_pre_tool_hook_can_block_call(install_provider):
    """A pre_tool_hook returning (False, args, 'reason') blocks the tool
    call — the model sees the reason as the tool result, execution is
    skipped, downstream cache/post-hook do not fire."""
    install_provider([
        ScriptedTurn(content_chunks=[], tool_calls=[
            ToolCall("read_file", {"path": "/etc/passwd"})
        ]),
        ScriptedTurn(content_chunks=["pivoted"]),
    ])

    call_count = [0]

    def execute(name: str, args: dict) -> str:
        call_count[0] += 1
        return "should not run"

    def pre(name: str, args: dict):
        if args.get("path", "").startswith("/etc"):
            return False, args, "policy: /etc paths are off-limits"
        return True, args, None

    response, _h, _c, _s = run_agent(
        user_input="try /etc",
        tools=[{"type": "function", "function": {"name": "read_file", "parameters": {}}}],
        execute_tool=execute,
        history=[],
        pre_tool_hook=pre,
    )

    assert call_count[0] == 0
    assert response == "pivoted"


def test_post_tool_hook_can_transform_result(install_provider):
    """A post_tool_hook can rewrite the result string before it reaches
    the model — useful for redaction, length capping, structured-output
    normalization."""
    install_provider([
        ScriptedTurn(content_chunks=[], tool_calls=[
            ToolCall("read_file", {"path": "secret.txt"})
        ]),
        ScriptedTurn(content_chunks=["seen"]),
    ])

    seen_by_end_hook: list[str] = []

    def end_hook(name, args, result, ok, meta=None):
        seen_by_end_hook.append(result)

    def post(name: str, args: dict, result: str, ok: bool):
        return "[REDACTED]", ok

    def execute(name, args):
        return "API_KEY=sk-deadbeef"

    run_agent(
        user_input="read secret",
        tools=[{"type": "function", "function": {"name": "read_file", "parameters": {}}}],
        execute_tool=execute,
        history=[],
        post_tool_hook=post,
        on_tool_end=end_hook,
    )

    assert seen_by_end_hook == ["[REDACTED]"]


def test_post_tool_hook_can_override_ok_status(install_provider):
    """A post_tool_hook can flip the ok flag — useful for marking a
    structurally successful call as a failure (lint result reports
    errors, HTTP 200 with error body, etc.)."""
    install_provider([
        ScriptedTurn(content_chunks=[], tool_calls=[
            ToolCall("read_file", {"path": "x"})
        ]),
        ScriptedTurn(content_chunks=["acknowledged"]),
    ])

    seen_oks: list[bool] = []

    def end_hook(name, args, result, ok, meta=None):
        seen_oks.append(ok)

    def post(name, args, result, ok):
        return result, False

    run_agent(
        user_input="x",
        tools=[{"type": "function", "function": {"name": "read_file", "parameters": {}}}],
        execute_tool=lambda n, a: "fine",
        history=[],
        post_tool_hook=post,
        on_tool_end=end_hook,
    )

    assert seen_oks == [False]


def test_no_hooks_is_pass_through(install_provider):
    """The default (no hooks) must keep existing behavior intact — args
    untouched, result untouched, ok untouched."""
    install_provider([
        ScriptedTurn(content_chunks=[], tool_calls=[
            ToolCall("read_file", {"path": "y"})
        ]),
        ScriptedTurn(content_chunks=["done"]),
    ])

    end_records: list[tuple[dict, str, bool]] = []

    def end_hook(name, args, result, ok, meta=None):
        end_records.append((dict(args), result, ok))

    run_agent(
        user_input="just run",
        tools=[{"type": "function", "function": {"name": "read_file", "parameters": {}}}],
        execute_tool=lambda n, a: f"got {a['path']}",
        history=[],
        on_tool_end=end_hook,
    )

    assert end_records == [({"path": "y"}, "got y", True)]
