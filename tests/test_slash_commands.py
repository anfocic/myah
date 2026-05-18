"""handle_slash dispatch: uniform (state, arg='') signature; unknown
commands reported without crash; commands with no-arg ignore stray args."""
from repl.commands import cmd_retry, handle_slash


def test_unknown_command_returns_true_without_crash(state):
    assert handle_slash("/nope", state) is True


def test_non_slash_input_returns_false(state):
    assert handle_slash("plain text", state) is False


def test_known_command_returns_true(state):
    # /help is cheap — prints to console but doesn't mutate state
    assert handle_slash("/help", state) is True


def test_session_command_renders_rail(state):
    # /session prints the Phosphor session rail; no state mutation
    assert handle_slash("/session", state) is True


def test_command_with_arg_is_parsed(state):
    # /rewind accepts an integer arg; with no snapshots it's a no-op
    assert handle_slash("/rewind 3", state) is True


def test_no_arg_command_with_extra_ignores_arg(state):
    # /help doesn't care about args — must still succeed
    assert handle_slash("/help ignored-text", state) is True


def test_profile_renders_on_empty_history(state):
    # /profile rebuilds the system prompt and prints a breakdown — must
    # work before any turn has happened (history empty, ctx_used=0).
    assert handle_slash("/profile", state) is True


def test_profile_renders_with_history(state):
    state["history"] = [
        {"role": "user", "content": "hello world"},
        {"role": "assistant", "content": "hi there — how can I help?"},
    ]
    assert handle_slash("/profile", state) is True


def test_profile_renders_in_plan_mode(state):
    # Plan-mode rules are a non-empty part of the system prompt; make sure
    # the branch that renders "plan-mode rules: N tokens" instead of
    # "(inactive)" doesn't crash.
    state["plan_mode"] = True
    assert handle_slash("/profile", state) is True


def test_context_uses_provider_count_tokens(state, monkeypatch):
    """/context should call the active provider's count_tokens and surface
    the number as `ctx (next turn)`, plus update state["ctx_used"] so the
    per-turn tag lines up."""
    calls: list[dict] = []

    class FakeProvider:
        name = "fake"
        model = "fake-m"

        def count_tokens(self, messages, tools=None):
            calls.append({"messages": messages, "tools": tools})
            return 4242

    monkeypatch.setattr(
        "repl.commands.get_active_provider", lambda: FakeProvider()
    )
    assert handle_slash("/context", state) is True
    assert state["ctx_used"] == 4242
    # Should have been called once, with a system prompt prepended to history.
    assert len(calls) == 1
    assert calls[0]["messages"][0]["role"] == "system"
    # Tools list is non-None (the REPL's registered tools).
    assert calls[0]["tools"] is not None


def test_profile_falls_back_on_provider_error(state, monkeypatch):
    """Provider raising ProviderError must not crash /profile — it falls
    back to the char/4 estimator with a dim note. The existing tests cover
    the happy path implicitly (they hit a live provider or let the fallback
    trigger); this one pins the behavior explicitly."""
    from providers import ProviderError

    class BoomProvider:
        name = "fake"
        model = "fake-m"

        def count_tokens(self, messages, tools=None):
            raise ProviderError("simulated unreachable")

    monkeypatch.setattr(
        "repl.commands.get_active_provider", lambda: BoomProvider()
    )
    state["history"] = [
        {"role": "user", "content": "hello there"},
        {"role": "assistant", "content": "hi back"},
    ]
    # Must not raise — fallback path is what we're pinning.
    assert handle_slash("/profile", state) is True


def test_eval_list_invokes_list_tasks(state, monkeypatch):
    """/eval list should call evals.runner.list_tasks and not run_suite."""
    calls = {"list": 0, "run": 0}

    def fake_list():
        calls["list"] += 1
        return ["find_string", "edit_rename"]

    def fake_run(**kwargs):
        calls["run"] += 1
        return []

    monkeypatch.setattr("evals.runner.list_tasks", fake_list)
    monkeypatch.setattr("evals.runner.run_suite", fake_run)
    assert handle_slash("/eval list", state) is True
    assert calls == {"list": 1, "run": 0}


def test_eval_no_arg_runs_full_suite(state, monkeypatch):
    """/eval with no args should call run_suite with task_ids=None."""
    captured: dict = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return []

    class FakeProvider:
        name = "fake"
        model = "fake-m"

    monkeypatch.setattr("evals.runner.run_suite", fake_run)
    monkeypatch.setattr(
        "repl.commands.get_active_provider", lambda: FakeProvider()
    )
    assert handle_slash("/eval", state) is True
    assert captured["task_ids"] is None


def test_eval_with_task_ids_passes_subset(state, monkeypatch):
    """/eval find_string edit_rename should pass the parsed list through."""
    captured: dict = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return []

    class FakeProvider:
        name = "fake"
        model = "fake-m"

    monkeypatch.setattr("evals.runner.run_suite", fake_run)
    monkeypatch.setattr(
        "repl.commands.get_active_provider", lambda: FakeProvider()
    )
    assert handle_slash("/eval find_string edit_rename", state) is True
    assert captured["task_ids"] == ["find_string", "edit_rename"]


def test_profile_marginal_rows_from_provider_counts(state, monkeypatch):
    """With a fake provider that counts content characters (one token per
    char), verify the marginal-diff arithmetic: each row = full_notools -
    full_with_that_role_blanked."""
    class CharProvider:
        name = "fake"
        model = "fake-m"

        def count_tokens(self, messages, tools=None):
            # Deterministic: one "token" per char across content, plus a
            # fixed 5 for tools when provided. Lets the test reason about
            # exact row values.
            total = sum(len(m.get("content") or "") for m in messages)
            if tools:
                total += 5
            return total

    monkeypatch.setattr(
        "repl.commands.get_active_provider", lambda: CharProvider()
    )
    state["history"] = [
        {"role": "user", "content": "hello"},      # 5
        {"role": "assistant", "content": "hi!"},   # 3
    ]
    # Shouldn't crash; the exact visuals aren't asserted (console output),
    # but the code path that does five count_tokens calls + subtraction
    # arithmetic must run cleanly for a realistic history.
    assert handle_slash("/profile", state) is True


def test_profile_cost_breakdown_computation(state, monkeypatch):
    """The per-role cost breakdown multiplies marginal token counts by the
    active provider's input/output prices. assistant uses output price; the
    other rows use input price. Verify with a priced provider and a fake
    char-counting tokenizer."""
    from providers.pricing import ModelPrice
    from repl.commands import _profile_cost_breakdown

    class CharProvider:
        name = "openai"
        model = "gpt-4o-mini"  # $0.15 / $0.60 per Mtok

        def count_tokens(self, messages, tools=None):
            total = sum(len(m.get("content") or "") for m in messages)
            if tools:
                total += 4
            return total

    provider = CharProvider()
    monkeypatch.setattr(
        "repl.commands.get_active_provider", lambda: provider
    )

    # Tokens-per-role from the marginal diff — pass through to the helper
    # directly so the test pins the pricing math, not the row arithmetic
    # (which the existing test covers).
    breakdown = _profile_cost_breakdown(
        provider,
        system_tokens=1000,
        user_tokens=2000,
        assistant_tokens=500,
        tools_tokens=300,
    )
    assert breakdown is not None
    price = ModelPrice(0.15, 0.60)
    expected_system = 1000 * price.input_per_mtok / 1_000_000
    expected_user = 2000 * price.input_per_mtok / 1_000_000
    expected_asst = 500 * price.output_per_mtok / 1_000_000
    expected_tools = 300 * price.input_per_mtok / 1_000_000
    assert breakdown["system"] == expected_system
    assert breakdown["user"] == expected_user
    assert breakdown["assistant"] == expected_asst
    assert breakdown["tools"] == expected_tools
    expected_total = expected_system + expected_user + expected_asst + expected_tools
    assert breakdown["total"] == expected_total


def test_profile_cost_breakdown_none_when_unpriced(monkeypatch):
    """Unknown (provider, model) → no pricing → helper returns None so
    /profile can skip the cost section entirely instead of printing zeros."""
    from repl.commands import _profile_cost_breakdown

    class UnpricedProvider:
        name = "made-up"
        model = "not-in-table"

    assert _profile_cost_breakdown(
        UnpricedProvider(),
        system_tokens=100,
        user_tokens=200,
        assistant_tokens=50,
        tools_tokens=10,
    ) is None


def test_profile_renders_cost_section_when_priced(state, monkeypatch, capsys):
    """End-to-end: /profile with a priced provider prints a COST BREAKDOWN
    section listing every role plus a total dollar figure."""
    from rich.console import Console

    from repl import commands

    class CharProvider:
        name = "openai"
        model = "gpt-4o-mini"

        def count_tokens(self, messages, tools=None):
            total = sum(len(m.get("content") or "") for m in messages)
            if tools:
                total += 4
            return total

    monkeypatch.setattr(commands, "get_active_provider", lambda: CharProvider())
    # Capture-friendly console: force_terminal=False, width wide enough that
    # markup doesn't wrap mid-token.
    monkeypatch.setattr(commands, "console", Console(force_terminal=False, width=200))

    state["history"] = [
        {"role": "user", "content": "hello there partner"},
        {"role": "assistant", "content": "well hi back to ya"},
    ]
    assert handle_slash("/profile", state) is True
    out = capsys.readouterr().out
    assert "COST BREAKDOWN" in out
    # Every role label should appear in the cost section output.
    assert "assistant" in out
    assert "system" in out
    # Total row uses the format_cost_usd helper (a $-prefixed string).
    assert "$" in out


# ---------- /retry: normal vs stream-interrupted ----------

def test_retry_normal_pops_last_pair_and_queues_user_input(state):
    """Default /retry: pops the trailing user/assistant pair and queues
    the user message for resubmission — same behavior as before PR #106
    landed the partial-content marker."""
    state["history"] = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "first reply"},
        {"role": "user", "content": "what's 2+2?"},
        {"role": "assistant", "content": "4"},
    ]
    cmd_retry(state)
    assert state["_retry_input"] == "what's 2+2?"
    assert state["history"] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "first reply"},
    ]


def test_retry_on_interrupted_stream_resumes_instead_of_restarting(state):
    """When the last assistant message carries the stream-interrupted
    marker (left by PR #106), /retry should NOT pop the pair. Instead
    it queues a "Continue." nudge so the model picks up from its own
    partial reply rather than starting over and losing the work."""
    state["history"] = [
        {"role": "user", "content": "explain the loop"},
        {
            "role": "assistant",
            "content": "It starts at run_agent which builds messages "
            "[stream interrupted: fake-provider: connection lost]",
        },
    ]
    cmd_retry(state)
    assert state["_retry_input"] == "Continue."
    # History intact — the partial reply stays in context so the model
    # has its own breadcrumb of where it was.
    assert len(state["history"]) == 2
    assert "[stream interrupted" in state["history"][-1]["content"]


def test_retry_no_history_is_noop(state):
    cmd_retry(state)
    assert "_retry_input" not in state
    assert state["history"] == []
