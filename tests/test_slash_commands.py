"""handle_slash dispatch: uniform (state, arg='') signature; unknown
commands reported without crash; commands with no-arg ignore stray args."""
from repl.commands import handle_slash


def test_unknown_command_returns_true_without_crash(state):
    assert handle_slash("/nope", state) is True


def test_non_slash_input_returns_false(state):
    assert handle_slash("plain text", state) is False


def test_known_command_returns_true(state):
    # /help is cheap — prints to console but doesn't mutate state
    assert handle_slash("/help", state) is True


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
