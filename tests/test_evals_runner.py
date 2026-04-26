"""Tests for the eval runner and check dispatcher.

Grader unit tests use synthetic bundles (no provider, no tools). The
end-to-end test reuses the scripted FakeProvider from
`tests/test_integration.py` to drive one fake task through `run_suite`.
"""
from __future__ import annotations

import sys
from pathlib import Path

from evals import checks as checks_mod
from evals import runner
from providers import ProviderError
from providers.base import ToolCall
from tests.test_integration import FakeProvider, ScriptedTurn, install_provider  # noqa: F401

# ---------- check dispatcher unit tests ----------

def _bundle(**kw) -> dict:
    base = {
        "content": "",
        "trace": [],
        "stats": {},
        "ctx_used": 0,
        "cwd": Path("."),
        "fixture_dir": None,
    }
    base.update(kw)
    return base


def test_tool_trace_pass_and_fail():
    b = _bundle(trace=[{"name": "grep"}, {"name": "read_file"}])
    ok, _ = checks_mod.dispatch(
        {"type": "tool_trace", "must_call": ["grep"], "must_not_call": ["bash"]}, b
    )
    assert ok

    ok, why = checks_mod.dispatch(
        {"type": "tool_trace", "must_call": ["edit_file"]}, b
    )
    assert not ok and "expected calls not made" in why

    ok, why = checks_mod.dispatch(
        {"type": "tool_trace", "must_not_call": ["read_file"]}, b
    )
    assert not ok and "forbidden calls made" in why


def test_tool_trace_call_count_max():
    b = _bundle(trace=[{"name": "grep"}] * 5)
    ok, _ = checks_mod.dispatch({"type": "tool_trace", "call_count_max": 5}, b)
    assert ok
    ok, why = checks_mod.dispatch({"type": "tool_trace", "call_count_max": 4}, b)
    assert not ok and "too many tool calls" in why


def test_content_regex_and_negate():
    b = _bundle(content="the answer is config.py and tokens.py")
    assert checks_mod.dispatch({"type": "content_regex", "pattern": r"config\.py"}, b)[0]
    assert not checks_mod.dispatch({"type": "content_regex", "pattern": r"missing"}, b)[0]
    # negate: pass when pattern is NOT present
    assert checks_mod.dispatch({"type": "content_regex", "pattern": r"missing", "negate": True}, b)[0]
    assert not checks_mod.dispatch({"type": "content_regex", "pattern": r"config", "negate": True}, b)[0]


def test_content_regex_anchors_are_multiline():
    """`^` / `$` anchor to line boundaries, not string boundaries. Models
    routinely prefix a reply with a lead-in line ("Here is the commit
    message:\\n\\nfeat(x): ..."); an anchored regex like `^(feat|fix)...`
    should still match the line below. Matches the commit_msg_from_diff
    task's real-world pattern."""
    b = _bundle(
        content="Here is the commit message:\n\nfeat(repl): swap chrome"
    )
    pattern = r"^(feat|fix|refactor|docs|test|chore|perf|build|ci|style)(\(.+?\))?!?:"
    assert checks_mod.dispatch(
        {"type": "content_regex", "pattern": pattern}, b
    )[0]


def test_content_substr():
    b = _bundle(content="Hello World")
    assert checks_mod.dispatch({"type": "content_substr", "value": "World"}, b)[0]
    assert checks_mod.dispatch({"type": "content_substr", "value": "world", "ignorecase": True}, b)[0]
    assert not checks_mod.dispatch({"type": "content_substr", "value": "world"}, b)[0]


def test_fs_file_contains(tmp_path: Path):
    f = tmp_path / "sample.py"
    f.write_text("def new_name():\n    pass\n")
    b = _bundle(cwd=tmp_path)
    assert checks_mod.dispatch(
        {"type": "fs_file_contains", "path": "sample.py", "pattern": r"^def new_name\("}, b
    )[0]
    assert checks_mod.dispatch(
        {"type": "fs_file_contains", "path": "sample.py", "pattern": r"^def old_name\(", "negate": True}, b
    )[0]
    assert not checks_mod.dispatch(
        {"type": "fs_file_contains", "path": "missing.py", "pattern": r"."}, b
    )[0]


def test_fs_file_equals_literal(tmp_path: Path):
    f = tmp_path / "x.txt"
    f.write_text("abc")
    b = _bundle(cwd=tmp_path)
    assert checks_mod.dispatch(
        {"type": "fs_file_equals", "path": "x.txt", "expected": "abc"}, b
    )[0]
    assert not checks_mod.dispatch(
        {"type": "fs_file_equals", "path": "x.txt", "expected": "abd"}, b
    )[0]


def test_fs_grep_count_eq_ge_le(tmp_path: Path):
    f = tmp_path / "src.py"
    f.write_text("foo\nfoo bar\nbaz\nfoo\n")
    b = _bundle(cwd=tmp_path)
    # eq is the default op
    assert checks_mod.dispatch(
        {"type": "fs_grep_count", "path": "src.py", "pattern": r"foo", "expected": 3}, b
    )[0]
    ok, why = checks_mod.dispatch(
        {"type": "fs_grep_count", "path": "src.py", "pattern": r"foo", "expected": 2}, b
    )
    assert not ok and "found 3" in why
    # ge passes when count >= expected
    assert checks_mod.dispatch(
        {"type": "fs_grep_count", "path": "src.py", "pattern": r"foo",
         "expected": 2, "op": "ge"}, b
    )[0]
    # le passes when count <= expected
    assert checks_mod.dispatch(
        {"type": "fs_grep_count", "path": "src.py", "pattern": r"foo",
         "expected": 5, "op": "le"}, b
    )[0]
    # missing file is a fail, not a raise
    assert not checks_mod.dispatch(
        {"type": "fs_grep_count", "path": "absent.py", "pattern": r"x", "expected": 0}, b
    )[0]


def test_bash_exit_zero_pass_and_fail(tmp_path: Path):
    b = _bundle(cwd=tmp_path)
    # Exit 0
    assert checks_mod.dispatch(
        {"type": "bash_exit_zero", "cmd": "true"}, b
    )[0]
    # Exit nonzero — why should surface exit code and command
    ok, why = checks_mod.dispatch(
        {"type": "bash_exit_zero", "cmd": "false"}, b
    )
    assert not ok and "exit 1" in why


def test_bash_exit_zero_uses_mia_python_for_plain_python_cmd(tmp_path: Path):
    b = _bundle(cwd=tmp_path)
    cmd = "python -c \"import sys; from pathlib import Path; Path('py.txt').write_text(sys.executable)\""
    assert checks_mod.dispatch({"type": "bash_exit_zero", "cmd": cmd}, b)[0]
    assert (tmp_path / "py.txt").read_text() == sys.executable


def test_normalize_python_cmd_quotes_current_interpreter():
    quoted = checks_mod.shlex.quote(sys.executable)
    assert checks_mod._normalize_python_cmd("python -m pytest tests/") == (
        f"{quoted} -m pytest tests/"
    )
    # `python3` gets the same rewrite — the asymmetry between `python` and
    # `python3` only bites you when the latter resolves to a different
    # interpreter than Myah's venv, which is exactly when it matters most.
    assert checks_mod._normalize_python_cmd("python3 -m pytest") == (
        f"{quoted} -m pytest"
    )
    # Unrelated shell commands pass through unchanged.
    assert checks_mod._normalize_python_cmd("ruff check .") == "ruff check ."


def test_bash_exit_zero_respects_cwd(tmp_path: Path):
    # Writes a sentinel into the task's cwd, then a second check reads it.
    # Proves the `cmd` runs with cwd=bundle["cwd"].
    b = _bundle(cwd=tmp_path)
    assert checks_mod.dispatch(
        {"type": "bash_exit_zero", "cmd": "echo hi > marker.txt"}, b
    )[0]
    assert (tmp_path / "marker.txt").exists()


def test_bash_exit_zero_cwd_rel(tmp_path: Path):
    sub = tmp_path / "sub"
    sub.mkdir()
    b = _bundle(cwd=tmp_path)
    assert checks_mod.dispatch(
        {"type": "bash_exit_zero", "cmd": "pwd | grep -q sub", "cwd_rel": "sub"}, b
    )[0]


def test_bash_exit_zero_timeout(tmp_path: Path):
    b = _bundle(cwd=tmp_path)
    ok, why = checks_mod.dispatch(
        {"type": "bash_exit_zero", "cmd": "sleep 2", "timeout_s": 1}, b
    )
    assert not ok and "timed out" in why


def test_python_callable_and_dict_form():
    b = _bundle(content="42")
    ok, _ = checks_mod.dispatch(lambda bundle: bundle["content"] == "42", b)
    assert ok
    ok, _ = checks_mod.dispatch({"type": "python", "fn": lambda bundle: (False, "nope")}, b)
    assert not ok


def test_unknown_check_type_returns_failure():
    ok, why = checks_mod.dispatch({"type": "does_not_exist"}, _bundle())
    assert not ok and "unknown check type" in why


# ---------- runner end-to-end against FakeProvider ----------

def test_run_suite_e2e(monkeypatch, tmp_path, install_provider):  # noqa: F811
    """Drive one fake task through run_suite. Scripts a fake model that
    emits a grep tool call then a final answer mentioning the target file,
    verifies the task passes, and checks JSONL output."""
    install_provider([
        ScriptedTurn(
            content_chunks=[],
            tool_calls=[ToolCall(name="grep", arguments={"pattern": "TOOL_RESULT_MAX_BYTES"})],
        ),
        ScriptedTurn(content_chunks=["Found in config.py"]),
    ])

    fake_task = {
        "id": "fake_find",
        "prompt": "find it",
        "setup": {"fs": None},
        "provider": None,
        "plan_mode": False,
        "permission": "allow_all",
        "limits": {"max_tool_calls": 4, "wall_timeout_s": 30},
        "checks": [
            {"type": "tool_trace", "must_call": ["grep"]},
            {"type": "content_regex", "pattern": r"config\.py"},
        ],
    }

    monkeypatch.setattr(runner, "discover_tasks", lambda: [fake_task])
    monkeypatch.setattr(runner, "RESULTS_ROOT", tmp_path)

    results = runner.run_suite()
    assert len(results) == 1
    r = results[0]
    assert r.passed, f"expected pass, got failures: {r.check_results}, err={r.error}"
    assert len(r.trace) == 1
    assert r.trace[0]["name"] == "grep"

    # JSONL file was written
    written = list(tmp_path.glob("*.jsonl"))
    assert len(written) == 1
    assert written[0].read_text().count("\n") == 1


def test_run_suite_fails_when_forbidden_tool_is_called(monkeypatch, tmp_path, install_provider):  # noqa: F811
    install_provider([
        ScriptedTurn(
            content_chunks=[],
            tool_calls=[ToolCall(name="bash", arguments={"command": "grep foo"})],
        ),
        ScriptedTurn(content_chunks=["Done."]),
    ])

    fake_task = {
        "id": "fake_refuse",
        "prompt": "find it",
        "permission": "allow_all",
        "checks": [
            {"type": "tool_trace", "must_not_call": ["bash"]},
        ],
    }
    monkeypatch.setattr(runner, "discover_tasks", lambda: [fake_task])
    monkeypatch.setattr(runner, "RESULTS_ROOT", tmp_path)

    results = runner.run_suite()
    assert len(results) == 1
    assert not results[0].passed
    assert not results[0].check_results[0]["pass"]
    assert "bash" in results[0].check_results[0]["why"]


def test_run_suite_passes_on_timeout_if_checks_pass(monkeypatch, tmp_path):
    """Regression for the "agent fixes the bug then keeps talking until
    wall_timeout_s" false-negative. After the timeout-decouple, task is
    PASS iff checks pass — timeout is a notes-column signal only.

    Constructs a provider that blocks in stream_chat, forcing the thread
    to still be alive at join(). task.checks pass via a trivial callable,
    so task_passed must be True while task.timeout is True."""
    import threading as _threading

    from providers import get_active_provider, set_active_provider
    from providers.base import StreamChunk

    stop = _threading.Event()

    class BlockingProvider:
        name = "fake-blocking"
        model = "fake-blocking-v1"

        def stream_chat(self, messages, tools, num_ctx):
            # Block until the test releases; when wall_timeout_s expires,
            # the runner's thread.join returns and the outer code continues
            # without waiting for us to exit. We eventually unblock so this
            # daemon thread can die cleanly.
            stop.wait(timeout=5)
            yield StreamChunk(done=True)

        def chat(self, messages, num_ctx):
            raise NotImplementedError

        def count_tokens(self, messages, tools=None):
            return 0

    original = get_active_provider()
    set_active_provider(BlockingProvider())
    try:
        fake_task = {
            "id": "fake_timeout",
            "prompt": "do nothing",
            "setup": {"fs": None},
            "permission": "allow_all",
            "limits": {"max_tool_calls": 4, "wall_timeout_s": 1},
            "checks": [lambda bundle: True],  # trivial python check, always passes
        }
        monkeypatch.setattr(runner, "discover_tasks", lambda: [fake_task])
        monkeypatch.setattr(runner, "RESULTS_ROOT", tmp_path)

        results = runner.run_suite()
        r = results[0]
        assert r.timeout is True, "provider should have blocked past wall_timeout_s"
        assert r.passed is True, (
            "task must pass when checks pass, regardless of timeout "
            f"(check_results={r.check_results})"
        )
        assert r.error and "wall_timeout_s" in r.error, (
            "timeout should still surface in r.error for triage"
        )
    finally:
        stop.set()
        set_active_provider(original)


def test_run_suite_fails_on_timeout_if_checks_fail(monkeypatch, tmp_path):
    """Sibling to the PASS case: when checks fail AND the task times out,
    task is still FAIL — the checks, not the timeout, are the gate."""
    import threading as _threading

    from providers import get_active_provider, set_active_provider
    from providers.base import StreamChunk

    stop = _threading.Event()

    class BlockingProvider:
        name = "fake-blocking"
        model = "fake-blocking-v1"

        def stream_chat(self, messages, tools, num_ctx):
            stop.wait(timeout=5)
            yield StreamChunk(done=True)

        def chat(self, messages, num_ctx):
            raise NotImplementedError

        def count_tokens(self, messages, tools=None):
            return 0

    original = get_active_provider()
    set_active_provider(BlockingProvider())
    try:
        fake_task = {
            "id": "fake_timeout_fail",
            "prompt": "do nothing",
            "setup": {"fs": None},
            "permission": "allow_all",
            "limits": {"max_tool_calls": 4, "wall_timeout_s": 1},
            "checks": [lambda bundle: False],
        }
        monkeypatch.setattr(runner, "discover_tasks", lambda: [fake_task])
        monkeypatch.setattr(runner, "RESULTS_ROOT", tmp_path)

        results = runner.run_suite()
        r = results[0]
        assert r.timeout is True
        assert r.passed is False
    finally:
        stop.set()
        set_active_provider(original)


def test_run_one_supports_multi_turn(monkeypatch, tmp_path, install_provider):  # noqa: F811
    """Multi-turn tasks: two user inputs are delivered in order, history
    threads between them, and the bundle exposes each turn's content."""
    install_provider([
        ScriptedTurn(content_chunks=["first answer"]),
        ScriptedTurn(content_chunks=["second answer"]),
    ])

    captured: dict = {}

    def capturing_check(bundle):
        # One task = one checks pass, so this runs once with the final bundle.
        captured["content"] = bundle["content"]
        captured["turn_contents"] = list(bundle["turn_contents"])
        return True, ""

    fake_task = {
        "id": "mt_unit",
        "turns": ["turn one prompt", "turn two prompt"],
        "setup": {"fs": None},
        "provider": None,
        "plan_mode": False,
        "permission": "allow_all",
        "limits": {"max_tool_calls": 4, "wall_timeout_s": 5},
        "checks": [capturing_check],
    }
    monkeypatch.setattr(runner, "discover_tasks", lambda: [fake_task])
    monkeypatch.setattr(runner, "RESULTS_ROOT", tmp_path)

    results = runner.run_suite()
    r = results[0]
    assert r.passed
    # Top-level content is the LAST turn's reply — keeps the single-turn
    # contract for checks that use content_substr / content_regex.
    assert r.content == "second answer"
    assert captured["content"] == "second answer"
    # turn_contents is the full per-turn list, in order.
    assert captured["turn_contents"] == ["first answer", "second answer"]


def test_run_one_single_turn_exposes_one_element_turn_contents(
    monkeypatch, tmp_path, install_provider,  # noqa: F811
):
    """Single-turn tasks still work and `turn_contents` is a one-element
    list — the multi-turn change is strictly additive."""
    install_provider([ScriptedTurn(content_chunks=["only answer"])])

    captured: dict = {}

    def capturing_check(bundle):
        captured["turn_contents"] = list(bundle["turn_contents"])
        return True, ""

    fake_task = {
        "id": "single_unit",
        "prompt": "hi",
        "setup": {"fs": None},
        "provider": None,
        "plan_mode": False,
        "permission": "allow_all",
        "limits": {"max_tool_calls": 4, "wall_timeout_s": 5},
        "checks": [capturing_check],
    }
    monkeypatch.setattr(runner, "discover_tasks", lambda: [fake_task])
    monkeypatch.setattr(runner, "RESULTS_ROOT", tmp_path)

    runner.run_suite()
    assert captured["turn_contents"] == ["only answer"]


def test_run_suite_continues_when_worker_finishes_within_grace(monkeypatch, tmp_path):
    """A task can time out at wall_timeout_s but finish cleanly during
    the grace window — the suite should continue to the next task in
    that case, because the worker is no longer mutating shared state."""
    import threading as _threading

    from providers import get_active_provider, set_active_provider
    from providers.base import StreamChunk, Usage

    # Plenty of grace so the BlockingProvider's 1s internal wait fits.
    monkeypatch.setattr(runner, "WALL_TIMEOUT_GRACE_S", 3)

    class SlowProvider:
        name = "fake-slow"
        model = "fake-slow-v1"

        def stream_chat(self, messages, tools, num_ctx):
            # Blocks for 1s, then finishes cleanly. wall_timeout_s=0.1
            # means the task is flagged `timeout=True`, but the worker
            # exits during grace → suite continues.
            _threading.Event().wait(timeout=1.0)
            yield StreamChunk(content_delta="done")
            yield StreamChunk(done=True, usage=Usage(prompt_tokens=1, completion_tokens=1))

        def chat(self, messages, num_ctx):
            raise NotImplementedError

        def count_tokens(self, messages, tools=None):
            return 0

    original = get_active_provider()
    set_active_provider(SlowProvider())
    try:
        first = {
            "id": "slow_finishes",
            "prompt": "slow",
            "setup": {"fs": None},
            "permission": "allow_all",
            "limits": {"max_tool_calls": 4, "wall_timeout_s": 0.1},
            "checks": [lambda bundle: True],
        }
        second = {
            "id": "runs_after_grace",
            "prompt": "fast",
            "setup": {"fs": None},
            "permission": "allow_all",
            "limits": {"max_tool_calls": 4, "wall_timeout_s": 5},
            "checks": [lambda bundle: True],
        }
        monkeypatch.setattr(runner, "discover_tasks", lambda: [first, second])
        monkeypatch.setattr(runner, "RESULTS_ROOT", tmp_path)

        results = runner.run_suite()
        # Both tasks ran. The first timed out but its worker finished in
        # grace, so the second was allowed to start.
        assert [r.task_id for r in results] == ["slow_finishes", "runs_after_grace"]
        assert results[0].timeout is True
        assert results[0].worker_still_alive is False
    finally:
        set_active_provider(original)


def test_run_suite_stops_when_worker_is_still_alive(monkeypatch, tmp_path):
    """A worker thread that's still running after the grace window forces
    the suite to stop — otherwise a hung `run_agent` would race on
    process-global cwd / provider state and poison the next task's run."""
    import threading as _threading

    from providers import get_active_provider, set_active_provider
    from providers.base import StreamChunk

    # Grace=0 turns the timeout gate strict: the worker is 'still alive'
    # as soon as wall_timeout_s hits. Otherwise the BlockingProvider's
    # own internal wait would expire during the grace window and the
    # suite would continue — correct behavior, but not what this test
    # wants to exercise.
    monkeypatch.setattr(runner, "WALL_TIMEOUT_GRACE_S", 0)

    stop = _threading.Event()

    class BlockingProvider:
        name = "fake-blocking"
        model = "fake-blocking-v1"

        def stream_chat(self, messages, tools, num_ctx):
            stop.wait(timeout=5)
            yield StreamChunk(done=True)

        def chat(self, messages, num_ctx):
            raise NotImplementedError

        def count_tokens(self, messages, tools=None):
            return 0

    original = get_active_provider()
    set_active_provider(BlockingProvider())
    try:
        first = {
            "id": "fake_timeout",
            "prompt": "block",
            "setup": {"fs": None},
            "permission": "allow_all",
            "limits": {"max_tool_calls": 4, "wall_timeout_s": 1},
            "checks": [lambda bundle: True],
        }
        second = {
            "id": "must_not_run",
            "prompt": "should not execute after timeout",
            "setup": {"fs": None},
            "permission": "allow_all",
            "checks": [lambda bundle: False],
        }
        monkeypatch.setattr(runner, "discover_tasks", lambda: [first, second])
        monkeypatch.setattr(runner, "RESULTS_ROOT", tmp_path)

        results = runner.run_suite()
        assert [r.task_id for r in results] == ["fake_timeout"]
        assert results[0].timeout is True
    finally:
        stop.set()
        set_active_provider(original)


def test_run_suite_marks_provider_error_as_error(monkeypatch, tmp_path):
    from providers import get_active_provider, set_active_provider

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
        fake_task = {
            "id": "provider_down",
            "prompt": "do work",
            "setup": {"fs": None},
            "permission": "allow_all",
            "checks": [],
        }
        monkeypatch.setattr(runner, "discover_tasks", lambda: [fake_task])
        monkeypatch.setattr(runner, "RESULTS_ROOT", tmp_path)

        results = runner.run_suite()
        r = results[0]
        assert not r.passed
        assert r.error == "provider_error: fake-failing: simulated outage"
        assert r.content == ""
    finally:
        set_active_provider(original)


def test_run_matrix_runs_suite_per_model(monkeypatch, tmp_path):
    """Matrix mode: loop the suite once per (provider, model) pair and key
    the combined results by that tuple. build_provider is monkeypatched
    per model so each 'model' returns its own scripted reply — proves
    the runner actually swapped providers between rounds, not just re-ran
    on whatever was already active."""
    scripts = {
        "fake-a-v1": [ScriptedTurn(content_chunks=["reply A"])],
        "fake-b-v1": [ScriptedTurn(content_chunks=["reply B"])],
    }
    constructed: list[str] = []

    def fake_build_provider(name, model):
        constructed.append(f"{name}:{model}")
        return FakeProvider(scripts[model])

    monkeypatch.setattr(runner, "build_provider", fake_build_provider)

    fake_task = {
        "id": "matrix_t", "prompt": "x", "setup": {"fs": None},
        "provider": None, "plan_mode": False, "permission": "allow_all",
        "limits": {"max_tool_calls": 4, "wall_timeout_s": 5},
        "checks": [lambda bundle: (True, "")],
    }
    monkeypatch.setattr(runner, "discover_tasks", lambda: [fake_task])
    monkeypatch.setattr(runner, "RESULTS_ROOT", tmp_path)

    matrix = runner.run_matrix(
        models=[("openai-compat", "fake-a-v1"), ("openai-compat", "fake-b-v1")],
    )

    # Keys preserved in insertion order.
    assert list(matrix.keys()) == [
        ("openai-compat", "fake-a-v1"),
        ("openai-compat", "fake-b-v1"),
    ]
    # Each model's results show its scripted reply — confirming the
    # provider swap actually happened between rounds.
    assert matrix[("openai-compat", "fake-a-v1")][0].content == "reply A"
    assert matrix[("openai-compat", "fake-b-v1")][0].content == "reply B"
    # build_provider was called exactly once per (name, model) combo.
    assert constructed == ["openai-compat:fake-a-v1", "openai-compat:fake-b-v1"]


def test_cmd_eval_parses_m_flags_and_routes_to_matrix(monkeypatch, tmp_path):
    """`/eval -m p1:m1 -m p2:m2 task1` routes to run_matrix with the parsed
    models + task subset. Bare `/eval` or `/eval task1` (no `-m`) keeps
    calling run_suite, preserving single-model behavior."""
    from repl import commands as cmd_mod
    from repl.state import new_state

    called_matrix: dict = {}
    called_suite: dict = {}

    def fake_matrix(**kw):
        called_matrix.update(kw)
        return {}

    def fake_suite(**kw):
        called_suite.update(kw)
        return []

    # cmd_eval reloads evals.runner unless PYTEST_CURRENT_TEST is set;
    # pytest sets this env automatically so the reload path is skipped.
    monkeypatch.setattr(runner, "run_matrix", fake_matrix)
    monkeypatch.setattr(runner, "run_suite", fake_suite)
    monkeypatch.setattr(runner, "list_tasks", lambda: ["t1", "t2"])

    state = new_state()
    cmd_mod.cmd_eval(state, "-m openai-compat:gemma-4 -m ollama:qwen2.5:7b-instruct t1")

    assert called_suite == {}
    assert called_matrix["models"] == [
        ("openai-compat", "gemma-4"),
        ("ollama", "qwen2.5:7b-instruct"),
    ]
    assert called_matrix["task_ids"] == ["t1"]

    # A bare call with no -m goes through run_suite.
    called_matrix.clear()
    cmd_mod.cmd_eval(state, "t1 t2")
    assert called_matrix == {}
    assert called_suite["task_ids"] == ["t1", "t2"]


def test_list_tasks_returns_all_known_ids():
    """Every task module must be discoverable; legacy ids must be gone."""
    ids = set(runner.list_tasks())
    # Phase 1.
    assert {"commit_msg_from_diff", "find_symbol_all",
            "fix_failing_test", "tdd_new_fn"} <= ids
    # Phase 2 — capability gaps added later.
    assert {"edit_rename_symbol", "scoped_bugfix", "pagination_read",
            "plan_mode_plan_only", "glob_resolve_bare_name"} <= ids
    # Phase 3 — multi-turn + subagent usage.
    assert {"multi_turn_fix", "subagent_delegation"} <= ids
    # Legacy names that were removed.
    assert "find_string" not in ids
    assert "edit_rename" not in ids


def test_every_discovered_task_has_valid_shape(tmp_path):
    """Static sanity check on every TASK dict: required keys present,
    fixture dir exists if `setup.fs` is set, `checks` is a list of
    dicts or callables, `limits` are positive ints. Catches typos
    (e.g. `"setup": {"fx": ...}`) that would otherwise silently skip
    the fixture copy at runtime."""
    # Exactly one of `prompt` or `turns` must carry the user input.
    required_core = {"id", "setup", "provider", "plan_mode",
                     "permission", "limits", "checks"}
    fixtures_root = Path(runner.FIXTURES_ROOT)
    for task in runner.discover_tasks():
        missing = required_core - task.keys()
        assert not missing, f"task {task.get('id')!r} missing keys: {missing}"

        assert isinstance(task["id"], str) and task["id"]
        has_prompt = "prompt" in task
        has_turns = "turns" in task
        assert has_prompt ^ has_turns, (
            f"task {task['id']!r} must specify exactly one of "
            "`prompt` (single-turn) or `turns` (multi-turn)"
        )
        if has_prompt:
            assert isinstance(task["prompt"], str) and task["prompt"]
        else:
            turns = task["turns"]
            assert isinstance(turns, list) and turns, (
                f"task {task['id']!r} `turns` must be a non-empty list"
            )
            for i, t in enumerate(turns):
                assert isinstance(t, str) and t, (
                    f"task {task['id']!r} turn {i} is empty or not a string"
                )
        assert isinstance(task["plan_mode"], bool)

        fs_fixture = (task.get("setup") or {}).get("fs")
        if fs_fixture:
            assert (fixtures_root / fs_fixture).is_dir(), (
                f"task {task['id']!r} references missing fixture "
                f"{fs_fixture!r}"
            )

        limits = task["limits"]
        assert isinstance(limits.get("max_tool_calls"), int)
        assert isinstance(limits.get("wall_timeout_s"), int)
        assert limits["max_tool_calls"] > 0
        assert limits["wall_timeout_s"] > 0

        checks = task["checks"]
        assert isinstance(checks, list) and checks, f"{task['id']} has no checks"
        for check in checks:
            if callable(check):
                continue
            assert isinstance(check, dict) and "type" in check, (
                f"{task['id']} has malformed check: {check!r}"
            )
            assert check["type"] in checks_mod.CHECKS, (
                f"{task['id']} uses unknown check type: {check['type']!r}"
            )
