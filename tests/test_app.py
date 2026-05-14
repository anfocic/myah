"""ReplApp wiring — boot screen, the submit / slash / turn-worker paths.

The full `prompt_toolkit.Application` event loop isn't driven here (that's the
manual smoke matrix); these exercise the orchestration methods directly with a
stubbed provider and `run_agent`."""

import threading
import time
from types import SimpleNamespace

import pytest
from prompt_toolkit.history import InMemoryHistory

import permissions
from providers import get_active_provider, set_active_provider
from repl.console import console
from repl.state import new_state


@pytest.fixture
def replapp(monkeypatch):
    """A constructed ReplApp with a stub provider, stubbed run_agent/trim, and
    an in-memory input history. Restores the global console proxy + permission
    asker on teardown so other tests aren't affected."""
    import repl.app as app_mod

    stub_provider = SimpleNamespace(name="fake", model="fake-model", context_size=4096)
    original_provider = get_active_provider()
    set_active_provider(stub_provider)

    original_inner = console._inner
    original_asker = permissions._ask_permission

    monkeypatch.setattr(app_mod, "FileHistory", lambda _path: InMemoryHistory())

    def fake_run_agent(user_input, *a, **k):
        k["console"].print(f"<< reply to {user_input}")
        history = (k.get("history") or []) + [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": "reply"},
        ]
        return ("reply", history, 123, {"ttft_ms": 10, "tok_per_s": 5.0})

    monkeypatch.setattr(app_mod, "run_agent", fake_run_agent)
    monkeypatch.setattr(app_mod, "trim_history", lambda hist, *a, **k: (hist, []))

    state = new_state()
    instance = app_mod.ReplApp(state, resume=False)
    yield instance

    console._set_inner(original_inner)
    permissions.set_permission_asker(original_asker)
    set_active_provider(original_provider)


def _buffer_text(instance) -> str:
    return "\n".join(instance.buffer.lines + [instance.buffer._partial])


def test_boot_screen_renders_into_the_buffer(replapp):
    # The boot screen prints lazily on the first render (once the real pane
    # width is known); call it directly here since no Application is running.
    replapp._print_boot_screen(resume=False)
    text = _buffer_text(replapp)
    assert "INITIALIZE" in text
    assert "READY" in text
    assert "fake-model" in text  # provider checklist line


def test_normal_input_runs_a_turn_worker(replapp):
    replapp._on_accept(SimpleNamespace(text="explain the loop"))
    replapp.turn_thread.join(timeout=3.0)
    assert not replapp.turn_thread.is_alive()
    text = _buffer_text(replapp)
    assert "TRANSMISSION" in text  # transmission header printed
    assert "<< reply to explain the loop" in text  # stubbed run_agent output
    assert replapp.sess_state == "READY"  # rail state reset after the turn
    assert len(replapp.state["history"]) == 2
    assert replapp.state["turn_history"]  # metrics stashed for /stats


def test_exit_input_requests_exit(replapp, monkeypatch):
    called = []
    monkeypatch.setattr(replapp, "_request_exit", lambda: called.append(True))
    replapp._on_accept(SimpleNamespace(text="exit"))
    assert called == [True]


def test_slash_command_is_dispatched(replapp):
    replapp._on_accept(SimpleNamespace(text="/help"))
    text = _buffer_text(replapp)
    assert "COMMANDS" in text  # cmd_help output landed in the buffer


def test_clear_slash_wipes_the_scrollback_buffer(replapp):
    replapp._print_boot_screen(resume=False)
    assert "INITIALIZE" in _buffer_text(replapp)  # boot screen is there
    replapp._on_accept(SimpleNamespace(text="/clear"))
    # /clear wiped history *and* the scrollback; only the post-dispatch
    # spacing/notice lines remain
    assert "INITIALIZE" not in _buffer_text(replapp)


def test_input_is_rejected_while_a_turn_runs(replapp):
    gate = threading.Event()

    def slow_worker(user_input):
        gate.wait(timeout=2.0)

    replapp._turn_worker = slow_worker
    replapp._start_turn("first")
    time.sleep(0.05)
    assert replapp._turn_running()

    replapp._on_accept(SimpleNamespace(text="second"))
    assert "turn in progress" in _buffer_text(replapp)

    gate.set()
    replapp.turn_thread.join(timeout=2.0)
