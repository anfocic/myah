"""UI layer tests — just the pieces that can be exercised without a TTY.

`SlashCompleter` is a pure `Completer`: feed it a `Document`, collect the
yielded `Completion`s. `build_prompt` is also pure: feed it a state dict,
inspect the returned `FormattedText` tuples.

Everything that needs a real terminal (history recall via arrow keys,
Ctrl+C handling, prompt pinning under patch_stdout) lives in the manual
smoke matrix in docs/CONCEPTS.md — pexpect would let us test them but
the cost/benefit doesn't justify it for a pedagogical harness."""
import pytest
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

from repl import ui
from repl.state import new_state
from repl.ui import SlashCompleter, build_prompt


@pytest.fixture(autouse=True)
def _stub_branch(monkeypatch):
    """Isolate prompt tests from the real repo state: a branch name like
    `feat/plan-something` would otherwise leak "plan" into the rendered
    prompt and break `test_build_prompt_omits_badges_when_clean`."""
    monkeypatch.setattr(ui, "_current_branch", lambda: None)


def _run_completer(completer: SlashCompleter, text: str) -> list[str]:
    doc = Document(text)
    return [c.text for c in completer.get_completions(doc, CompleteEvent())]


def test_slash_completer_yields_commands_on_slash_prefix():
    c = SlashCompleter({"/help": None, "/clear": None, "/history": None})
    completions = _run_completer(c, "/h")
    assert "/help" in completions
    assert "/history" in completions
    assert "/clear" not in completions


def test_slash_completer_ignores_prose():
    c = SlashCompleter({"/help": None, "/clear": None})
    assert _run_completer(c, "hel") == []
    assert _run_completer(c, "") == []


def _rendered(formatted_text) -> str:
    return "".join(text for _, text in formatted_text)


def test_build_prompt_includes_plan_badge_when_plan_mode_on():
    state = new_state()
    state["plan_mode"] = True
    rendered = _rendered(build_prompt(state))
    assert "plan" in rendered
    assert "You" in rendered
    assert "›" in rendered


def test_build_prompt_omits_badges_when_clean():
    state = new_state()
    state["plan_mode"] = False
    state["debug"] = False
    rendered = _rendered(build_prompt(state))
    assert "plan" not in rendered
    assert "debug" not in rendered
