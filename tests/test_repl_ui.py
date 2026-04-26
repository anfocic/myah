"""UI layer tests — just the pieces that can be exercised without a TTY.

`SlashCompleter` is a pure `Completer`: feed it a `Document`, collect the
yielded `Completion`s. `build_prompt` is also pure: feed it a state dict,
inspect the returned `FormattedText` tuples.

Everything that needs a real terminal (history recall via arrow keys,
Ctrl+C handling, prompt pinning under patch_stdout) is exercised by a
manual smoke matrix — pexpect would let us automate it but the
cost/benefit doesn't justify it for a pedagogical harness."""
from types import SimpleNamespace

import pytest
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

from repl import ui
from repl.state import new_state
from repl.ui import (
    SlashCompleter,
    build_prompt,
    build_rprompt,
    build_turn_footer,
    build_turn_header,
)


@pytest.fixture(autouse=True)
def _stub_branch(monkeypatch):
    """Isolate prompt tests from the real repo state: a branch name like
    `feat/plan-something` would otherwise leak "plan" into the rendered
    prompt and break `test_build_prompt_omits_badges_when_clean`."""
    monkeypatch.setattr(ui, "_current_branch", lambda: None)
    monkeypatch.setattr(ui, "NUM_CTX", 4096)
    monkeypatch.setattr(
        ui,
        "get_active_provider",
        lambda: SimpleNamespace(name="ollama", model="qwen2.5:7b-instruct"),
    )


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
    assert "You" in rendered
    assert "›" in rendered
    assert "plan" not in rendered


def test_build_prompt_omits_badges_when_clean():
    state = new_state()
    state["plan_mode"] = False
    state["debug"] = False
    rendered = _rendered(build_prompt(state))
    assert "plan" not in rendered
    assert "debug" not in rendered


def test_build_rprompt_order_is_branch_pct_model(monkeypatch):
    """Layout contract: `branch · ctx% · provider:model`, left-to-right."""
    monkeypatch.setattr(ui, "_current_branch", lambda: "feat/rprompt-pass")
    state = new_state()
    state["ctx_used"] = 1024
    rendered = _rendered(build_rprompt(state))
    assert rendered == "feat/rprompt-pass · 25% · ollama:qwen2.5:7b-instruct"


def test_build_rprompt_keeps_full_model_name_with_org(monkeypatch):
    """Unlike the short model label, the rprompt shows `provider:model`
    verbatim — including any `org/` prefix like `google/gemma-4-e4b` —
    so the user can tell at a glance which build is loaded."""
    monkeypatch.setattr(
        ui,
        "get_active_provider",
        lambda: SimpleNamespace(name="openai-compat", model="google/gemma-4-e4b"),
    )
    state = new_state()
    rendered = _rendered(build_rprompt(state))
    assert "openai-compat:google/gemma-4-e4b" in rendered


def test_build_rprompt_omits_branch_segment_when_not_in_repo():
    state = new_state()
    rendered = _rendered(build_rprompt(state))
    # No branch segment → pct leads, one ` · ` separator before the model.
    assert rendered == "0% · ollama:qwen2.5:7b-instruct"


def test_build_rprompt_gradient_shifts_with_ctx():
    """Smoke test for the HSL gradient on the pct segment: a 2%-fill
    prompt and a 93%-fill prompt must render with different foreground
    colors so the user can perceive pressure without reading the number."""
    state_low = new_state()
    state_low["ctx_used"] = 100  # ~2%
    state_high = new_state()
    state_high["ctx_used"] = 3800  # ~93%

    def style_of_pct(segments):
        return next(style for style, text in segments if text.endswith("%"))

    low = style_of_pct(build_rprompt(state_low))
    high = style_of_pct(build_rprompt(state_high))
    assert low != high
    assert low.startswith("fg:#")
    assert high.startswith("fg:#")


def test_build_turn_header_uses_turn_count_and_badges(monkeypatch):
    monkeypatch.setattr(ui, "_current_branch", lambda: "main")
    state = new_state()
    state["history"] = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "one"},
        {"role": "system", "content": "summary"},
        {"role": "user", "content": "two"},
        {"role": "assistant", "content": "two"},
    ]
    state["ctx_used"] = 2048
    state["debug"] = True
    header = build_turn_header(state)
    assert "Turn 3" in header
    assert "main" in header
    assert "ollama:qwen2.5:7b-instruct" in header
    assert "2,048/4,096" in header
    assert "debug" in header


def test_build_turn_footer_includes_ttft_and_rate():
    footer = build_turn_footer(
        3072,
        4096,
        3.4,
        {"ttft_ms": 220, "tok_per_s": 87.2},
    )
    assert "3,072/4,096" in footer
    assert "75%" in footer
    assert "3.4s" in footer
    assert "ttft 220ms" in footer
    assert "87 tok/s" in footer
