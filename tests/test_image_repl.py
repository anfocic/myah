"""REPL-side wiring for Ctrl+V image paste.

Two layers are unit-testable: the pure `compose_user_message` helper
that builds the message list from text + optional pending image, and
the prompt floor's indicator when an image is staged.

The Ctrl+V keybinding itself lives inside `repl.app.App` and is
exercised live (no headless prompt_toolkit Application in tests)."""
from __future__ import annotations

from repl.state import new_state
from repl.ui import build_prompt, compose_user_message


def test_new_state_has_no_pending_image_keys():
    """Pending image state is transient — fresh state must not carry
    one over from a prior session restore."""
    state = new_state()
    assert "_pending_image" not in state
    assert "_pending_image_size" not in state


def test_compose_user_message_plain_text_returns_string():
    """No pending image → caller still passes a plain string downstream
    so existing tests / providers see no shape change."""
    msg = compose_user_message("hello", pending_image=None)
    assert msg == "hello"


def test_compose_user_message_with_image_builds_block_list():
    """A pending image (base64, media_type) becomes a list with a text
    block and an image block in our internal format. Anthropic accepts
    this verbatim; the openai-compat and ollama adapters rewrite it."""
    msg = compose_user_message(
        "what's in the screenshot?",
        pending_image=("AAAA", "image/png"),
    )
    assert isinstance(msg, list)
    assert msg[0] == {"type": "text", "text": "what's in the screenshot?"}
    assert msg[1]["type"] == "image"
    assert msg[1]["source"] == {
        "type": "base64",
        "media_type": "image/png",
        "data": "AAAA",
    }


def test_compose_user_message_with_image_and_empty_text():
    """User can paste an image and hit Enter without typing — the text
    block is empty but the message still carries the image so the
    model has something to react to."""
    msg = compose_user_message("", pending_image=("BBBB", "image/jpeg"))
    assert isinstance(msg, list)
    assert msg[0] == {"type": "text", "text": ""}
    assert msg[1]["source"]["media_type"] == "image/jpeg"


def test_build_prompt_shows_image_indicator_when_pending():
    """The Phosphor prompt floor prepends `[img NNk]` in cyan when a
    pasted image is waiting to be sent."""
    state = new_state()
    state["_pending_image"] = "X" * 100  # base64 payload
    state["_pending_image_size"] = 250_000
    fragments = build_prompt(state)
    rendered = "".join(text for _style, text in fragments)
    assert "img" in rendered
    assert "245" in rendered or "244" in rendered  # ~245KB displayed


def test_build_prompt_no_indicator_without_pending_image():
    """No pending image → prompt floor is unchanged from the no-image
    baseline. Tests pin this so a stray render path doesn't print a
    blank `[img 0KB]` in the common case."""
    state = new_state()
    fragments = build_prompt(state)
    rendered = "".join(text for _style, text in fragments)
    assert "img" not in rendered.lower()
