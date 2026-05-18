"""Provider-side translation of internal image content blocks.

The internal format (produced when the user pastes an image with Ctrl+V):

    {"role": "user", "content": [
        {"type": "text", "text": "what is this?"},
        {"type": "image", "source": {"type": "base64",
                                     "media_type": "image/png",
                                     "data": "<b64>"}},
    ]}

Each adapter has a different wire format for vision input — these tests
pin the per-provider translation."""
from __future__ import annotations


def _user_with_image(b64: str = "AAAA", media: str = "image/png") -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": "what is this?"},
            {"type": "image", "source": {
                "type": "base64", "media_type": media, "data": b64,
            }},
        ],
    }


# ---------- Anthropic ----------

def test_anthropic_passes_internal_image_blocks_through():
    """Anthropic's content block format IS our internal format — text +
    image with `{type, source: {type:base64, media_type, data}}`. The
    translator must keep the list shape intact (not collapse to a
    string), otherwise the SDK rejects the request."""
    from providers.anthropic_adapter import _translate_messages

    _, msgs = _translate_messages([_user_with_image()])
    assert len(msgs) == 1
    out = msgs[0]
    assert out["role"] == "user"
    assert isinstance(out["content"], list)
    types = [b["type"] for b in out["content"]]
    assert types == ["text", "image"]
    img = out["content"][1]
    assert img["source"]["media_type"] == "image/png"
    assert img["source"]["data"] == "AAAA"


# ---------- OpenAI-compat ----------

def test_openai_compat_translates_image_to_image_url_data_uri():
    """OpenAI-style servers want `{type: image_url, image_url: {url:
    data:image/png;base64,...}}`. The translator must build the data
    URI from media_type + base64 data."""
    from providers.openai_compat import _translate_messages

    out = _translate_messages([_user_with_image(b64="ZZZZ", media="image/jpeg")])
    assert len(out) == 1
    msg = out[0]
    assert msg["role"] == "user"
    assert isinstance(msg["content"], list)
    text_block, img_block = msg["content"]
    assert text_block == {"type": "text", "text": "what is this?"}
    assert img_block["type"] == "image_url"
    assert img_block["image_url"]["url"] == "data:image/jpeg;base64,ZZZZ"


def test_openai_compat_plain_string_user_message_unaffected():
    """The existing string-content path must keep working — only list
    content gets the image translation branch."""
    from providers.openai_compat import _translate_messages

    out = _translate_messages([{"role": "user", "content": "hello"}])
    assert out == [{"role": "user", "content": "hello"}]


# ---------- Ollama ----------

def test_ollama_translates_image_to_images_top_level_key():
    """Ollama expects images as a top-level `images` array of base64
    strings on the message, and `content` to be a plain string. The
    translator must pull image blocks out of the list, concat text
    blocks into the string, and stash the base64 data under `images`."""
    from providers.ollama_adapter import _strip_internal_tool_calls

    out = _strip_internal_tool_calls([_user_with_image(b64="BBBB")])
    assert len(out) == 1
    msg = out[0]
    assert msg["role"] == "user"
    assert msg["content"] == "what is this?"
    assert msg.get("images") == ["BBBB"]


def test_ollama_plain_string_user_message_unaffected():
    from providers.ollama_adapter import _strip_internal_tool_calls

    out = _strip_internal_tool_calls([{"role": "user", "content": "hi there"}])
    assert out == [{"role": "user", "content": "hi there"}]


def test_ollama_multiple_text_blocks_concatenated():
    """If the user message somehow has multiple text blocks (rare but
    possible after future composition), concat them with newline
    separators into the string content."""
    from providers.ollama_adapter import _strip_internal_tool_calls

    msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
            {"type": "image", "source": {"data": "X", "media_type": "image/png"}},
        ],
    }
    out = _strip_internal_tool_calls([msg])
    assert out[0]["content"] == "first\nsecond"
    assert out[0]["images"] == ["X"]
