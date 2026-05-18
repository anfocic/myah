"""Clipboard image extraction — tests for the macOS osascript-backed
`get_clipboard_image` helper. Subprocess invocations are mocked so the
suite stays hermetic and runs on Linux/CI without an actual clipboard."""
from __future__ import annotations

import base64
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

from tools.clipboard import IMAGE_MAX_BYTES, get_clipboard_image


def _png_bytes(n: int) -> bytes:
    """Synthetic PNG-shaped blob. The real file is irrelevant — the
    function never inspects the body, only the size."""
    return b"\x89PNG\r\n\x1a\n" + b"\x00" * (n - 8)


def test_no_image_in_clipboard_returns_none(tmp_path):
    """When osascript exits non-zero for every probed format, the helper
    must return None (silent no-op) — Cmd+V text paste continues to
    work because we never touched the prompt's input buffer."""

    def fake_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="")

    with patch("tools.clipboard.subprocess.run", side_effect=fake_run):
        assert get_clipboard_image() is None


def _format_aware_scaffold(monkeypatch, payloads: dict[str, bytes]):
    """Wire `_tmp_path_for` + `subprocess.run` so each osascript probe
    that succeeds also lands the matching payload at the tmpfile path.
    `payloads` maps suffix (.png/.jpg) to the bytes the test wants to
    surface for that format; missing keys make that probe fail."""
    state: dict[str, Path] = {}

    def fake_tmp_path(suffix: str) -> Path:
        # Honor the requested suffix so the script's path matches what
        # the run hook expects to populate.
        import tempfile as _tf
        fd, p = _tf.mkstemp(suffix=suffix, prefix="mia-clip-test-")
        os.close(fd)
        state["latest_" + suffix] = Path(p)
        return Path(p)

    def fake_run(cmd, *args, **kwargs):
        joined = " ".join(cmd) if isinstance(cmd, list) else cmd
        if "PNGf" in joined and ".png" in payloads:
            state["latest_.png"].write_bytes(payloads[".png"])
            return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")
        if "JPEG" in joined and ".jpg" in payloads:
            state["latest_.jpg"].write_bytes(payloads[".jpg"])
            return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="")

    monkeypatch.setattr("tools.clipboard._tmp_path_for", fake_tmp_path)
    return fake_run


def test_png_clipboard_returns_base64_and_media_type(monkeypatch):
    """A PNG on the clipboard should be base64-encoded and tagged with
    image/png. The helper writes osascript output to a tmpfile then
    reads it back."""
    payload = _png_bytes(1024)
    fake_run = _format_aware_scaffold(monkeypatch, {".png": payload})
    with patch("tools.clipboard.subprocess.run", side_effect=fake_run):
        result = get_clipboard_image()

    assert result is not None
    b64, media_type, size = result
    assert media_type == "image/png"
    assert size == len(payload)
    assert base64.b64decode(b64) == payload


def test_jpeg_clipboard_returns_jpeg_media_type(monkeypatch):
    payload = b"\xff\xd8\xff\xe0" + b"\x00" * 1020
    fake_run = _format_aware_scaffold(monkeypatch, {".jpg": payload})
    with patch("tools.clipboard.subprocess.run", side_effect=fake_run):
        result = get_clipboard_image()

    assert result is not None
    _b64, media_type, _size = result
    assert media_type == "image/jpeg"


def test_oversized_image_is_rejected(monkeypatch):
    """Anything above IMAGE_MAX_BYTES is dropped — protecting the
    session file and the model's context window from multi-megabyte
    base64 payloads."""
    payload = _png_bytes(IMAGE_MAX_BYTES + 1)
    fake_run = _format_aware_scaffold(monkeypatch, {".png": payload})
    with patch("tools.clipboard.subprocess.run", side_effect=fake_run):
        result = get_clipboard_image()

    assert result is None


def test_tiff_is_skipped(tmp_path, monkeypatch):
    """TIFF arrives on the macOS clipboard for some Preview copies, but
    no vision API accepts it without conversion. We skip TIFF entirely
    rather than ship invalid data."""

    def fake_run(cmd, *args, **kwargs):
        joined = " ".join(cmd) if isinstance(cmd, list) else cmd
        # All probes fail — TIFF is not even attempted.
        if "TIFF" in joined:
            raise AssertionError("TIFF should never be probed")
        return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="")

    with patch("tools.clipboard.subprocess.run", side_effect=fake_run):
        assert get_clipboard_image() is None


def test_osascript_missing_is_silent(monkeypatch):
    """Linux / CI / minimal macOS — osascript binary is absent. The
    helper must not raise; it just returns None."""

    def fake_run(cmd, *args, **kwargs):
        raise FileNotFoundError("osascript not found")

    with patch("tools.clipboard.subprocess.run", side_effect=fake_run):
        assert get_clipboard_image() is None


def test_osascript_timeout_is_silent(monkeypatch):
    """A hung osascript call must not freeze the REPL. Timeout → None."""

    def fake_run(cmd, *args, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 2)

    with patch("tools.clipboard.subprocess.run", side_effect=fake_run):
        assert get_clipboard_image() is None
