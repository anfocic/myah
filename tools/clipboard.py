"""macOS clipboard image extraction via osascript.

Why osascript and not pbpaste? `pbpaste` only emits text — for image data
we need AppleScript's `the clipboard as «class PNGf»` (or `JPEG`) which
returns raw image bytes that we write to a tmpfile, read back, and
base64-encode for the provider.

No Pillow, no third-party dependencies. The probe is limited to PNG and
JPEG because every vision API in use (Anthropic, OpenAI, OpenAI-compat)
accepts those formats natively; TIFF is rare on the clipboard and would
need conversion we cannot do without a real image library."""
from __future__ import annotations

import base64
import os
import subprocess
import tempfile
from pathlib import Path

IMAGE_MAX_BYTES = 3_000_000  # 3MB — keeps the session file from ballooning

# (osascript class identifier, MIME type, file suffix) — order is probe
# preference. PNG first because macOS screenshots default to PNG.
_FORMATS: list[tuple[str, str, str]] = [
    ("«class PNGf»", "image/png", ".png"),
    ("JPEG picture", "image/jpeg", ".jpg"),
]


def _tmp_path_for(suffix: str) -> Path:
    """Return a fresh tmpfile path with the given suffix. Pulled out so
    tests can monkeypatch the destination to a controlled file."""
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="mia-clip-")
    os.close(fd)
    return Path(path)


def _try_format(class_id: str, suffix: str) -> Path | None:
    """Ask osascript for the clipboard contents as `class_id` and write
    to a tmpfile. Returns the path on success, None on any failure
    (clipboard doesn't hold that class, osascript missing, timeout)."""
    target = _tmp_path_for(suffix)
    script = (
        f'try\n'
        f'  set imgData to (the clipboard as {class_id})\n'
        f'  set fh to open for access POSIX file "{target}" with write permission\n'
        f'  write imgData to fh\n'
        f'  close access fh\n'
        f'on error\n'
        f'  try\n'
        f'    close access POSIX file "{target}"\n'
        f'  end try\n'
        f'  error\n'
        f'end try'
    )
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        target.unlink(missing_ok=True)
        return None
    if result.returncode != 0:
        target.unlink(missing_ok=True)
        return None
    return target


def get_clipboard_image() -> tuple[str, str, int] | None:
    """Return (base64_data, media_type, size_bytes) for the first
    matching image on the macOS clipboard, or None.

    Returns None on every non-image clipboard state: empty clipboard,
    text-only paste, TIFF-only paste, oversized image, osascript
    unavailable (non-macOS or stripped install), timeout. The REPL
    falls back to its normal Cmd+V text-paste in all of those cases."""
    for class_id, media_type, suffix in _FORMATS:
        path = _try_format(class_id, suffix)
        if path is None:
            continue
        try:
            data = path.read_bytes()
        except OSError:
            path.unlink(missing_ok=True)
            continue
        path.unlink(missing_ok=True)
        if not data:
            continue
        if len(data) > IMAGE_MAX_BYTES:
            return None
        return base64.b64encode(data).decode("ascii"), media_type, len(data)
    return None
