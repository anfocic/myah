import io
import os

from rich.console import Console

import display.streaming as streaming_display
import display.tools as tool_display
from display import StreamingMarkdown, on_tool_end, on_tool_start


def _console_and_buffer() -> tuple[Console, io.StringIO]:
    buf = io.StringIO()
    return Console(file=buf, force_terminal=False, width=80), buf


def test_streaming_markdown_short_reply_stays_in_markdown_mode(monkeypatch):
    monkeypatch.setattr(streaming_display, "_REPAINT_INTERVAL_MS", 0)
    monkeypatch.setattr(
        streaming_display.shutil,
        "get_terminal_size",
        lambda: os.terminal_size((80, 24)),
    )
    console, buf = _console_and_buffer()
    renderer = StreamingMarkdown(console)

    renderer.update("Hello")
    renderer.finish("Hello")

    assert renderer._mode == "markdown"
    assert renderer._printed_len == 0
    assert "Hello" in buf.getvalue()


def test_streaming_markdown_switches_to_append_mode_for_tall_output(monkeypatch):
    monkeypatch.setattr(streaming_display, "_REPAINT_INTERVAL_MS", 0)
    monkeypatch.setattr(
        streaming_display.shutil,
        "get_terminal_size",
        lambda: os.terminal_size((80, 4)),
    )
    console, buf = _console_and_buffer()
    renderer = StreamingMarkdown(console)
    monkeypatch.setattr(
        renderer,
        "_render_and_count",
        lambda content: (content, 99),
    )

    renderer.update("one\ntwo\nthree")
    renderer.update("one\ntwo\nthree\nfour")

    assert renderer._mode == "append"
    assert renderer._printed_len == len("one\ntwo\nthree\nfour")
    assert buf.getvalue().endswith("one\ntwo\nthree\nfour")


def test_streaming_markdown_finish_does_not_duplicate_after_append_fallback(monkeypatch):
    monkeypatch.setattr(streaming_display, "_REPAINT_INTERVAL_MS", 0)
    monkeypatch.setattr(
        streaming_display.shutil,
        "get_terminal_size",
        lambda: os.terminal_size((80, 4)),
    )
    console, buf = _console_and_buffer()
    renderer = StreamingMarkdown(console)
    content = "alpha\nbeta\ngamma"
    monkeypatch.setattr(
        renderer,
        "_render_and_count",
        lambda content: (content, 99),
    )

    renderer.update(content)
    renderer.finish(content)

    exported = buf.getvalue()
    assert renderer._mode == "append"
    assert exported.count(content) == 1
    assert exported.endswith(content + "\n")


def test_tool_callbacks_render_ids_duration_and_bash_summary(monkeypatch):
    record = Console(record=True, force_terminal=False, width=120)
    monkeypatch.setattr(tool_display, "console", record)

    on_tool_start("bash", {"command": "pytest -q"}, {"tool_id": "T03"})
    on_tool_end(
        "bash",
        {"command": "pytest -q"},
        "all good\n\nexit: 0",
        True,
        {"tool_id": "T03", "duration_s": 0.042},
    )

    exported = record.export_text()
    assert "T03" in exported
    assert "bash" in exported
    assert "pytest -q" in exported
    assert "exit 0" in exported
    assert "42ms" in exported


def test_tool_callbacks_render_edit_diff_summary(monkeypatch):
    record = Console(record=True, force_terminal=False, width=120)
    monkeypatch.setattr(tool_display, "console", record)

    on_tool_end(
        "edit_file",
        {"path": "x.py", "old_string": "old\n", "new_string": "new\n"},
        "Edited x.py: 1 replacement(s)",
        True,
        {"tool_id": "T01", "duration_s": 0.1},
    )

    exported = record.export_text()
    assert "T01" in exported
    assert "+1/-1 lines" in exported
    assert "x.py" in exported
    assert "--- x.py" in exported
    assert "+new" in exported
