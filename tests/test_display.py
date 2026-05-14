import io

from rich.console import Console

import display.tools as tool_display
from display import StreamingMarkdown, on_tool_end, on_tool_start


def _console_and_buffer() -> tuple[Console, io.StringIO]:
    buf = io.StringIO()
    return Console(file=buf, force_terminal=False, width=80), buf


def test_streaming_markdown_writes_raw_content():
    console, buf = _console_and_buffer()
    renderer = StreamingMarkdown(console)

    renderer.update("Hello ")
    renderer.update("Hello world")
    renderer.finish("Hello world")

    exported = buf.getvalue()
    assert "Hello world" in exported
    assert exported.count("Hello world") == 1


def test_streaming_markdown_appends_deltas_without_duplication():
    console, buf = _console_and_buffer()
    renderer = StreamingMarkdown(console)

    renderer.update("alpha")
    renderer.update("alphabeta")
    renderer.update("alphabetagamma")
    renderer.finish("alphabetagamma")

    exported = buf.getvalue()
    assert exported.count("alphabetagamma") == 1


def test_streaming_markdown_finish_appends_trailing_newline_when_missing():
    console, buf = _console_and_buffer()
    renderer = StreamingMarkdown(console)

    renderer.finish("no trailing newline")

    assert buf.getvalue().endswith("\n")


def test_tool_callbacks_render_name_args_and_duration(monkeypatch):
    record = Console(record=True, force_terminal=False, width=120)
    monkeypatch.setattr(tool_display, "console", record)

    on_tool_start("bash", {"command": "pytest -q"}, {"duration_s": None})
    on_tool_end(
        "bash",
        {"command": "pytest -q"},
        "all good\n\nexit: 0",
        True,
        {"duration_s": 0.042},
    )

    exported = record.export_text()
    assert "BASH" in exported
    assert "pytest -q" in exported
    assert "exit 0" in exported
    assert "42ms" in exported
    # Phosphor glyphs frame the block
    assert "▶" in exported
    assert "⤷" in exported
    # bash stdout lands in its own panel
    assert "all good" in exported


def test_tool_callbacks_render_error_tails(monkeypatch):
    record = Console(record=True, force_terminal=False, width=120)
    monkeypatch.setattr(tool_display, "console", record)

    on_tool_end("bash", {}, "User denied this tool call.", False, {})
    on_tool_end("edit_file", {}, "Plan mode: tool call rejected.", False, {})
    on_tool_end("read_file", {}, "Tool raised: FileNotFoundError", False, {})

    exported = record.export_text()
    assert "denied by user" in exported
    assert "blocked · plan mode" in exported
    assert "Tool raised: FileNotFoundError" in exported


def test_tool_callbacks_render_edit_diff_summary(monkeypatch):
    record = Console(record=True, force_terminal=False, width=120)
    monkeypatch.setattr(tool_display, "console", record)

    on_tool_end(
        "edit_file",
        {"path": "x.py", "old_string": "old\n", "new_string": "new\n"},
        "Edited x.py: 1 replacement(s)",
        True,
        {"duration_s": 0.1},
    )

    exported = record.export_text()
    assert "+1/-1 lines" in exported
    assert "x.py" in exported
    # Diff panel still renders
    assert "--- x.py" in exported
    assert "+new" in exported
