"""Tests for /export — plain markdown transcript writer."""
from pathlib import Path

from repl.export import export_conversation


def _history():
    return [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
        {"role": "user", "content": "what's the weather"},
        {"role": "assistant", "content": "I don't know, no tools."},
    ]


def test_export_writes_transcript(tmp_path):
    path = tmp_path / "out.md"
    result = export_conversation(_history(), "qwen2.5:7b", "ollama", str(path))
    assert result == str(path.resolve())
    body = path.read_text()
    assert "# Mia conversation export" in body
    assert "### user" in body and "### assistant" in body
    assert "hello" in body and "hi there" in body


def test_export_empty_history_refuses(tmp_path):
    path = tmp_path / "out.md"
    result = export_conversation([], "qwen2.5:7b", "ollama", str(path))
    assert result.startswith("Refused")
    assert not path.exists()


def test_export_default_filename_in_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = export_conversation(_history(), "qwen2.5:7b", "ollama", None)
    assert Path(result).exists()
    assert Path(result).name.startswith("mia-transcript-")
    assert Path(result).suffix == ".md"
    assert Path(result).parent.resolve() == tmp_path.resolve()


def test_export_into_directory_gets_default_name(tmp_path):
    result = export_conversation(_history(), "qwen2.5:7b", "ollama", str(tmp_path))
    assert Path(result).parent.resolve() == tmp_path.resolve()
    assert Path(result).name.startswith("mia-transcript-")


def test_export_creates_parent_dirs(tmp_path):
    nested = tmp_path / "deep" / "nested" / "out.md"
    result = export_conversation(_history(), "qwen2.5:7b", "ollama", str(nested))
    assert Path(result).exists()
    assert nested.exists()


def test_export_includes_provider_and_model_header(tmp_path):
    path = tmp_path / "out.md"
    export_conversation(_history(), "gpt-4o-mini", "openai", str(path))
    body = path.read_text()
    assert "gpt-4o-mini" in body and "openai" in body


def test_export_preserves_system_summary_notes(tmp_path):
    history = [
        {"role": "system", "content": "Earlier summary: discussed X and Y."},
        {"role": "user", "content": "continue"},
        {"role": "assistant", "content": "ok"},
    ]
    path = tmp_path / "out.md"
    export_conversation(history, "x", "y", str(path))
    body = path.read_text()
    assert "### system note" in body
    assert "Earlier summary" in body
