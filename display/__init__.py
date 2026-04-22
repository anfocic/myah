"""Rich display package for the REPL.

Public surface stays intentionally small and stable so the rest of the
codebase can keep importing from `display` while the implementation is
split by mechanism under this package.
"""

from display.previews import (
    _parse_web_results,
    build_unified_diff,
    render_diff,
    render_file_preview,
    render_web_search_results,
)
from display.streaming import StreamingMarkdown
from display.tools import on_tool_end, on_tool_start

__all__ = [
    "StreamingMarkdown",
    "build_unified_diff",
    "on_tool_end",
    "on_tool_start",
    "render_diff",
    "render_file_preview",
    "render_web_search_results",
    "_parse_web_results",
]
