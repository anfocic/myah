"""Locate every file that defines or references `READ_ONLY_TOOLS`.

Hardened successor to the legacy `find_string` task. Stricter in two ways:
- Demands ALL five real hit sites, not two out of four. A model that reports
  only part of the answer is wrong, even if the text looks confident.
- Bounds the tool budget (<=6 calls). A model that reads every file instead
  of grepping is also wrong, even if the final answer is correct.

No fixture — runs in the repo cwd. That means the ground truth (which files
contain `READ_ONLY_TOOLS`) is pinned to this repo's current layout. If a
future refactor moves the symbol, update the regex list below.
"""

TASK = {
    "id": "find_symbol_all",
    "prompt": (
        "List every file in this codebase that defines or references the "
        "identifier `READ_ONLY_TOOLS`. Use the grep tool. Report the file "
        "paths in your answer."
    ),
    "setup": {"fs": None},
    "provider": None,
    "plan_mode": False,
    "permission": "allow_all",
    "limits": {"max_tool_calls": 6, "wall_timeout_s": 60},
    "checks": [
        {
            "type": "tool_trace",
            "must_call": ["grep"],
            "must_not_call": ["write_file", "edit_file", "bash"],
            "call_count_max": 6,
        },
        {"type": "content_regex", "pattern": r"agent/__init__\.py"},
        {"type": "content_regex", "pattern": r"agent/loop\.py"},
        {"type": "content_regex", "pattern": r"agent/system_prompt\.py"},
        {"type": "content_regex", "pattern": r"evals/runner\.py"},
        {"type": "content_regex", "pattern": r"tests/test_web_search\.py"},
    ],
}
