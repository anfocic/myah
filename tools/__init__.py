"""Tool implementations — the functions behind the tool schemas declared
in `repl/tool_registry.py`. Each submodule owns one or more related tools:

- `files`   — read_file, write_file, edit_file
- `search`  — glob, grep
- `web_search` — live web search via Brave Search API
- `bash`    — shell-out
- `git`     — git_checkout
- `harness` — harness_info / harness_snapshot
- `utils`   — get_current_time
"""
