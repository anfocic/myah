"""Mia's REPL layer. Each module owns one concern:

- `state`         — the `State` TypedDict + snapshot-stack constant
- `console`       — the shared rich.Console singleton
- `persistence`   — session + input-history load/save
- `tool_registry` — tool schemas + execute_tool dispatcher factory
- `ui`            — prompt chrome, hint line, context-bar tag, tab completer
- `commands`      — slash commands + dispatcher (control plane per §22)

`main.py` composes these into the actual REPL loop. Nothing here reaches
into agent.py's turn loop — all coupling is one-way (REPL calls agent,
not the other way around)."""
