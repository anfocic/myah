# Myah

A hand-rolled agent harness, built to learn how agent harnesses work.

Python REPL that talks to Ollama by default, but also supports OpenAI-compatible servers and first-party OpenAI / Anthropic / DeepSeek adapters. It lets the model call tools, manages context pressure, and exposes the harness mechanics directly. ~2000 lines across ~60 files. Not a product.

## Run

Requires Python 3.11+ and either Ollama or credentials / base URLs for another supported provider.

```bash
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
ollama pull qwen/qwen3.5-9b
python main.py
```

Type `/help` in the REPL. Use `/model` to inspect or swap the active provider/model at runtime.

## Reading order

- `main.py` — REPL entry point
- `agent/loop.py` — chat/tool/chat cycle
- `agent/context.py` — trim, compact, microcompact
- `providers/` — provider adapters
- `tools/` — built-in tools the model can call

## License

[MIT](LICENSE)
