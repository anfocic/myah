# Myah

A hand-rolled agent harness, built to learn how agent harnesses work.

Python REPL that talks to Ollama by default, but also supports OpenAI-compatible servers and first-party OpenAI / Anthropic / DeepSeek adapters. It lets the model call tools, manages context pressure, and exposes the harness mechanics directly. ~2000 lines across ~60 files. Not a product.

The stable architecture is written up in [`docs/HANDBOOK.md`](docs/HANDBOOK.md). The full chronological build story lives in [`docs/BUILD_NOTES.md`](docs/BUILD_NOTES.md).

## Run

Requires Python 3.11+ and either Ollama or credentials / base URLs for another supported provider.

```bash
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
ollama pull qwen2.5:7b-instruct
python main.py
```

Type `/help` in the REPL. Use `/model` to inspect or swap the active provider/model at runtime.

## Reading order

- [`docs/HANDBOOK.md`](docs/HANDBOOK.md) — stable concepts and architecture
- [`docs/BUILD_NOTES.md`](docs/BUILD_NOTES.md) — design choices, chronological
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — shipped vs. next

## License

[MIT](LICENSE)
