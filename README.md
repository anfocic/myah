# Mia

A hand-rolled agent harness, built to learn how agent harnesses work.

Python REPL that runs a local LLM (Ollama) or an OpenAI-compatible endpoint, lets it call tools, and manages context. ~2000 lines across ~30 files. Not a product.

Each design choice is written up in [`docs/CONCEPTS.md`](docs/CONCEPTS.md) — read top-to-bottom to trace the build.

## Run

Requires Python 3.11+ and Ollama (or an OpenAI-compatible endpoint).

```bash
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
ollama pull qwen2.5:7b-instruct
python main.py
```

Type `/help` in the REPL.

## Reading order

- [`docs/CONCEPTS.md`](docs/CONCEPTS.md) — design choices, chronological
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — shipped vs. next

## License

[MIT](LICENSE)
