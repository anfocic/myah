"""CLI for the eval runner.

Usage:
    python -m scripts.run_evals                      # whole suite
    python -m scripts.run_evals --task find_string   # one task
    python -m scripts.run_evals --task find_string --task edit_rename
    python -m scripts.run_evals --provider anthropic --model claude-sonnet-4-6
    python -m scripts.run_evals --list

Exit code is 0 iff every task passed, so this is CI-gateable.
"""
from __future__ import annotations

import argparse
import sys

from evals.runner import list_tasks, run_suite


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Mia's eval suite.")
    parser.add_argument(
        "--task",
        action="append",
        default=None,
        help="task id to run (repeatable); default: all",
    )
    parser.add_argument("--provider", default=None, help="override provider name (e.g. ollama, anthropic)")
    parser.add_argument("--model", default=None, help="override model id")
    parser.add_argument("--list", action="store_true", help="list task ids and exit")
    args = parser.parse_args()

    if args.list:
        for tid in list_tasks():
            print(tid)
        return 0

    results = run_suite(
        task_ids=args.task,
        cli_provider=args.provider,
        cli_model=args.model,
    )
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
