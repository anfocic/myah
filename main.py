# main.py
"""Myah's REPL entry point.

Thin by design: parse CLI flags, build the REPL state, and hand off to the
full-screen REPL Application in `repl/app.py`. Everything else — the layout,
the turn loop, the permission gate, session restore/save — lives there."""
import argparse

from repl.app import ReplApp
from repl.state import State, new_state


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="myah", description="Myah — a hand-rolled agent harness.")
    p.add_argument(
        "--resume",
        action="store_true",
        help="load the prior session from ~/.mia_session.json (default: fresh start).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    state: State = new_state()
    ReplApp(state, resume=args.resume).run()


if __name__ == "__main__":
    main()
