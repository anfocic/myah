"""Shared fixtures. Keeps tests from silently talking to a real ollama
daemon (which would be flaky in CI) and gives us a clean State factory."""
from collections import deque

import pytest


@pytest.fixture
def state():
    """Minimal REPL state with the shape main.py uses, no persistent files."""
    return {
        "history": [],
        "ctx_used": 0,
        "plan_mode": False,
        "debug": False,
        "snapshots": deque(maxlen=20),
        "todos": [],
    }
