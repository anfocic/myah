"""Permission bridge — the worker/UI-thread handshake behind the full-screen
REPL's permission gate."""

import threading
import time

from repl.permission_bridge import PermissionBridge


def test_ask_blocks_until_resolved_and_returns_the_choice():
    bridge = PermissionBridge()
    result: list[str] = []

    def worker():
        result.append(bridge.ask("Allow?"))

    t = threading.Thread(target=worker)
    t.start()

    # worker is blocked in ask(); a request is now pending
    for _ in range(100):
        if bridge.pending is not None:
            break
        time.sleep(0.005)
    assert bridge.pending is not None
    assert bridge.pending.prompt == "Allow?"
    assert not result  # still blocked

    bridge.resolve("a")
    t.join(timeout=2.0)
    assert not t.is_alive()
    assert result == ["a"]
    assert bridge.pending is None  # cleared on resolve


def test_resolve_with_nothing_pending_returns_false():
    bridge = PermissionBridge()
    assert bridge.resolve("y") is False


def test_on_change_fires_when_a_request_is_posted_and_resolved():
    hits: list[int] = []
    bridge = PermissionBridge(on_change=lambda: hits.append(1))

    def worker():
        bridge.ask("Allow?")

    t = threading.Thread(target=worker)
    t.start()
    for _ in range(100):
        if bridge.pending is not None:
            break
        time.sleep(0.005)
    assert hits  # fired on post

    before = len(hits)
    bridge.resolve("n")
    t.join(timeout=2.0)
    assert len(hits) > before  # fired again once resolved


def test_default_result_is_deny():
    # A request resolved off the happy path (abort/exit) must fail safe.
    bridge = PermissionBridge()
    result: list[str] = []
    t = threading.Thread(target=lambda: result.append(bridge.ask("Allow?")))
    t.start()
    for _ in range(100):
        if bridge.pending is not None:
            break
        time.sleep(0.005)
    bridge.resolve("n")  # what the app does on Ctrl+C / exit while pending
    t.join(timeout=2.0)
    assert result == ["n"]
