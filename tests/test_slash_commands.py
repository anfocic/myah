"""handle_slash dispatch: uniform (state, arg='') signature; unknown
commands reported without crash; commands with no-arg ignore stray args."""
from repl.commands import handle_slash


def test_unknown_command_returns_true_without_crash(state):
    assert handle_slash("/nope", state) is True


def test_non_slash_input_returns_false(state):
    assert handle_slash("plain text", state) is False


def test_known_command_returns_true(state):
    # /help is cheap — prints to console but doesn't mutate state
    assert handle_slash("/help", state) is True


def test_command_with_arg_is_parsed(state):
    # /rewind accepts an integer arg; with no snapshots it's a no-op
    assert handle_slash("/rewind 3", state) is True


def test_no_arg_command_with_extra_ignores_arg(state):
    # /help doesn't care about args — must still succeed
    assert handle_slash("/help ignored-text", state) is True
