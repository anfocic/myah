"""Subagent tool tests — `spawn_subagent` runs a nested run_agent with
fresh history, isolated from the parent's turn.

Uses the same FakeProvider pattern as test_integration: a scripted list
of turns; each call to `stream_chat` pops the next one. The subagent
consumes turns just like the parent does, so the test scripts a
parent→subagent→parent flow as three consecutive turns.

Coverage:
- Subagent runs with empty history (doesn't see parent's prior turns).
- Subagent result is wrapped in <subagent_result> markers.
- Depth limit: nested spawn_subagent calls are refused.
- The spawn_subagent schema is filtered out of the child's tool list."""
import pytest

from agent import run_agent
from providers import Usage, set_active_provider
from providers.base import StreamChunk, ToolCall
from tools import subagent


@pytest.fixture
def reset_depth():
    """Reset the module-level depth counter around each test. Normally
    try/finally in spawn_subagent keeps this balanced, but a test that
    triggers an exception mid-call could leave _depth > 0 and poison
    subsequent tests."""
    subagent._depth = 0
    yield
    subagent._depth = 0


class ScriptedProvider:
    """Yields a pre-written sequence of turns. Each `stream_chat` call
    pops one turn and emits its chunks + optional tool_calls + a
    final done-marker with usage."""
    name = "fake"
    model = "fake-v1"

    def __init__(self, script):
        self._script = list(script)
        self.seen_messages: list[list] = []
        self.seen_tools: list[list] = []

    def stream_chat(self, messages, tools, num_ctx):
        # Snapshot what the provider was called with so tests can assert
        # on isolation (fresh history in the subagent call, filtered tools,
        # etc.) without needing to patch internals.
        self.seen_messages.append(list(messages))
        self.seen_tools.append(list(tools))

        if not self._script:
            raise AssertionError("ScriptedProvider ran out of turns")
        turn = self._script.pop(0)
        for chunk in turn.get("chunks", []):
            yield StreamChunk(content_delta=chunk)
        if turn.get("tool_calls"):
            yield StreamChunk(tool_calls=turn["tool_calls"])
        yield StreamChunk(
            done=True,
            usage=Usage(prompt_tokens=100, completion_tokens=20),
        )

    def chat(self, messages, num_ctx):
        raise NotImplementedError


@pytest.fixture
def install_provider():
    from providers import get_active_provider
    original = get_active_provider()

    def _install(script):
        p = ScriptedProvider(script)
        set_active_provider(p)
        return p

    yield _install
    set_active_provider(original)


def _spawn_tool_schema():
    return {
        "type": "function",
        "function": {
            "name": "spawn_subagent",
            "description": "spawn",
            "parameters": {
                "type": "object",
                "properties": {"task": {"type": "string"}},
                "required": ["task"],
            },
        },
    }


def test_subagent_runs_with_isolated_history(install_provider, reset_depth):
    """Parent invokes spawn_subagent; child runs with history=[] and sees
    only its task as the user message. Parent's history is not visible
    inside the subagent's messages array."""
    parent_history = [
        {"role": "user", "content": "earlier parent question"},
        {"role": "assistant", "content": "earlier parent answer"},
    ]

    provider = install_provider([
        # Parent turn 1: call spawn_subagent
        {
            "chunks": [],
            "tool_calls": [ToolCall(
                name="spawn_subagent",
                arguments={"task": "count the files"},
            )],
        },
        # Subagent turn 1: final answer, no tools
        {"chunks": ["There are three files."]},
        # Parent turn 2: final answer referencing subagent output
        {"chunks": ["The subagent says three."]},
    ])

    tool_list = [_spawn_tool_schema()]

    def allow(name, args):
        return True

    def execute_tool(name, args):
        # Dispatch spawn_subagent identically to the registry — no other
        # tools are in play here.
        if name == "spawn_subagent":
            return subagent.spawn_subagent(
                task=args["task"],
                tools=tool_list,
                execute_tool=execute_tool,
                permission_check=allow,
                console=None,
            )
        return "no other tools"

    response, history, _ctx, _stats = run_agent(
        user_input="delegate it",
        tools=tool_list,
        execute_tool=execute_tool,
        history=list(parent_history),
        permission_check=allow,
    )

    # Three stream_chat calls total: parent-1, subagent-1, parent-2.
    assert len(provider.seen_messages) == 3

    # Subagent call (index 1): messages are [system, user_task]. The
    # parent's history must NOT appear.
    subagent_messages = provider.seen_messages[1]
    assert len(subagent_messages) == 2
    assert subagent_messages[0]["role"] == "system"
    assert subagent_messages[1] == {"role": "user", "content": "count the files"}
    # Explicit: parent history strings don't leak in.
    blob = " ".join(m.get("content", "") for m in subagent_messages)
    assert "earlier parent question" not in blob
    assert "earlier parent answer" not in blob

    # Subagent system prompt uses the subagent persona (§43).
    assert "subagent" in subagent_messages[0]["content"].lower()

    # Parent's final answer reflects the subagent tool result.
    assert response == "The subagent says three."


def test_subagent_result_is_wrapped(install_provider, reset_depth):
    """The parent sees the subagent's final content wrapped in
    <subagent_result>...</subagent_result> markers — same convention as
    the §41 prompt-injection annotation."""
    install_provider([
        {"chunks": []},  # parent turn 1: (we'll call directly, skip run_agent)
    ])
    # Direct call, simpler than scripting through run_agent.
    # Replace the provider script with just the subagent's turn.
    install_provider([
        {"chunks": ["The answer is 42."]},
    ])

    def allow(name, args):
        return True

    def execute_tool(name, args):
        return ""

    result = subagent.spawn_subagent(
        task="compute the answer",
        tools=[_spawn_tool_schema()],
        execute_tool=execute_tool,
        permission_check=allow,
        console=None,
    )

    assert result.startswith("<subagent_result>")
    assert result.endswith("</subagent_result>")
    assert "The answer is 42." in result


def test_depth_limit_blocks_nested_spawns(reset_depth):
    """Even if somehow a nested call reaches spawn_subagent with depth=1,
    the module-level counter short-circuits it without running a second
    nested run_agent. Belt-and-suspenders over the schema-filter layer."""
    # Simulate parent already inside one subagent call.
    subagent._depth = subagent._MAX_DEPTH

    def allow(name, args):
        return True

    def execute_tool(name, args):
        raise AssertionError("execute_tool should not be called — depth gate fired")

    result = subagent.spawn_subagent(
        task="try to spawn nested",
        tools=[_spawn_tool_schema()],
        execute_tool=execute_tool,
        permission_check=allow,
        console=None,
    )

    assert "<subagent_error>" in result
    assert "depth" in result.lower()


def test_child_tool_list_excludes_spawn_subagent(install_provider, reset_depth):
    """The child's tool schema list must not contain spawn_subagent —
    the parent model might still emit a call to it if the schema
    appeared, and we don't want to rely solely on the depth counter
    to catch that at runtime. Save the tokens + avoid the roundtrip."""
    provider = install_provider([
        {"chunks": ["done"]},  # subagent's only turn
    ])

    tool_list = [
        _spawn_tool_schema(),
        {
            "type": "function",
            "function": {
                "name": "glob",
                "description": "find files",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
    ]

    def allow(name, args):
        return True

    def execute_tool(name, args):
        return ""

    subagent.spawn_subagent(
        task="find things",
        tools=tool_list,
        execute_tool=execute_tool,
        permission_check=allow,
        console=None,
    )

    # The subagent's stream_chat was called with tools that exclude spawn_subagent.
    child_tools = provider.seen_tools[0]
    names = {t["function"]["name"] for t in child_tools}
    assert "spawn_subagent" not in names
    assert "glob" in names  # other tools pass through unchanged


def test_depth_counter_resets_after_successful_call(install_provider, reset_depth):
    """After a subagent returns normally, _depth is back to 0 so a later
    subagent call (from a future turn) runs normally."""
    install_provider([
        {"chunks": ["ok"]},
    ])

    def allow(name, args):
        return True

    def execute_tool(name, args):
        return ""

    assert subagent._depth == 0
    subagent.spawn_subagent(
        task="t1",
        tools=[_spawn_tool_schema()],
        execute_tool=execute_tool,
        permission_check=allow,
        console=None,
    )
    assert subagent._depth == 0
