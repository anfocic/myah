import httpx
from rich.console import Console

from agent import READ_ONLY_TOOLS
from display import _parse_web_results, render_web_search_results
from repl.tool_registry import TOOL_NAMES, make_execute_tool, tools
import tools.web_search as web_search_module
from tools.web_search import SEARCH_API_URL, web_search


def test_web_search_requires_api_key(monkeypatch):
    for env_var in ("BRAVE_SEARCH_API_KEY", "BRAVE_API_KEY", "SEARCH_API_KEY"):
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setattr(web_search_module, "load_dotenv", lambda: None)

    out = web_search("latest python release")

    assert out.startswith("Web search failed: missing API key.")
    assert "BRAVE_SEARCH_API_KEY" in out


def test_web_search_formats_brave_results(monkeypatch):
    monkeypatch.setenv("BRAVE_API_KEY", "brave-test-key")
    seen: dict = {}

    def fake_get(url, *, headers, params, timeout):
        seen["url"] = url
        seen["headers"] = headers
        seen["params"] = params
        seen["timeout"] = timeout
        request = httpx.Request("GET", url, params=params, headers=headers)
        return httpx.Response(
            200,
            json={
                "query": {
                    "original": "python agent harness",
                    "more_results_available": True,
                },
                "web": {
                    "results": [
                        {
                            "title": "Example Result",
                            "url": "https://example.com/agent",
                            "description": "A practical guide to agent harnesses.",
                            "age": "2 days ago",
                        },
                        {
                            "title": "Second Result",
                            "url": "https://example.com/second",
                            "description": "More background on tool calling.",
                        },
                    ]
                },
            },
            request=request,
        )

    monkeypatch.setattr(httpx, "get", fake_get)

    out = web_search("python agent harness", max_results=2)

    assert "Web results for: python agent harness" in out
    assert "More results available: yes" in out
    assert "[1] Example Result" in out
    assert "URL: https://example.com/agent" in out
    assert "Age: 2 days ago" in out
    assert "Summary: A practical guide to agent harnesses." in out
    assert seen["url"] == SEARCH_API_URL
    assert seen["headers"]["X-Subscription-Token"] == "brave-test-key"
    assert seen["params"]["q"] == "python agent harness"
    assert seen["params"]["count"] == 2
    assert seen["params"]["country"] == "ALL"
    assert seen["params"]["search_lang"] == "en"


def test_registry_exposes_and_dispatches_web_search(monkeypatch, state):
    schema = next(t for t in tools if t["function"]["name"] == "web_search")
    assert "web_search" in TOOL_NAMES
    assert "web_search" in READ_ONLY_TOOLS
    assert schema["function"]["parameters"]["required"] == ["query"]

    monkeypatch.setattr(
        "repl.tool_registry.web_search",
        lambda query, max_results=5: f"search:{query}:{max_results}",
    )

    execute_tool = make_execute_tool(state)
    out = execute_tool("web_search", {"query": "fresh facts", "max_results": 3})

    assert out == "search:fresh facts:3"


def test_parse_web_results_pairs_titles_with_urls():
    sample = (
        "Web results for: agents\n"
        "More results available: yes\n"
        "\n"
        "[1] Example Result\n"
        "URL: https://example.com/agent\n"
        "Age: 2 days ago\n"
        "Summary: something\n"
        "\n"
        "[2] Second Result\n"
        "URL: https://example.com/second\n"
        "Summary: something else\n"
    )

    assert _parse_web_results(sample) == [
        (1, "Example Result", "https://example.com/agent"),
        (2, "Second Result", "https://example.com/second"),
    ]


def test_parse_web_results_skips_missing_urls():
    sample = "[1] Broken\nURL: No URL\nSummary: x\n"
    assert _parse_web_results(sample) == []


def test_render_web_search_results_emits_osc8_hyperlinks():
    # force_terminal=True so Rich emits ANSI even under pytest capture;
    # OSC 8 uses "\x1b]8;;URL\x1b\\TEXT\x1b]8;;\x1b\\".
    buf = Console(force_terminal=True, record=True, width=120)
    render_web_search_results(
        buf,
        "[1] Example Result\nURL: https://example.com/agent\nSummary: x\n",
    )
    exported = buf.export_text(styles=True)

    assert "\x1b]8;" in exported
    assert "https://example.com/agent" in exported
    assert "Example Result" in exported


def test_web_search_loads_api_key_from_dotenv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("BRAVE_API_KEY=dotenv-brave-key\n")
    for env_var in ("BRAVE_SEARCH_API_KEY", "BRAVE_API_KEY", "SEARCH_API_KEY"):
        monkeypatch.delenv(env_var, raising=False)

    seen: dict = {}

    def fake_get(url, *, headers, params, timeout):
        seen["headers"] = headers
        request = httpx.Request("GET", url, params=params, headers=headers)
        return httpx.Response(
            200,
            json={"query": {"original": "fresh facts"}, "web": {"results": []}},
            request=request,
        )

    monkeypatch.setattr(httpx, "get", fake_get)

    web_search("fresh facts")

    assert seen["headers"]["X-Subscription-Token"] == "dotenv-brave-key"
