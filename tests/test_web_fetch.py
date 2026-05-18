import httpx

from agent import READ_ONLY_TOOLS
from repl.tool_registry import TOOL_NAMES, TOOL_SCHEMAS, make_execute_tool
from tools.web_fetch import web_fetch


class _FakeProvider:
    def __init__(self, reply):
        self._reply = reply
        self.seen_messages = None

    def chat(self, messages, num_ctx):
        self.seen_messages = messages
        return self._reply, None


def _html_response(url, headers, body):
    request = httpx.Request("GET", url, headers=headers)
    return httpx.Response(
        200,
        headers={"content-type": "text/html; charset=utf-8"},
        text=body,
        request=request,
    )


def test_web_fetch_rejects_empty_url():
    assert web_fetch("").startswith("Web fetch failed: url must not be empty.")
    assert web_fetch("   ").startswith("Web fetch failed: url must not be empty.")


def test_web_fetch_rejects_non_http_url():
    out = web_fetch("ftp://example.com/file")
    assert out.startswith("Web fetch failed: url must start with http:// or https://.")


def test_web_fetch_returns_generic_summary_without_query(monkeypatch):
    def fake_get(url, *, headers, timeout, follow_redirects):
        return _html_response(
            url,
            headers,
            "<html><body><script>junk()</script><p>Article body text.</p></body></html>",
        )

    fake_provider = _FakeProvider("The page is an article about widgets.")
    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr("providers.get_active_provider", lambda: fake_provider)

    out = web_fetch("https://example.com/article")

    assert out.startswith("Summary of https://example.com/article:")
    assert "The page is an article about widgets." in out
    user_msg = fake_provider.seen_messages[-1]["content"]
    assert "Query:" not in user_msg
    assert "Article body text." in user_msg
    assert "<script>" not in user_msg


def test_web_fetch_uses_query_when_provided(monkeypatch):
    def fake_get(url, *, headers, timeout, follow_redirects):
        return _html_response(url, headers, "<p>Relevant facts.</p>")

    fake_provider = _FakeProvider("Answer focused on the query.")
    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr("providers.get_active_provider", lambda: fake_provider)

    out = web_fetch("https://example.com/doc", query="what does X do?")

    assert "Summary of https://example.com/doc (query: what does X do?):" in out
    assert "Answer focused on the query." in out
    system_msg = fake_provider.seen_messages[0]["content"]
    user_msg = fake_provider.seen_messages[-1]["content"]
    assert "user's query" in system_msg
    assert "Query: what does X do?" in user_msg


def test_web_fetch_blank_query_falls_back_to_generic(monkeypatch):
    def fake_get(url, *, headers, timeout, follow_redirects):
        return _html_response(url, headers, "<p>Body.</p>")

    fake_provider = _FakeProvider("Generic summary.")
    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr("providers.get_active_provider", lambda: fake_provider)

    out = web_fetch("https://example.com/page", query="   ")

    assert out.startswith("Summary of https://example.com/page:")
    assert "(query:" not in out


def test_web_fetch_reports_fetch_failure(monkeypatch):
    def fake_get(url, *, headers, timeout, follow_redirects):
        raise httpx.ConnectError("boom")

    monkeypatch.setattr(httpx, "get", fake_get)

    out = web_fetch("https://example.com/missing")

    assert out.startswith("Web fetch failed: could not retrieve readable text")
    assert "https://example.com/missing" in out


def test_web_fetch_reports_summarization_failure(monkeypatch):
    def fake_get(url, *, headers, timeout, follow_redirects):
        return _html_response(url, headers, "<p>Body.</p>")

    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr("tools.web_fetch._summarize_page", lambda *a, **kw: None)

    out = web_fetch("https://example.com/page")

    assert out.startswith("Web fetch failed: summarization of")


def test_web_fetch_skips_non_text_content_types(monkeypatch):
    def fake_get(url, *, headers, timeout, follow_redirects):
        request = httpx.Request("GET", url, headers=headers)
        return httpx.Response(
            200,
            headers={"content-type": "application/octet-stream"},
            content=b"\x00\x01\x02",
            request=request,
        )

    monkeypatch.setattr(httpx, "get", fake_get)

    out = web_fetch("https://example.com/binary")

    assert out.startswith("Web fetch failed: could not retrieve readable text")


def test_registry_exposes_and_dispatches_web_fetch(monkeypatch, state):
    schema = next(t for t in TOOL_SCHEMAS if t["function"]["name"] == "web_fetch")
    assert "web_fetch" in TOOL_NAMES
    assert "web_fetch" in READ_ONLY_TOOLS
    assert schema["function"]["parameters"]["required"] == ["url"]

    monkeypatch.setattr(
        "tools.web_fetch.web_fetch",
        lambda url, query=None: f"fetch:{url}:{query}",
    )

    execute_tool = make_execute_tool(state)
    out = execute_tool(
        "web_fetch",
        {"url": "https://example.com/x", "query": "why"},
    )

    assert out == "fetch:https://example.com/x:why"
