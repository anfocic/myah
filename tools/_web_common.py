"""Shared helpers for tools that fetch and summarize web pages.

Extracted from `tools.web_search` so `tools.web_fetch` can reuse the same
HTML-stripping and LLM-summarization path without forking the code."""

import html
import re

import httpx

PAGE_FETCH_TIMEOUT_S = 12.0
# Cap the page text we hand the summarizer: enough to capture the substance
# of an article without blowing the summarization call's context budget.
MAX_PAGE_CHARS = 6000

_SCRIPT_STYLE_RE = re.compile(r"<(script|style|noscript)\b.*?</\1>", re.I | re.S)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _fetch_page_text(url: str) -> str | None:
    """Fetch `url` and return its visible text, or None on any failure.

    Regex HTML stripping rather than a parser dependency: the harness keeps
    its dependency surface small, and the summarizer model tolerates the
    rough edges (stray whitespace, dropped structure) fine."""
    try:
        response = httpx.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; MiaBot/1.0)"},
            timeout=PAGE_FETCH_TIMEOUT_S,
            follow_redirects=True,
        )
        response.raise_for_status()
    except httpx.HTTPError:
        return None

    content_type = response.headers.get("content-type", "").lower()
    if "html" not in content_type and "text" not in content_type:
        return None

    body = _SCRIPT_STYLE_RE.sub(" ", response.text)
    body = _TAG_RE.sub(" ", body)
    body = html.unescape(body)
    body = _WS_RE.sub(" ", body).strip()
    return body[:MAX_PAGE_CHARS] or None


def _summarize_page(url: str, page_text: str, query: str | None = None) -> str | None:
    """Ask the active provider for a short summary of a page.

    With `query`, bias the summary toward facts that answer it. Without one,
    give a generic summary of the page's substance. Fails closed (returns
    None) on any provider error so a flaky summarization call never takes
    down the caller's primary result."""
    try:
        from config import NUM_CTX
        from providers import get_active_provider

        provider = get_active_provider()
        if query:
            system = (
                "You summarize a single web page for an agent. Reply with "
                "2-4 sentences capturing the facts most relevant to the "
                "user's query. Plain text only, no preamble, no markdown."
            )
            user = (
                f"Query: {query}\nPage URL: {url}\n\n"
                f"Page content:\n{page_text}"
            )
        else:
            system = (
                "You summarize a single web page for an agent. Reply with "
                "3-5 sentences capturing what the page is about and its "
                "main substance. Plain text only, no preamble, no markdown."
            )
            user = f"Page URL: {url}\n\nPage content:\n{page_text}"
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        content, _usage = provider.chat(messages, NUM_CTX)
    except Exception:
        return None

    content = content.strip()
    return content or None
