"""Fetch any URL and return an LLM summary of the page.

A peer to `web_search`: same page-fetch + summarize pipeline, but the model
hands in a URL directly instead of routing through a Brave search round-trip.
Useful for articles the user pasted, docs pages, and any other URL the model
already knows it wants to read."""

from tools._web_common import _fetch_page_text, _summarize_page
from tools.spec import register


def web_fetch(url: str, query: str | None = None) -> str:
    """Fetch `url` and return a compact summary of its contents.

    With `query`, the summary is biased toward facts that answer it. Without
    one, the summary describes what the page is about and its main substance.
    Returns an error string on fetch failure so the model can recover instead
    of crashing the tool-call loop."""
    url = (url or "").strip()
    if not url:
        return "Web fetch failed: url must not be empty."
    if not (url.startswith("http://") or url.startswith("https://")):
        return "Web fetch failed: url must start with http:// or https://."

    page_text = _fetch_page_text(url)
    if not page_text:
        return f"Web fetch failed: could not retrieve readable text from {url}."

    query = (query or "").strip() or None
    summary = _summarize_page(url, page_text, query)
    if not summary:
        return f"Web fetch failed: summarization of {url} returned no content."

    header = f"Summary of {url}"
    if query:
        header += f" (query: {query})"
    return f"{header}:\n{summary}"


def _web_fetch_adapter(args: dict, _cwd: str):
    return web_fetch(args["url"], args.get("query"))


register(
    name="web_fetch",
    description=(
        "Fetch a single URL and return an LLM-written summary of its page. "
        "Use this when you already know the URL you want to read — articles "
        "the user pasted, documentation pages, GitHub files — instead of "
        "routing through a search. Pass `query` to bias the summary toward a "
        "specific question; omit it for a generic summary of the page."
    ),
    adapter=_web_fetch_adapter,
    properties={
        "url": {
            "type": "string",
            "description": "The full http(s) URL to fetch and summarize.",
        },
        "query": {
            "type": "string",
            "description": (
                "Optional question or topic the summary should focus on. "
                "Omit for a general summary of the page."
            ),
        },
    },
    required=["url"],
    read_only=True,
)
