"""Live web search via Brave Search API.

Narrow by design: one query in, a compact plain-text result summary out.
That keeps the tool easy for smaller local models to call correctly while
still grounding answers in fresh web data.
"""

import os

import httpx

from tools.spec import register

from env import load_dotenv

SEARCH_API_URL = "https://api.search.brave.com/res/v1/web/search"
DEFAULT_MAX_RESULTS = 5
MAX_RESULTS = 20
REQUEST_TIMEOUT_S = 15.0
API_KEY_ENV_VARS = (
    "BRAVE_SEARCH_API_KEY",
    "BRAVE_API_KEY",
    "SEARCH_API_KEY",  # backwards-compatible with the original stub
)


def _get_api_key() -> str | None:
    load_dotenv()
    for env_var in API_KEY_ENV_VARS:
        value = os.environ.get(env_var, "").strip()
        if value:
            return value
    return None


def _clean(text: str | None, default: str) -> str:
    if not text:
        return default
    return " ".join(str(text).split())


def _clamp_max_results(max_results: int) -> int:
    try:
        count = int(max_results)
    except (TypeError, ValueError):
        return DEFAULT_MAX_RESULTS
    return max(1, min(count, MAX_RESULTS))


def _format_web_results(query: str, data: dict, max_results: int) -> str:
    query_info = data.get("query") or {}
    original = _clean(query_info.get("original"), query)
    altered = _clean(query_info.get("altered"), "")
    more = bool(query_info.get("more_results_available"))
    results = ((data.get("web") or {}).get("results") or [])[:max_results]

    if not results:
        return ""

    lines = [f"Web results for: {original}"]
    if altered and altered != original:
        lines.append(f"Spellcheck suggestion used: {altered}")
    lines.append(f"More results available: {'yes' if more else 'no'}")
    lines.append("")

    for i, result in enumerate(results, start=1):
        title = _clean(result.get("title"), "Untitled result")
        url = _clean(result.get("url"), "No URL")
        description = _clean(result.get("description"), "No description.")
        age = _clean(result.get("age"), "")
        extra_snippets = [
            _clean(snippet, "")
            for snippet in (result.get("extra_snippets") or [])[:2]
            if _clean(snippet, "")
        ]

        lines.append(f"[{i}] {title}")
        lines.append(f"URL: {url}")
        if age:
            lines.append(f"Age: {age}")
        lines.append(f"Summary: {description}")
        for extra in extra_snippets:
            lines.append(f"Extra: {extra}")
        lines.append("")

    return "\n".join(lines).rstrip()


def _format_location_results(query: str, data: dict, max_results: int) -> str:
    query_info = data.get("query") or {}
    original = _clean(query_info.get("original"), query)
    results = ((data.get("locations") or {}).get("results") or [])[:max_results]

    if not results:
        return ""

    lines = [f"Location results for: {original}", ""]
    for i, result in enumerate(results, start=1):
        title = _clean(result.get("title"), "Untitled location")
        url = _clean(result.get("url"), "No URL")
        description = _clean(result.get("description"), "No description.")
        lines.append(f"[{i}] {title}")
        lines.append(f"URL: {url}")
        lines.append(f"Summary: {description}")
        lines.append("")
    return "\n".join(lines).rstrip()


def web_search(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> str:
    """Search the public web and return a compact, model-friendly summary."""
    query = query.strip()
    if not query:
        return "Web search failed: query must not be empty."

    api_key = _get_api_key()
    if not api_key:
        envs = ", ".join(API_KEY_ENV_VARS)
        return f"Web search failed: missing API key. Set one of: {envs}."

    count = _clamp_max_results(max_results)
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key,
    }
    # Explicit annotation: without it, mypy widens the value type to
    # `object` and httpx.get's `params` param rejects it.
    params: dict[str, str | int] = {
        "q": query,
        "count": count,
        "country": "ALL",
        "search_lang": "en",
    }

    try:
        response = httpx.get(
            SEARCH_API_URL,
            headers=headers,
            params=params,
            timeout=REQUEST_TIMEOUT_S,
        )
        response.raise_for_status()
    except httpx.TimeoutException:
        return f"Web search failed: request timed out after {REQUEST_TIMEOUT_S:.0f}s."
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        if status == 401:
            return "Web search failed: Brave API rejected the key (HTTP 401)."
        if status == 429:
            return "Web search failed: Brave API rate limit hit (HTTP 429)."
        body = e.response.text.strip()
        body_preview = _clean(body[:200], "")
        detail = f" {body_preview}" if body_preview else ""
        return f"Web search failed: HTTP {status}.{detail}".rstrip()
    except httpx.RequestError as e:
        return f"Web search failed: network error ({type(e).__name__}): {e}"

    try:
        data = response.json()
    except ValueError:
        return "Web search failed: provider returned invalid JSON."

    formatted = _format_web_results(query, data, count)
    if formatted:
        return formatted

    formatted = _format_location_results(query, data, count)
    if formatted:
        return formatted

    sections = sorted(k for k, v in data.items() if isinstance(v, dict))
    suffix = f" Response sections: {', '.join(sections)}." if sections else ""
    return f"No search results found for: {query}.{suffix}"


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


def _web_search_adapter(args: dict, _cwd: str):
    return web_search(
        args["query"],
        int(args.get("max_results", 5)),
    )


register(
    name="web_search",
    description="Search the live public web. Use this for current events, recent facts, or external information that may not be in the model's training data.",
    adapter=_web_search_adapter,
    properties={
        "query": {"type": "string", "description": "The web search query to run."},
        "max_results": {
            "type": "integer",
            "description": "How many results to return (1-20, default 5).",
        },
    },
    required=["query"],
    read_only=True,
)
