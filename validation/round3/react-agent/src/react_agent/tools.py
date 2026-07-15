"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.

VALIDATION DEVIATION, ROUND 2 (see ../../README.md): same free-alternative
search swap as round 1 (Tavily -> ddgs), reused verbatim, plus a NEW second
tool, `extract_url_content` -- a real, deliberately undefensive (not
deliberately broken) URL-fetch-and-clean-text tool: real HTTP via httpx,
real HTML-to-text via BeautifulSoup4, strict UTF-8 decoding (no silent
fallback). Chosen to widen the real failure surface beyond one search API's
failure modes -- confirmed, before ever wiring this into the agent, that
real pages organically produce: 403s (Wikipedia's own bot detection, not
anything rigged), DNS failures (a bad domain), UnicodeDecodeError (a real
PDF's binary bytes fail strict UTF-8 decoding), and redirect loops. A
realistic User-Agent header is set -- not to dodge failures, but because a
tool with no User-Agent at all gets blocked by a much larger, more boring
fraction of the real web (this was confirmed empirically too), which would
have just recreated round 1's problem of one failure type drowning out
everything else.
"""

from typing import Any, Callable, List, Optional, cast

import anyio
import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS
from langgraph.runtime import get_runtime

from react_agent.context import Context

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


def _ddg_search(query: str, max_results: int) -> dict[str, Any]:
    with DDGS() as ddgs:
        hits = list(ddgs.text(query, max_results=max_results))
    return {
        "query": query,
        "results": [
            {"title": h.get("title"), "url": h.get("href"), "content": h.get("body")}
            for h in hits
        ],
    }


async def search(query: str) -> Optional[dict[str, Any]]:
    """Search for general web results.

    This function performs a search using DuckDuckGo (free, keyless — see the
    VALIDATION DEVIATION note above), which is particularly useful for
    answering questions about current events.
    """
    runtime = get_runtime(Context)
    return cast(
        dict[str, Any],
        await anyio.to_thread.run_sync(
            _ddg_search, query, runtime.context.max_search_results
        ),
    )


def _fetch_and_extract(url: str) -> dict[str, Any]:
    with httpx.Client(follow_redirects=True, timeout=10.0, headers={"User-Agent": _USER_AGENT}) as client:
        response = client.get(url)
        response.raise_for_status()
    text = response.content.decode("utf-8")
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    cleaned = soup.get_text(separator=" ", strip=True)
    return {"url": url, "content": cleaned}


async def extract_url_content(url: str) -> dict[str, Any]:
    """Fetch a real web page and return its cleaned, human-readable text
    content (scripts/styles stripped, HTML tags removed). Use this when the
    user gives you a specific URL, or after `search` finds one worth
    reading in full. Only works on real, fetchable, HTML/text pages —
    binary content (PDFs, images) or unreachable/erroring URLs will fail,
    same as it would for a real person trying to read them.
    """
    return cast(dict[str, Any], await anyio.to_thread.run_sync(_fetch_and_extract, url))


TOOLS: List[Callable[..., Any]] = [search, extract_url_content]
