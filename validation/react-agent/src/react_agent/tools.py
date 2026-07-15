"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.

VALIDATION DEVIATION (see ../../README.md): the upstream `search` tool below
called Tavily, which needs a paid/signup-gated API key. For this ResilientForge
real-world validation exercise, it's swapped for `ddgs` (DuckDuckGo) — free,
keyless, still a real, live, unpredictable web search, just not Tavily
specifically. Nothing else in this repo is modified. Real errors (rate
limiting, network failures, empty results) are deliberately left to propagate
unmodified, same as the original — this is the whole point of the exercise.
"""

from typing import Any, Callable, List, Optional, cast

import anyio
from ddgs import DDGS
from langgraph.runtime import get_runtime

from react_agent.context import Context


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


TOOLS: List[Callable[..., Any]] = [search]
