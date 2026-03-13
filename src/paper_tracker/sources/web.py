"""Search diverse web platforms for non-academic perspectives.

Targets: Reddit, HackerNews, Zhihu, Xiaohongshu, general forums.
Used by the brainstorm novelty challenge pipeline to inject practitioner
viewpoints, industry pain points, and cross-domain analogies that academic
papers miss.

Search engine priority (cascading fallback):
  1. ddgs (multi-backend: bing → google → mojeek → yandex → yahoo)
  2. Serper.dev (if SERPER_API_KEY set — Google SERP quality)
  3. Tavily (if TAVILY_API_KEY set — AI-optimized search)
  4. Brave HTML scraping (last resort, heavy rate-limiting)
  5. HackerNews Algolia API (always available, no rate limit)
"""

from __future__ import annotations

import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

log = logging.getLogger(__name__)

_TIMEOUT = 15  # seconds per request

# ---------------------------------------------------------------------------
# Engine 1: ddgs library (multi-backend, no API key)
# ---------------------------------------------------------------------------

_DDGS_BACKENDS = ["auto", "google", "mojeek", "yandex", "yahoo", "brave"]


def _search_ddgs(
    query: str,
    *,
    max_results: int = 10,
) -> list[dict]:
    """Search via ddgs library with backend rotation fallback."""
    try:
        from ddgs import DDGS
    except ImportError:
        return []

    for backend in _DDGS_BACKENDS:
        try:
            raw = DDGS().text(query, max_results=max_results, backend=backend)
            if raw:
                results = []
                for r in raw:
                    results.append({
                        "source": f"ddgs:{backend}",
                        "title": r.get("title", ""),
                        "snippet": r.get("body", ""),
                        "url": r.get("href", ""),
                    })
                if results:
                    log.debug("ddgs/%s returned %d results for: %s",
                              backend, len(results), query[:50])
                    return results
        except Exception as exc:
            log.debug("ddgs/%s failed: %s", backend, str(exc)[:80])
            continue

    return []


# ---------------------------------------------------------------------------
# Engine 2: Serper.dev API (Google SERP quality, needs SERPER_API_KEY)
# ---------------------------------------------------------------------------

def _search_serper(
    query: str,
    *,
    max_results: int = 10,
) -> list[dict]:
    """Search via Serper.dev Google SERP API."""
    api_key = os.environ.get("SERPER_API_KEY", "")
    if not api_key:
        return []

    try:
        resp = httpx.post(
            "https://google.serper.dev/search",
            headers={
                "X-API-KEY": api_key,
                "Content-Type": "application/json",
            },
            json={"q": query, "num": min(max_results, 20)},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        log.warning("Serper search failed: %s", exc)
        return []

    results: list[dict] = []
    for item in data.get("organic", []):
        results.append({
            "source": "serper",
            "title": item.get("title", ""),
            "snippet": item.get("snippet", ""),
            "url": item.get("link", ""),
        })
    return results[:max_results]


# ---------------------------------------------------------------------------
# Engine 3: Tavily API (AI-optimized, needs TAVILY_API_KEY)
# ---------------------------------------------------------------------------

def _search_tavily(
    query: str,
    *,
    max_results: int = 10,
    include_domains: list[str] | None = None,
) -> list[dict]:
    """Search via Tavily API."""
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        return []

    body: dict = {
        "query": query,
        "max_results": min(max_results, 20),
    }
    if include_domains:
        body["include_domains"] = include_domains

    try:
        resp = httpx.post(
            "https://api.tavily.com/search",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        log.warning("Tavily search failed: %s", exc)
        return []

    results: list[dict] = []
    for item in data.get("results", []):
        results.append({
            "source": "tavily",
            "title": item.get("title", ""),
            "snippet": item.get("content", ""),
            "url": item.get("url", ""),
        })
    return results[:max_results]


# ---------------------------------------------------------------------------
# Engine 4: Brave HTML scraping (last resort)
# ---------------------------------------------------------------------------

_brave_last_request: float = 0.0
_BRAVE_MIN_INTERVAL = 10.0  # seconds between Brave requests


def _search_brave(
    query: str,
    *,
    max_results: int = 10,
) -> list[dict]:
    """Search via Brave Search HTML scraping (heavy rate-limiting)."""
    global _brave_last_request
    elapsed = time.monotonic() - _brave_last_request
    if elapsed < _BRAVE_MIN_INTERVAL:
        time.sleep(_BRAVE_MIN_INTERVAL - elapsed)

    try:
        _brave_last_request = time.monotonic()
        resp = httpx.get(
            "https://search.brave.com/search",
            params={"q": query, "source": "web"},
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml",
            },
            timeout=_TIMEOUT,
            follow_redirects=True,
        )
        if resp.status_code == 429:
            log.info("Brave 429 for: %s", query[:50])
            return []
        resp.raise_for_status()
    except Exception as exc:
        log.debug("Brave search failed: %s", exc)
        return []

    # Parse HTML for results (best-effort regex)
    import html as html_mod
    results: list[dict] = []
    seen_urls: set[str] = set()
    link_pattern = re.compile(
        r'<a[^>]+href="(https?://(?!search\.brave\.com)[^"]+)"[^>]*>(.*?)</a>',
        re.DOTALL,
    )
    for url, raw_title in link_pattern.findall(resp.text):
        if any(skip in url for skip in (
            "brave.com", "javascript:", "google.com/search", "#",
            ".css", ".js", ".png", ".jpg", "favicon",
        )):
            continue
        title = re.sub(r"<[^>]+>", " ", raw_title).strip()
        title = html_mod.unescape(title)
        title = re.sub(r"\s+", " ", title).strip()
        if len(title) < 10 or url in seen_urls:
            continue
        seen_urls.add(url)

        snippet = ""
        url_idx = resp.text.find(url)
        if url_idx > 0:
            nearby = resp.text[url_idx:url_idx + 2000]
            desc_match = re.search(
                r'class="[^"]*(?:snippet-description|description|body|text|content)[^"]*"[^>]*>(.*?)</(?:div|span|p)',
                nearby, re.DOTALL,
            )
            if desc_match:
                snippet = re.sub(r"<[^>]+>", " ", desc_match.group(1)).strip()
                snippet = html_mod.unescape(snippet)

        results.append({
            "source": "brave",
            "title": title[:200],
            "snippet": re.sub(r"\s+", " ", snippet)[:400],
            "url": url,
        })

    return results[:max_results]


# ---------------------------------------------------------------------------
# HackerNews (Algolia API — free, no auth, no rate limit)
# ---------------------------------------------------------------------------

def search_hackernews(
    query: str,
    *,
    max_results: int = 10,
) -> list[dict]:
    """Search HackerNews via Algolia API. Returns list of snippet dicts."""
    url = "https://hn.algolia.com/api/v1/search"
    params = {
        "query": query,
        "tags": "story",
        "hitsPerPage": min(max_results, 20),
    }
    try:
        resp = httpx.get(url, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        log.warning("HackerNews search failed: %s", exc)
        return []

    results: list[dict] = []
    for hit in data.get("hits", []):
        title = hit.get("title", "")
        story_text = hit.get("story_text") or ""
        if story_text:
            story_text = re.sub(r"<[^>]+>", " ", story_text)[:400]
        points = hit.get("points", 0)
        num_comments = hit.get("num_comments", 0)
        if title:
            results.append({
                "source": "hackernews",
                "title": title,
                "snippet": story_text or "(external link)",
                "score": points,
                "comments": num_comments,
                "url": f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}",
            })
    return results[:max_results]


# ---------------------------------------------------------------------------
# Unified search with cascading fallback
# ---------------------------------------------------------------------------

def search_web(
    query: str,
    *,
    site: str | None = None,
    max_results: int = 10,
) -> list[dict]:
    """Search the web with cascading engine fallback.

    Tries engines in order: ddgs → serper → tavily → brave.
    If site is specified, prepends site: operator to query.
    """
    full_query = f"site:{site} {query}" if site else query

    # 1. ddgs (multi-backend, free)
    results = _search_ddgs(full_query, max_results=max_results)
    if results:
        if site:
            src = site.replace(".com", "").replace("www.", "")
            for r in results:
                r["source"] = src
        return results

    # 2. Serper.dev (if API key available)
    results = _search_serper(full_query, max_results=max_results)
    if results:
        if site:
            src = site.replace(".com", "").replace("www.", "")
            for r in results:
                r["source"] = src
        return results

    # 3. Tavily (if API key available, supports domain filtering)
    if site:
        results = _search_tavily(query, max_results=max_results,
                                 include_domains=[site])
    else:
        results = _search_tavily(full_query, max_results=max_results)
    if results:
        if site:
            src = site.replace(".com", "").replace("www.", "")
            for r in results:
                r["source"] = src
        return results

    # 4. Brave HTML (last resort)
    results = _search_brave(full_query, max_results=max_results)
    if results:
        if site:
            src = site.replace(".com", "").replace("www.", "")
            for r in results:
                r["source"] = src
        return results

    log.warning("All search engines failed for: %s", full_query[:80])
    return []


def search_reddit(
    query: str,
    *,
    max_results: int = 10,
) -> list[dict]:
    """Search Reddit via web search engines. Returns list of snippet dicts."""
    results = search_web(query, site="reddit.com", max_results=max_results)
    for r in results:
        r["source"] = "reddit"
    return results


# Backward-compatible alias
search_duckduckgo = search_web


# ---------------------------------------------------------------------------
# Unified multi-platform search
# ---------------------------------------------------------------------------

def _build_search_queries(
    idea_title: str,
    idea_problem: str,
    idea_method: str,
) -> tuple[str, str]:
    """Build English + Chinese search queries from idea fields.

    Returns (english_query, chinese_query).
    """
    all_text = f"{idea_title} {idea_problem}"
    stop = {
        "the", "and", "for", "with", "from", "that", "this", "are", "was",
        "using", "based", "novel", "new", "approach", "method", "via", "our",
        "we", "propose", "proposed", "paper", "show", "can", "how", "what",
        "which", "towards", "toward", "into",
    }
    words = re.sub(r"[^\w\s-]", " ", all_text.lower()).split()
    key_terms = [w for w in words if len(w) >= 4 and w not in stop][:8]
    en_query = " ".join(key_terms)
    cn_query = f"{en_query} 实现 应用 痛点 经验"
    return en_query, cn_query


def gather_perspectives(
    idea_title: str,
    idea_problem: str,
    idea_method: str,
    *,
    round_num: int = 1,
    max_per_source: int = 5,
) -> str:
    """Search multiple web platforms for diverse perspectives on an idea.

    Round 1: focused search (Reddit + HN)
    Round 2: broader search (+ method-specific + challenges)
    Round 3: even broader (+ Chinese platforms + industry)

    Returns formatted text of perspectives, ready for injection into prompts.
    """
    en_query, cn_query = _build_search_queries(idea_title, idea_problem, idea_method)
    if not en_query.strip():
        return ""

    # Method-specific query for deeper technical discussions
    method_words = re.sub(r"[^\w\s-]", " ", idea_method.lower()).split()
    method_terms = [w for w in method_words if len(w) >= 4 and w not in {
        "the", "and", "for", "with", "from", "that", "this", "using", "based",
    }][:5]
    method_query = " ".join(method_terms) if method_terms else en_query

    # Define search tasks based on round.
    # HN (Algolia API) is always free and fast.
    # Web search uses cascading fallback (ddgs → serper → tavily → brave).
    tasks: list[tuple[str, dict]] = [
        ("hackernews", {"query": en_query, "max_results": max_per_source}),
        ("reddit", {"query": en_query, "max_results": max_per_source}),
    ]

    if round_num >= 2:
        tasks.extend([
            ("hackernews_method", {"query": method_query, "max_results": max_per_source}),
            ("web_general", {"query": f"{en_query} practical challenges limitations real-world", "max_results": max_per_source * 2}),
            ("web_reddit_method", {"query": method_query, "site": "reddit.com", "max_results": max_per_source}),
        ])

    if round_num >= 3:
        tasks.extend([
            ("hackernews_practical", {"query": f"{en_query} implementation production", "max_results": max_per_source}),
            ("web_chinese", {"query": cn_query, "max_results": max_per_source * 2}),
            ("web_industry", {"query": f"{en_query} deep learning implementation deployment", "max_results": max_per_source}),
            ("web_zhihu", {"query": cn_query, "site": "zhihu.com", "max_results": max_per_source}),
        ])

    all_results: list[dict] = []

    def _run_task(name: str, kwargs: dict) -> list[dict]:
        if name.startswith("hackernews"):
            return search_hackernews(**kwargs)
        elif name.startswith("reddit"):
            return search_reddit(**kwargs)
        elif name.startswith("web_"):
            return search_web(**kwargs)
        return []

    # Run HN tasks in parallel (different API, no interference)
    hn_tasks = [(n, k) for n, k in tasks if n.startswith("hackernews")]
    web_tasks = [(n, k) for n, k in tasks if not n.startswith("hackernews")]

    with ThreadPoolExecutor(max_workers=max(len(hn_tasks), 1)) as executor:
        futures = {
            executor.submit(_run_task, name, kwargs): name
            for name, kwargs in hn_tasks
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                results = future.result()
                if results:
                    all_results.extend(results)
                    log.info("Web search %s: %d results", name, len(results))
            except Exception as exc:
                log.warning("Web search %s failed: %s", name, exc)

    # Run web tasks — ddgs handles rate limiting internally
    for name, kwargs in web_tasks:
        try:
            results = _run_task(name, kwargs)
            if results:
                all_results.extend(results)
                log.info("Web search %s: %d results", name, len(results))
        except Exception as exc:
            log.warning("Web search %s failed: %s", name, exc)

    if not all_results:
        return ""

    # Format as text
    lines: list[str] = []
    seen_titles: set[str] = set()

    for r in all_results:
        title = r.get("title", "")
        title_key = title.lower().strip()[:60]
        if title_key in seen_titles:
            continue
        seen_titles.add(title_key)

        source = r.get("source", "web")
        snippet = r.get("snippet", "")
        extra = ""
        if r.get("subreddit"):
            extra = f" ({r['subreddit']})"
        elif r.get("score"):
            extra = f" (score: {r['score']})"

        lines.append(f"[{source}{extra}] {title}")
        if snippet and snippet != "(link post)" and snippet != "(external link)":
            lines.append(f"  → {snippet[:300]}")

    return "\n".join(lines) if lines else ""
