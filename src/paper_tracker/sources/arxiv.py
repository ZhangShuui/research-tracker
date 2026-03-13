"""Search arXiv API and parse Atom XML results."""

from __future__ import annotations

import logging
import random
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

import httpx

log = logging.getLogger(__name__)

_ARXIV_API = "https://export.arxiv.org/api/query"
_NS = {"atom": "http://www.w3.org/2005/Atom"}
_RATE_LIMIT_SECS = 3


def _build_query(keywords: list[str], categories: list[str]) -> str:
    """Build an arXiv search query string with OR-combined keywords and category filter."""
    kw_part = " OR ".join(f'all:"{kw}"' for kw in keywords)
    cat_part = " OR ".join(f"cat:{c}" for c in categories)
    return f"({kw_part}) AND ({cat_part})"


def _parse_entries(root: ET.Element, cutoff: datetime, date_to: datetime | None = None) -> list[dict]:
    """Parse arXiv Atom entries, filtering by date range."""
    papers: list[dict] = []

    for entry in root.findall("atom:entry", _NS):
        paper_id_url = entry.findtext("atom:id", "", _NS).strip()
        # Extract arXiv ID from URL like http://arxiv.org/abs/2401.12345v1
        arxiv_id = paper_id_url.rsplit("/", 1)[-1]
        # Remove version suffix for dedup
        if "v" in arxiv_id:
            arxiv_id = arxiv_id.rsplit("v", 1)[0]

        published_str = entry.findtext("atom:published", "", _NS).strip()
        try:
            published = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
        except ValueError:
            continue

        if published < cutoff:
            continue
        if date_to and published > date_to:
            continue

        title = " ".join(entry.findtext("atom:title", "", _NS).split())
        authors = ", ".join(
            a.findtext("atom:name", "", _NS)
            for a in entry.findall("atom:author", _NS)
        )
        abstract = " ".join(entry.findtext("atom:summary", "", _NS).split())

        papers.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "published": published_str,
            "summary": "",       # filled by summarizer
            "key_insight": "",   # filled by summarizer
            "method": "",        # filled by summarizer
            "contribution": "",  # filled by summarizer
            "math_concepts": [], # filled by summarizer
            "venue": "",         # filled by summarizer
            "cited_works": [],   # filled by summarizer
        })

    return papers


def search(cfg: dict) -> list[dict]:
    """Return list of paper dicts from arXiv matching the configured keywords.

    Supports pagination (arXiv max 100 per page) via ``arxiv_max_results``
    config key (default 200).
    """
    search_cfg = cfg["search"]
    query = _build_query(search_cfg["arxiv_keywords"], search_cfg["arxiv_categories"])

    # Use explicit date range if set, otherwise fallback to lookback_days
    date_from = search_cfg.get("search_date_from", "")
    if date_from:
        cutoff = datetime.fromisoformat(date_from).replace(tzinfo=timezone.utc)
    else:
        cutoff = datetime.now(timezone.utc) - timedelta(days=search_cfg["arxiv_lookback_days"])
    date_to_str = search_cfg.get("search_date_to", "")
    date_to_dt = None
    if date_to_str:
        date_to_dt = datetime.fromisoformat(date_to_str).replace(
            hour=23, minute=59, second=59, tzinfo=timezone.utc
        )
    max_results = search_cfg.get("arxiv_max_results", 200)

    log.info("arXiv query: %s (max=%d)", query, max_results)

    all_papers: list[dict] = []
    start = 0
    page_size = min(100, max_results)

    while start < max_results:
        params = {
            "search_query": query,
            "start": start,
            "max_results": page_size,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        try:
            resp = httpx.get(_ARXIV_API, params=params, timeout=30, follow_redirects=True)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            log.error("arXiv API request failed (start=%d): %s", start, e)
            break

        root = ET.fromstring(resp.text)
        page_papers = _parse_entries(root, cutoff, date_to_dt)
        all_papers.extend(page_papers)

        # If we got fewer entries than page_size, no more pages
        entries_found = len(root.findall("atom:entry", _NS))
        if entries_found < page_size:
            break

        start += page_size
        time.sleep(_RATE_LIMIT_SECS)

    log.info("arXiv returned %d papers after date filter", len(all_papers))
    return all_papers


def search_broad(
    categories: list[str],
    lookback_days: int = 7,
    max_results: int = 200,
) -> list[dict]:
    """Broad category-only search for discovery pipelines.

    Unlike search(), this doesn't require keywords — it fetches recent papers
    from the given categories. Supports pagination (arXiv max 100 per page).
    """
    cat_query = " OR ".join(f"cat:{c}" for c in categories)
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    all_papers: list[dict] = []
    start = 0
    page_size = min(100, max_results)

    log.info("arXiv broad search: categories=%s, lookback=%dd, max=%d",
             categories, lookback_days, max_results)

    while start < max_results:
        params = {
            "search_query": cat_query,
            "start": start,
            "max_results": page_size,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        try:
            resp = httpx.get(_ARXIV_API, params=params, timeout=30, follow_redirects=True)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            log.error("arXiv broad search page %d failed: %s", start, e)
            break

        root = ET.fromstring(resp.text)
        page_papers = _parse_entries(root, cutoff)
        all_papers.extend(page_papers)

        # If we got fewer than page_size, no more pages
        entries_found = len(root.findall("atom:entry", _NS))
        if entries_found < page_size:
            break

        start += page_size
        time.sleep(_RATE_LIMIT_SECS)

    log.info("arXiv broad search returned %d papers total", len(all_papers))
    return all_papers


def _parse_entries_any(root: ET.Element) -> list[dict]:
    """Parse arXiv Atom entries without date filtering (for historical search)."""
    papers: list[dict] = []
    for entry in root.findall("atom:entry", _NS):
        paper_id_url = entry.findtext("atom:id", "", _NS).strip()
        arxiv_id = paper_id_url.rsplit("/", 1)[-1]
        if "v" in arxiv_id:
            arxiv_id = arxiv_id.rsplit("v", 1)[0]

        published_str = entry.findtext("atom:published", "", _NS).strip()
        title = " ".join(entry.findtext("atom:title", "", _NS).split())
        authors = ", ".join(
            a.findtext("atom:name", "", _NS)
            for a in entry.findall("atom:author", _NS)
        )
        abstract = " ".join(entry.findtext("atom:summary", "", _NS).split())

        if not title or not arxiv_id:
            continue

        papers.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "published": published_str,
            "summary": "",
            "key_insight": "",
            "method": "",
            "contribution": "",
            "math_concepts": [],
            "venue": "",
            "cited_works": [],
        })
    return papers


def search_random_era(
    categories: list[str],
    max_results: int = 30,
) -> list[dict]:
    """Search arXiv for papers from random historical eras.

    Picks 3-4 random year-month windows across 1994-2024 and fetches papers
    from those periods. This enables serendipitous discovery of older
    mathematical ideas that could inspire modern ML research.
    """
    # Pick random eras: some old (1994-2010), some medium (2010-2020), some recent (2020-2024)
    eras = []
    # 1-2 "classic" eras (1994-2010)
    for _ in range(random.randint(1, 2)):
        y = random.randint(1994, 2010)
        m = random.randint(1, 12)
        eras.append((y, m))
    # 1-2 "modern" eras (2010-2020)
    for _ in range(random.randint(1, 2)):
        y = random.randint(2010, 2020)
        m = random.randint(1, 12)
        eras.append((y, m))
    # 0-1 "recent" era (2020-2024)
    if random.random() < 0.5:
        y = random.randint(2020, 2024)
        m = random.randint(1, 12)
        eras.append((y, m))

    random.shuffle(eras)
    cat_query = " OR ".join(f"cat:{c}" for c in categories)

    all_papers: list[dict] = []
    per_era = max(5, max_results // len(eras))

    log.info("arXiv random era search: %d eras → %s, categories=%s",
             len(eras), [(y, m) for y, m in eras], categories)

    for year, month in eras:
        # Build date range for the month
        start_date = f"{year}{month:02d}01"
        if month == 12:
            end_date = f"{year + 1}0101"
        else:
            end_date = f"{year}{month + 1:02d}01"

        query = f"({cat_query}) AND submittedDate:[{start_date}0000 TO {end_date}0000]"

        # Random offset to avoid always getting the same papers
        random_start = random.randint(0, 50)

        params = {
            "search_query": query,
            "start": random_start,
            "max_results": per_era,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        try:
            resp = httpx.get(_ARXIV_API, params=params, timeout=30, follow_redirects=True)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            log.warning("arXiv era search %d-%02d failed: %s", year, month, e)
            time.sleep(_RATE_LIMIT_SECS)
            continue

        root = ET.fromstring(resp.text)
        papers = _parse_entries_any(root)

        for p in papers:
            p["era"] = f"{year}-{month:02d}"

        all_papers.extend(papers)
        log.info("Era %d-%02d: fetched %d papers (offset=%d)",
                 year, month, len(papers), random_start)

        time.sleep(_RATE_LIMIT_SECS)

        if len(all_papers) >= max_results:
            break

    log.info("arXiv random era search returned %d papers total", len(all_papers))
    return all_papers[:max_results]


def _sanitize_query(query: str) -> str:
    """Remove characters that break arXiv API queries (unbalanced parens, etc.)."""
    # Remove parentheses, brackets, and other special arXiv query chars
    sanitized = re.sub(r"[()[\]{}:\"']", " ", query)
    # Collapse whitespace
    return re.sub(r"\s+", " ", sanitized).strip()


def search_by_query(
    query: str,
    max_results: int = 20,
) -> list[dict]:
    """Search arXiv by free-text query, sorted by relevance (all time).

    Used for prior-art checks — finds the most relevant papers across
    all of arXiv history for a given research concept.
    """
    query = _sanitize_query(query)
    if not query:
        return []
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    log.info("arXiv query search: %s (max=%d)", query, max_results)

    try:
        resp = httpx.get(_ARXIV_API, params=params, timeout=30, follow_redirects=True)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        log.error("arXiv query search failed: %s", e)
        return []

    time.sleep(_RATE_LIMIT_SECS)

    root = ET.fromstring(resp.text)
    papers = _parse_entries_any(root)

    log.info("arXiv query search returned %d papers", len(papers))
    return papers
