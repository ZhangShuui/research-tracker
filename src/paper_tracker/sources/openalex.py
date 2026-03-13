"""Search OpenAlex API for papers.

OpenAlex is a free, open catalog of 250M+ scholarly works.
No API key required. Rate limit: ~10 req/s (polite pool with mailto header).
Docs: https://docs.openalex.org/
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone

import httpx

log = logging.getLogger(__name__)

_OA_API = "https://api.openalex.org/works"
_RATE_LIMIT_SECS = 0.2  # generous: 10 req/s allowed


def search(cfg: dict) -> list[dict]:
    """Keyword search + venue filter + date cutoff.

    Reads from cfg["search"]:
      - openalex_keywords (falls back to arxiv_keywords)
      - openalex_lookback_days (default 7)
      - openalex_venues (list of venue name fragments, e.g. ["ICLR", "NeurIPS"])
      - openalex_max_results (default 200)
    """
    search_cfg = cfg["search"]
    keywords = search_cfg.get("openalex_keywords") or search_cfg.get("arxiv_keywords", [])
    lookback_days = search_cfg.get("openalex_lookback_days", 7)
    venues_filter = [v.lower() for v in search_cfg.get("openalex_venues", [])]
    max_results = search_cfg.get("openalex_max_results", 200)

    if not keywords:
        log.warning("OpenAlex: no keywords configured, skipping")
        return []

    query = " ".join(keywords)
    # Use explicit date range if set, otherwise fallback to lookback_days
    date_from = search_cfg.get("search_date_from", "")
    date_to = search_cfg.get("search_date_to", "")
    if date_from:
        cutoff_str = date_from
    else:
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        cutoff_str = cutoff.strftime("%Y-%m-%d")

    # Polite pool: include mailto for higher rate limit
    mailto = os.environ.get("OPENALEX_MAILTO", "paper-tracker@example.com")
    headers = {"User-Agent": f"paper-tracker (mailto:{mailto})"}

    log.info("OpenAlex search: query=%r, venues=%s, cutoff=%s, max=%d",
             query, venues_filter, cutoff_str, max_results)

    all_papers: list[dict] = []
    seen_ids: set[str] = set()
    page = 1
    per_page = 50 if venues_filter else min(50, max_results)
    # Limit total pages to avoid infinite paging when venue filter is strict
    max_pages = max(max_results * 3 // per_page, 4) if venues_filter else max(max_results // per_page, 1)

    while len(all_papers) < max_results and page <= max_pages:
        filters = [f"from_publication_date:{cutoff_str}"]
        if date_to:
            filters.append(f"to_publication_date:{date_to}")

        params = {
            "search": query,
            "per_page": per_page,
            "page": page,
            "mailto": mailto,
        }
        if filters:
            params["filter"] = ",".join(filters)

        try:
            resp = httpx.get(_OA_API, params=params, headers=headers,
                             timeout=30, follow_redirects=True)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            log.error("OpenAlex API request failed (page=%d): %s", page, e)
            break

        data = resp.json()
        results = data.get("results", [])
        if not results:
            break

        for item in results:
            paper = _parse_item(item)
            if paper is None:
                continue
            # Dedup within results
            pid = paper["paper_id"]
            if pid in seen_ids:
                continue
            seen_ids.add(pid)
            # Venue post-filter
            if venues_filter:
                paper_venue = (paper.get("venue") or "").lower()
                if not any(v in paper_venue for v in venues_filter):
                    continue
            all_papers.append(paper)

        total_available = data.get("meta", {}).get("count", 0)
        if page * per_page >= total_available:
            break

        page += 1
        time.sleep(_RATE_LIMIT_SECS)

    log.info("OpenAlex returned %d papers after filtering", len(all_papers))
    return all_papers[:max_results]


def _parse_item(item: dict) -> dict | None:
    """Convert an OpenAlex work into the standard paper dict format."""
    title = item.get("title")
    if not title:
        return None

    ids = item.get("ids", {})
    doi_raw = ids.get("doi", "") or item.get("doi", "") or ""
    # OpenAlex DOI format: "https://doi.org/10.xxxx/yyyy"
    doi = doi_raw.replace("https://doi.org/", "").replace("http://doi.org/", "")

    # Try to extract arXiv ID from alternate locations or DOI
    arxiv_id = ""
    # Check if DOI is an arXiv DOI (e.g. 10.48550/arxiv.2603.00194)
    if doi and "arxiv." in doi.lower():
        # Extract arXiv ID from DOI like "10.48550/arxiv.2603.00194"
        parts = doi.split("arxiv.")
        if len(parts) == 2:
            arxiv_id = parts[1]
    if not arxiv_id:
        for loc in item.get("locations", []):
            landing = loc.get("landing_page_url", "") or ""
            if "arxiv.org/abs/" in landing:
                arxiv_id = landing.split("arxiv.org/abs/")[-1].split("v")[0]
                break

    # ID strategy: prefer arXiv ID, then DOI, then OpenAlex ID
    if arxiv_id:
        paper_id = arxiv_id
    elif doi:
        paper_id = f"doi:{doi}"
    else:
        oa_id = ids.get("openalex", "") or item.get("id", "")
        short_id = oa_id.rsplit("/", 1)[-1] if "/" in oa_id else oa_id
        if short_id:
            paper_id = f"oa:{short_id}"
        else:
            return None

    # Authors
    authorships = item.get("authorships", [])
    authors = ", ".join(
        a.get("author", {}).get("display_name", "")
        for a in authorships
        if a.get("author", {}).get("display_name")
    )

    abstract = item.get("abstract") or ""
    # OpenAlex may provide abstract_inverted_index instead
    if not abstract and item.get("abstract_inverted_index"):
        abstract = _reconstruct_abstract(item["abstract_inverted_index"])

    citation_count = item.get("cited_by_count", 0)
    pub_date = item.get("publication_date", "")

    # Venue info
    primary_loc = item.get("primary_location") or {}
    source = primary_loc.get("source") or {}
    venue_name = source.get("display_name", "")
    pub_year = item.get("publication_year", "")
    venue_display = f"{venue_name} {pub_year}" if venue_name and pub_year else venue_name or str(pub_year)

    # URL
    if arxiv_id:
        url = f"https://arxiv.org/abs/{arxiv_id}"
    elif doi:
        url = f"https://doi.org/{doi}"
    else:
        url = item.get("id", "")

    return {
        "arxiv_id": paper_id,  # backward compat: stored in arxiv_id column
        "paper_id": paper_id,
        "source": "openalex",
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "url": url,
        "published": pub_date,
        "summary": "",
        "key_insight": "",
        "method": "",
        "contribution": "",
        "math_concepts": [],
        "venue": venue_display,
        "cited_works": [],
        "citation_count": citation_count,
    }


def _reconstruct_abstract(inverted_index: dict) -> str:
    """Reconstruct abstract text from OpenAlex inverted index format."""
    if not inverted_index:
        return ""
    # Build (position, word) pairs
    pairs: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        for pos in positions:
            pairs.append((pos, word))
    pairs.sort()
    return " ".join(w for _, w in pairs)
