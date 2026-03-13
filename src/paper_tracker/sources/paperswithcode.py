"""Fetch trending papers from Papers With Code."""

from __future__ import annotations

import logging
import re

import httpx

log = logging.getLogger(__name__)

_PWC_API = "https://paperswithcode.com/api/v1/papers/"


def fetch_trending(max_papers: int = 50) -> list[dict]:
    """Fetch trending papers from Papers With Code API.

    Returns list of paper dicts with source="paperswithcode".
    """
    log.info("Fetching Papers With Code trending (max=%d)", max_papers)

    try:
        resp = httpx.get(
            _PWC_API,
            params={"ordering": "-trending", "items_per_page": min(max_papers, 50)},
            timeout=30,
            follow_redirects=True,
        )
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as e:
        log.error("Papers With Code API request failed: %s", e)
        return []
    except Exception as e:
        log.error("Papers With Code API parse error: %s", e)
        return []

    results = data.get("results", [])
    papers: list[dict] = []

    for item in results:
        arxiv_id = ""
        arxiv_url = item.get("arxiv_id") or ""
        if arxiv_url:
            # Extract ID from URL like "https://arxiv.org/abs/2401.12345v1"
            match = re.search(r"(\d{4}\.\d{4,5})", arxiv_url)
            if match:
                arxiv_id = match.group(1)

        if not arxiv_id:
            continue

        papers.append({
            "arxiv_id": arxiv_id,
            "title": item.get("title", ""),
            "authors": ", ".join(item.get("authors", [])) if isinstance(item.get("authors"), list) else "",
            "abstract": item.get("abstract", ""),
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "published": item.get("published", ""),
            "source": "paperswithcode",
        })

    log.info("Papers With Code returned %d papers with arXiv IDs", len(papers))
    return papers
