"""Fetch daily curated papers from HuggingFace."""

from __future__ import annotations

import logging

import httpx

log = logging.getLogger(__name__)

_HF_API = "https://huggingface.co/api/daily_papers"


def fetch_daily_papers() -> list[dict]:
    """Fetch papers from HuggingFace Daily Papers API.

    Returns list of paper dicts with source="huggingface".
    """
    log.info("Fetching HuggingFace daily papers")

    try:
        resp = httpx.get(_HF_API, timeout=30, follow_redirects=True)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as e:
        log.error("HuggingFace API request failed: %s", e)
        return []
    except Exception as e:
        log.error("HuggingFace API parse error: %s", e)
        return []

    papers: list[dict] = []
    for item in data:
        paper = item.get("paper", {})
        arxiv_id = paper.get("id", "")
        if not arxiv_id:
            continue

        title = paper.get("title", "")
        abstract = paper.get("summary", "")
        authors = ", ".join(
            a.get("name", "") for a in paper.get("authors", [])
        )
        published = paper.get("publishedAt", "")

        papers.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "published": published,
            "source": "huggingface",
            "hf_upvotes": item.get("paper", {}).get("upvotes", 0),
        })

    log.info("HuggingFace returned %d papers", len(papers))
    return papers
