"""Search OpenReview API for conference papers."""

from __future__ import annotations

import logging
import time

import httpx

log = logging.getLogger(__name__)

_OR_API = "https://api2.openreview.net/notes/search"
_RATE_LIMIT_SECS = 1.0

KNOWN_VENUES = {
    "iclr2025": "ICLR.cc/2025/Conference",
    "iclr2024": "ICLR.cc/2024/Conference",
    "neurips2024": "NeurIPS.cc/2024/Conference",
    "neurips2025": "NeurIPS.cc/2025/Conference",
    "icml2024": "ICML.cc/2024/Conference",
    "icml2025": "ICML.cc/2025/Conference",
    "acl2024": "aclweb.org/ACL/2024/Conference",
    "acl2025": "aclweb.org/ACL/2025/Conference",
    "aaai2025": "AAAI.org/AAAI/2025/Conference",
    "cvpr2025": "thecvf.com/CVPR/2025/Conference",
}


def search(cfg: dict) -> list[dict]:
    """Search OpenReview by venues x keywords.

    Reads from cfg["search"]:
      - openreview_venues: list of venue short names (e.g. ["iclr2025"])
      - openreview_keywords: list of keywords (falls back to arxiv_keywords)
      - openreview_max_results: max per venue (default 100)
    """
    search_cfg = cfg["search"]
    venue_keys = search_cfg.get("openreview_venues", [])
    keywords = search_cfg.get("openreview_keywords") or search_cfg.get("arxiv_keywords", [])
    max_results = search_cfg.get("openreview_max_results", 100)
    date_from = search_cfg.get("search_date_from", "")
    date_to = search_cfg.get("search_date_to", "")

    if not venue_keys:
        log.warning("OpenReview: no venues configured, skipping")
        return []

    all_papers: list[dict] = []
    seen_ids: set[str] = set()

    for venue_key in venue_keys:
        venue_id = KNOWN_VENUES.get(venue_key.lower(), venue_key)
        log.info("OpenReview: searching venue=%s (%s), keywords=%s",
                 venue_key, venue_id, keywords)

        papers = _search_venue(venue_id, keywords, max_results)
        for p in papers:
            if p["paper_id"] in seen_ids:
                continue
            # Date range post-filter
            pub = p.get("published", "")
            if date_from and pub and pub < date_from:
                continue
            if date_to and pub and pub > date_to + "T23:59:59":
                continue
            seen_ids.add(p["paper_id"])
            all_papers.append(p)

        time.sleep(_RATE_LIMIT_SECS)

    log.info("OpenReview returned %d papers total", len(all_papers))
    return all_papers


def _venue_matches(note_venue: str, venue_id: str) -> bool:
    """Check if a note's venue field matches the target venue.

    e.g. 'ICLR 2025 Poster' matches 'ICLR.cc/2025/Conference'
    """
    nv = note_venue.lower()
    parts = venue_id.split("/")
    # Extract key parts: venue name and year
    name = parts[0].split(".")[0].lower() if "." in parts[0] else parts[0].lower()
    year = ""
    for p in parts:
        if p.isdigit() and len(p) == 4:
            year = p
            break
    return name in nv and (not year or year in nv)


def _search_venue(
    venue_id: str,
    keywords: list[str],
    max_results: int,
) -> list[dict]:
    """Search a single OpenReview venue with optional keywords."""
    query = " ".join(keywords) if keywords else ""

    if query:
        # Use search endpoint with keyword filtering + venue post-filter
        search_url = _OR_API
        # Fetch more to account for venue filtering
        fetch_limit = min(max_results * 3, 300)
        params: dict = {
            "query": query,
            "limit": min(fetch_limit, 100),
            "offset": 0,
            "source": "forum",
        }
    else:
        # No keywords: use invitation-based listing
        search_url = "https://api2.openreview.net/notes"
        fetch_limit = max_results
        params = {
            "invitation": f"{venue_id}/-/Submission",
            "limit": min(fetch_limit, 100),
            "offset": 0,
        }

    all_papers: list[dict] = []
    offset = 0

    while offset < fetch_limit and len(all_papers) < max_results:
        params["offset"] = offset

        try:
            resp = httpx.get(search_url, params=params, timeout=30, follow_redirects=True)
            if resp.status_code == 429:
                log.warning("OpenReview rate limited, waiting 5s")
                time.sleep(5)
                continue
            resp.raise_for_status()
        except httpx.HTTPError as e:
            log.error("OpenReview API failed (offset=%d): %s", offset, e)
            break

        data = resp.json()
        notes = data.get("notes", [])
        if not notes:
            break

        for note in notes:
            # Venue post-filter when using search endpoint
            if query:
                content = note.get("content", {})
                note_venue = content.get("venue")
                if isinstance(note_venue, dict):
                    note_venue = note_venue.get("value", "")
                if not note_venue or not _venue_matches(note_venue, venue_id):
                    continue

            paper = _parse_note(note, venue_id)
            if paper is not None:
                all_papers.append(paper)
                if len(all_papers) >= max_results:
                    break

        if len(notes) < params.get("limit", 100):
            break

        offset += len(notes)
        time.sleep(_RATE_LIMIT_SECS)

    return all_papers


def _parse_note(note: dict, venue_id: str) -> dict | None:
    """Convert an OpenReview note into the standard paper dict."""
    content = note.get("content", {})
    forum_id = note.get("forum") or note.get("id", "")
    if not forum_id:
        return None

    # OpenReview v2: content fields have "value" subkey
    def _val(field: str, default: str = "") -> str:
        v = content.get(field)
        if isinstance(v, dict):
            return v.get("value", default)
        if isinstance(v, str):
            return v
        return default

    title = _val("title")
    if not title:
        return None

    abstract = _val("abstract")

    # Authors: can be a list of dicts or a list of strings
    authors_raw = content.get("authors")
    if isinstance(authors_raw, dict):
        authors_raw = authors_raw.get("value", [])
    if isinstance(authors_raw, list):
        authors = ", ".join(
            a if isinstance(a, str) else a.get("name", "") if isinstance(a, dict) else str(a)
            for a in authors_raw
        )
    else:
        authors = ""

    paper_id = f"or:{forum_id}"

    # Try to extract venue name from the venue_id
    # e.g. "ICLR.cc/2025/Conference" -> "ICLR 2025"
    parts = venue_id.split("/")
    venue_short = parts[0].split(".")[0] if "." in parts[0] else parts[0]
    year = ""
    for part in parts:
        if part.isdigit() and len(part) == 4:
            year = part
            break
    venue_display = f"{venue_short} {year}".strip()

    # Publication date
    cdate = note.get("cdate") or note.get("tcdate")
    if isinstance(cdate, (int, float)):
        from datetime import datetime, timezone
        pub_date = datetime.fromtimestamp(cdate / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    else:
        pub_date = ""

    return {
        "arxiv_id": paper_id,  # backward compat
        "paper_id": paper_id,
        "source": "openreview",
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "url": f"https://openreview.net/forum?id={forum_id}",
        "published": pub_date,
        "summary": "",
        "key_insight": "",
        "method": "",
        "contribution": "",
        "math_concepts": [],
        "venue": venue_display,
        "cited_works": [],
        "citation_count": 0,
    }
