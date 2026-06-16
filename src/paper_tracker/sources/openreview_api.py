"""Search OpenReview API for conference papers."""

from __future__ import annotations

import logging
import time

import httpx

from paper_tracker.sources import httpx_get_with_retry

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

    if not venue_keys:
        log.warning("OpenReview: no venues configured, skipping")
        return []

    all_papers: list[dict] = []
    seen_ids: set[str] = set()

    for venue_key in venue_keys:
        venue_id = KNOWN_VENUES.get(venue_key.lower(), venue_key)
        log.info("OpenReview: searching venue=%s (%s), keywords=%s",
                 venue_key, venue_id, keywords)

        # Venue selection already scopes recency (e.g. the *2025 venues), so we
        # do NOT apply search_date_from here: conference papers are submitted
        # months before the event, so a date floor would wrongly drop them all.
        papers = _search_venue(venue_id, keywords, max_results)
        for p in papers:
            if p["paper_id"] in seen_ids:
                continue
            seen_ids.add(p["paper_id"])
            all_papers.append(p)

        time.sleep(_RATE_LIMIT_SECS)

    log.info("OpenReview returned %d papers total", len(all_papers))
    return all_papers


_OR_NOTES = "https://api2.openreview.net/notes"
_VENUE_SCAN_CAP = 1000  # max notes to scan per venue while keyword-filtering


def _search_venue(
    venue_id: str,
    keywords: list[str],
    max_results: int,
) -> list[dict]:
    """List a venue's papers (by ``content.venueid``) and keyword-filter locally.

    OpenReview's keyword /search endpoint ranks across ALL venues, so it almost
    never surfaces a *specific* venue's papers. Listing the venue directly and
    matching keywords against title+abstract reliably scopes results to the venue.
    Scans up to ``_VENUE_SCAN_CAP`` notes (venues can hold thousands).
    """
    kw_lower = [k.lower() for k in keywords]
    matched: list[dict] = []
    scanned = 0
    offset = 0
    page = 100

    while scanned < _VENUE_SCAN_CAP and len(matched) < max_results:
        params = {"content.venueid": venue_id, "limit": page, "offset": offset}
        try:
            resp = httpx_get_with_retry(_OR_NOTES, params=params, timeout=30)
        except httpx.HTTPError as e:
            log.error("OpenReview venue listing failed (venue=%s offset=%d): %s",
                      venue_id, offset, e)
            break

        notes = resp.json().get("notes", [])
        if not notes:
            break
        scanned += len(notes)

        for note in notes:
            paper = _parse_note(note, venue_id)
            if paper is None:
                continue
            if kw_lower:
                hay = f"{paper['title']} {paper['abstract']}".lower()
                if not any(k in hay for k in kw_lower):
                    continue
            matched.append(paper)
            if len(matched) >= max_results:
                break

        if len(notes) < page:
            break
        offset += len(notes)
        time.sleep(_RATE_LIMIT_SECS)

    log.info("OpenReview venue=%s: %d keyword matches (scanned %d notes)",
             venue_id, len(matched), scanned)
    return matched


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
