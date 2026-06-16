"""Ingest a paper from a PDF URL (for work not on arXiv — common in mathematics).

Downloads the PDF, extracts text with pypdf, and uses an LLM to pull
bibliographic metadata (title / authors / abstract).
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import re

import httpx

from paper_tracker.sources import httpx_get_with_retry
from paper_tracker.llm import call_cli

log = logging.getLogger(__name__)

_MAX_PDF_BYTES = 30 * 1024 * 1024   # 30 MB cap
_MAX_TEXT_CHARS = 12000             # feed the first ~12k chars to the LLM
_MAX_PAGES = 12                     # parse at most this many pages for metadata

_META_PROMPT = """\
Extract bibliographic metadata from the beginning of this paper.

Reply ONLY with a JSON object:
{{"title": "<title>", "authors": "<comma-separated author names>", "abstract": "<abstract or first-paragraph summary>"}}
Use "" for any field you cannot find.

--- paper text ---
{text}"""


def is_pdf_url(value: str) -> bool:
    """Heuristic: an http(s) URL we should try to fetch as a PDF."""
    v = value.strip().lower()
    return v.startswith("http://") or v.startswith("https://")


def _parse_json_obj(raw: str) -> dict:
    if not raw:
        return {}
    text = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    text = re.sub(r"\s*```$", "", text)
    try:
        d = json.loads(text)
        return d if isinstance(d, dict) else {}
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                d = json.loads(m.group(0))
                return d if isinstance(d, dict) else {}
            except json.JSONDecodeError:
                return {}
    return {}


def _extract_text(pdf_bytes: bytes, max_pages: int = _MAX_PAGES) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(pdf_bytes))
    parts: list[str] = []
    for page in reader.pages[:max_pages]:
        try:
            parts.append(page.extract_text() or "")
        except Exception:  # a single bad page shouldn't kill extraction
            continue
    return "\n".join(parts)


def download_and_extract_text(url: str, max_pages: int = _MAX_PAGES) -> str | None:
    """Download a PDF URL and extract its text. None on failure.

    Reused for concept-card generation, where we want the body text — not just
    the abstract — for long math papers/books (pass a high ``max_pages``).
    """
    url = url.strip()
    try:
        resp = httpx_get_with_retry(url, timeout=60)
    except httpx.HTTPError as e:
        log.warning("PDF download failed %s: %s", url, e)
        return None
    content = getattr(resp, "content", b"") or b""
    if not content:
        log.warning("PDF empty: %s", url)
        return None
    if len(content) > _MAX_PDF_BYTES:
        log.warning("PDF too large (%d bytes): %s", len(content), url)
        return None
    if not content[:5].startswith(b"%PDF"):
        log.warning("URL is not a PDF (no %%PDF header): %s", url)
        return None
    try:
        text = _extract_text(content, max_pages=max_pages)
    except Exception as e:
        log.warning("PDF parse failed %s: %s", url, e)
        return None
    return text.strip() or None


def fetch_pdf_paper(url: str, cfg: dict) -> dict | None:
    """Fetch + parse a PDF URL into the standard paper dict, or None on failure.

    The id is deterministic (``pdf:<sha1(url)[:12]>``) so re-adding the same URL
    dedups. Metadata (title/authors/abstract) is LLM-extracted from the text.
    """
    url = url.strip()
    text = download_and_extract_text(url)
    if not text:
        return None

    raw = call_cli(_META_PROMPT.format(text=text[:_MAX_TEXT_CHARS]), cfg,
                   model="sonnet", timeout=300)
    meta = _parse_json_obj(raw or "")
    title = (meta.get("title") or "").strip()
    if not title:
        log.warning("Could not extract a title from PDF: %s", url)
        return None

    abstract = (meta.get("abstract") or "").strip() or text[:1500].strip()
    key = "pdf:" + hashlib.sha1(url.encode()).hexdigest()[:12]
    return {
        "arxiv_id": key,
        "paper_id": key,
        "source": "pdf",
        "title": title,
        "authors": (meta.get("authors") or "").strip(),
        "abstract": abstract,
        "url": url,
        "published": "",
        "summary": abstract,
        "key_insight": "",
        "method": "",
        "contribution": "",
        "math_concepts": [],
        "venue": "",
        "cited_works": [],
        "citation_count": 0,
        "doi": "",
    }
