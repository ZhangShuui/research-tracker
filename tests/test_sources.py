"""Unit tests for sources — arxiv.search_broad, huggingface, paperswithcode."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

import pytest

import httpx

from paper_tracker.sources import httpx_get_with_retry, DEFAULT_USER_AGENT
from paper_tracker.sources.arxiv import (
    search_broad,
    search as arxiv_search,
    _parse_entries,
    extract_arxiv_id,
    fetch_by_id,
    fetch_many_by_id,
)
from paper_tracker.sources.huggingface import fetch_daily_papers
from paper_tracker.sources.openalex import _parse_item as oa_parse_item
from paper_tracker.sources.paperswithcode import fetch_trending


def _status_error(code: int, headers: dict | None = None) -> httpx.HTTPStatusError:
    """Build an httpx.HTTPStatusError whose response carries *code* + *headers*."""
    resp = MagicMock()
    resp.status_code = code
    resp.headers = headers or {}
    return httpx.HTTPStatusError(f"HTTP {code}", request=MagicMock(), response=resp)


def _resp(text: str = "", *, raises: Exception | None = None) -> MagicMock:
    r = MagicMock()
    r.text = text
    r.raise_for_status = MagicMock(side_effect=raises)
    return r


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

_RECENT_PUBLISHED = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _make_arxiv_xml(entries: list[dict]) -> str:
    """Build a minimal arXiv Atom XML response.

    Default ``published`` is yesterday UTC so entries survive any reasonable
    lookback window in tests without needing to be refreshed over time.
    """
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<feed xmlns="http://www.w3.org/2005/Atom">']
    for e in entries:
        lines.append(f"""
  <entry>
    <id>http://arxiv.org/abs/{e['id']}v1</id>
    <published>{e.get('published', _RECENT_PUBLISHED)}</published>
    <title>{e.get('title', 'Test Paper')}</title>
    <summary>{e.get('abstract', 'An abstract.')}</summary>
    <author><name>{e.get('author', 'Author A')}</name></author>
  </entry>""")
    lines.append("</feed>")
    return "\n".join(lines)


# ---------------------------------------------------------------
# _parse_entries
# ---------------------------------------------------------------

class TestParseEntries:
    def test_basic_parsing(self):
        xml = _make_arxiv_xml([{"id": "2603.01234", "title": "Good Paper"}])
        root = ET.fromstring(xml)
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        papers = _parse_entries(root, cutoff)
        assert len(papers) == 1
        assert papers[0]["arxiv_id"] == "2603.01234"
        assert papers[0]["title"] == "Good Paper"
        assert papers[0]["url"] == "https://arxiv.org/abs/2603.01234"
        # arxiv source should synthesize an arxiv DOI for cross-source dedup
        assert papers[0]["doi"] == "10.48550/arxiv.2603.01234"

    def test_filters_old_papers(self):
        recent = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
        old = (datetime.now(timezone.utc) - timedelta(days=60)).strftime("%Y-%m-%dT%H:%M:%SZ")
        xml = _make_arxiv_xml([
            {"id": "2603.01234", "published": recent},
            {"id": "2401.99999", "published": old},
        ])
        root = ET.fromstring(xml)
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        papers = _parse_entries(root, cutoff)
        assert len(papers) == 1
        assert papers[0]["arxiv_id"] == "2603.01234"

    def test_strips_version_suffix(self):
        xml = _make_arxiv_xml([{"id": "2603.01234"}])
        root = ET.fromstring(xml)
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        papers = _parse_entries(root, cutoff)
        assert papers[0]["arxiv_id"] == "2603.01234"

    def test_empty_feed(self):
        xml = _make_arxiv_xml([])
        root = ET.fromstring(xml)
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        papers = _parse_entries(root, cutoff)
        assert papers == []

    def test_paper_fields_populated(self):
        xml = _make_arxiv_xml([{
            "id": "2603.01234",
            "title": "Title",
            "abstract": "Abstract text",
            "author": "John Doe",
        }])
        root = ET.fromstring(xml)
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        papers = _parse_entries(root, cutoff)
        p = papers[0]
        assert p["authors"] == "John Doe"
        assert p["abstract"] == "Abstract text"
        assert p["summary"] == ""
        assert p["key_insight"] == ""
        assert p["math_concepts"] == []


# ---------------------------------------------------------------
# OpenAlex _parse_item — DOI emission for cross-source dedup
# ---------------------------------------------------------------

class TestOpenAlexDoiEmission:
    def _base_item(self, **overrides) -> dict:
        item = {
            "title": "Test Paper",
            "ids": {"doi": "https://doi.org/10.1109/CVPR.2024.123",
                    "openalex": "https://openalex.org/W12345"},
            "authorships": [],
            "abstract": "",
            "publication_date": "2024-05-01",
            "publication_year": 2024,
            "primary_location": {"source": {"display_name": "CVPR"}},
            "locations": [],
            "cited_by_count": 0,
        }
        item.update(overrides)
        return item

    def test_emits_doi_field(self):
        p = oa_parse_item(self._base_item())
        assert p is not None
        assert "doi" in p
        # Raw DOI preserved — storage normalizes on insert
        assert "cvpr" in p["doi"].lower()

    def test_doi_field_empty_when_missing(self):
        item = self._base_item(ids={"openalex": "https://openalex.org/W1"})
        item.pop("doi", None)
        p = oa_parse_item(item)
        assert p is not None
        assert p["doi"] == ""

    def test_arxiv_doi_extraction(self):
        """Arxiv-style DOI → arxiv_id extracted AND doi preserved for dedup."""
        item = self._base_item(
            ids={"doi": "https://doi.org/10.48550/arxiv.2401.12345"},
        )
        p = oa_parse_item(item)
        assert p["paper_id"] == "2401.12345"
        assert "arxiv.2401.12345" in p["doi"]


# ---------------------------------------------------------------
# search_broad
# ---------------------------------------------------------------

class TestSearchBroad:
    @patch("paper_tracker.sources.arxiv.httpx.get")
    @patch("paper_tracker.sources.arxiv.time.sleep")
    def test_single_page(self, mock_sleep, mock_get):
        xml = _make_arxiv_xml([
            {"id": "2603.01234"},
            {"id": "2603.01235"},
        ])
        mock_resp = MagicMock()
        mock_resp.text = xml
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        papers = search_broad(["cs.AI", "cs.LG"], lookback_days=7, max_results=100)
        assert len(papers) == 2
        mock_get.assert_called_once()

    @patch("paper_tracker.sources.arxiv.httpx.get")
    @patch("paper_tracker.sources.arxiv.time.sleep")
    def test_pagination(self, mock_sleep, mock_get):
        """When first page is full (100 entries), should fetch another page."""
        entries_page1 = [{"id": f"2603.{i:05d}"} for i in range(100)]
        entries_page2 = [{"id": f"2603.{i:05d}"} for i in range(100, 120)]

        resp1 = MagicMock()
        resp1.text = _make_arxiv_xml(entries_page1)
        resp1.raise_for_status = MagicMock()

        resp2 = MagicMock()
        resp2.text = _make_arxiv_xml(entries_page2)
        resp2.raise_for_status = MagicMock()

        mock_get.side_effect = [resp1, resp2]

        papers = search_broad(["cs.AI"], lookback_days=30, max_results=200)
        assert len(papers) == 120
        assert mock_get.call_count == 2

    @patch("paper_tracker.sources.arxiv.httpx.get")
    @patch("paper_tracker.sources.arxiv.time.sleep")
    def test_stops_at_max_results(self, mock_sleep, mock_get):
        """Should not fetch more pages than needed."""
        entries = [{"id": f"2603.{i:05d}"} for i in range(50)]
        resp = MagicMock()
        resp.text = _make_arxiv_xml(entries)
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp

        papers = search_broad(["cs.AI"], lookback_days=30, max_results=50)
        assert mock_get.call_count == 1

    @patch("paper_tracker.sources.arxiv.httpx.get")
    @patch("paper_tracker.sources.arxiv.time.sleep")
    def test_http_error_returns_partial(self, mock_sleep, mock_get):
        import httpx
        mock_get.side_effect = httpx.HTTPError("Connection failed")

        papers = search_broad(["cs.AI"], lookback_days=7, max_results=100)
        assert papers == []

    @patch("paper_tracker.sources.arxiv.httpx.get")
    @patch("paper_tracker.sources.arxiv.time.sleep")
    def test_respects_rate_limit(self, mock_sleep, mock_get):
        """Should sleep between pages."""
        entries_page1 = [{"id": f"2603.{i:05d}"} for i in range(100)]
        entries_page2 = [{"id": f"2603.{i:05d}"} for i in range(100, 110)]

        resp1 = MagicMock()
        resp1.text = _make_arxiv_xml(entries_page1)
        resp1.raise_for_status = MagicMock()
        resp2 = MagicMock()
        resp2.text = _make_arxiv_xml(entries_page2)
        resp2.raise_for_status = MagicMock()
        mock_get.side_effect = [resp1, resp2]

        search_broad(["cs.AI"], lookback_days=30, max_results=200)
        # sleep called after first page (before fetching second)
        assert mock_sleep.call_count >= 1


# ---------------------------------------------------------------
# HuggingFace fetch_daily_papers
# ---------------------------------------------------------------

class TestHuggingFaceFetch:
    @patch("paper_tracker.sources.huggingface.httpx.get")
    def test_basic_fetch(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {
                "paper": {
                    "id": "2603.01234",
                    "title": "HF Paper 1",
                    "summary": "An abstract",
                    "authors": [{"name": "Alice"}, {"name": "Bob"}],
                    "publishedAt": "2026-03-04",
                    "upvotes": 42,
                },
            },
            {
                "paper": {
                    "id": "2603.01235",
                    "title": "HF Paper 2",
                    "summary": "Another abstract",
                    "authors": [],
                    "publishedAt": "2026-03-04",
                },
            },
        ]
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        papers = fetch_daily_papers()
        assert len(papers) == 2
        assert papers[0]["arxiv_id"] == "2603.01234"
        assert papers[0]["title"] == "HF Paper 1"
        assert papers[0]["source"] == "huggingface"
        assert papers[0]["authors"] == "Alice, Bob"
        assert papers[0]["url"] == "https://arxiv.org/abs/2603.01234"

    @patch("paper_tracker.sources.huggingface.httpx.get")
    def test_skips_entries_without_id(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"paper": {"id": "", "title": "No ID"}},
            {"paper": {"id": "2603.01234", "title": "Has ID"}},
        ]
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        papers = fetch_daily_papers()
        assert len(papers) == 1
        assert papers[0]["arxiv_id"] == "2603.01234"

    @patch("paper_tracker.sources.huggingface.httpx.get")
    def test_http_error(self, mock_get):
        import httpx
        mock_get.side_effect = httpx.HTTPError("API down")

        papers = fetch_daily_papers()
        assert papers == []

    @patch("paper_tracker.sources.huggingface.httpx.get")
    def test_empty_response(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        papers = fetch_daily_papers()
        assert papers == []


# ---------------------------------------------------------------
# Papers With Code fetch_trending
# ---------------------------------------------------------------

class TestPapersWithCodeFetch:
    @patch("paper_tracker.sources.paperswithcode.httpx.get")
    def test_basic_fetch(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "results": [
                {
                    "arxiv_id": "https://arxiv.org/abs/2603.01234v1",
                    "title": "PwC Paper 1",
                    "abstract": "Abstract",
                    "authors": ["Alice", "Bob"],
                    "published": "2026-03-04",
                },
                {
                    "arxiv_id": "https://arxiv.org/abs/2603.01235",
                    "title": "PwC Paper 2",
                    "abstract": "Abstract 2",
                    "authors": [],
                    "published": "2026-03-03",
                },
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        papers = fetch_trending(max_papers=50)
        assert len(papers) == 2
        assert papers[0]["arxiv_id"] == "2603.01234"
        assert papers[0]["source"] == "paperswithcode"
        assert papers[0]["authors"] == "Alice, Bob"
        assert papers[1]["arxiv_id"] == "2603.01235"

    @patch("paper_tracker.sources.paperswithcode.httpx.get")
    def test_skips_entries_without_arxiv_url(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "results": [
                {"arxiv_id": "", "title": "No arXiv"},
                {"arxiv_id": "https://arxiv.org/abs/2603.01234", "title": "Has arXiv"},
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        papers = fetch_trending()
        assert len(papers) == 1

    @patch("paper_tracker.sources.paperswithcode.httpx.get")
    def test_http_error(self, mock_get):
        import httpx
        mock_get.side_effect = httpx.HTTPError("API down")

        papers = fetch_trending()
        assert papers == []

    @patch("paper_tracker.sources.paperswithcode.httpx.get")
    def test_empty_results(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": []}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        papers = fetch_trending()
        assert papers == []

    @patch("paper_tracker.sources.paperswithcode.httpx.get")
    def test_passes_max_papers_param(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": []}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        fetch_trending(max_papers=25)
        call_kwargs = mock_get.call_args
        assert call_kwargs[1]["params"]["items_per_page"] == 25


# ---------------------------------------------------------------
# httpx_get_with_retry — User-Agent + 429/503 backoff (arXiv hardening)
# ---------------------------------------------------------------

class TestHttpxGetWithRetry:
    @patch("paper_tracker.sources.httpx.get")
    def test_sends_descriptive_user_agent(self, mock_get):
        mock_get.return_value = _resp("ok")
        httpx_get_with_retry("https://example.com")
        sent_headers = mock_get.call_args[1]["headers"]
        assert sent_headers["User-Agent"] == DEFAULT_USER_AGENT
        assert "paper-tracker" in sent_headers["User-Agent"]

    @patch("paper_tracker.sources.httpx.get")
    def test_caller_headers_override_default_ua(self, mock_get):
        mock_get.return_value = _resp("ok")
        httpx_get_with_retry("https://example.com", headers={"User-Agent": "custom/1.0"})
        assert mock_get.call_args[1]["headers"]["User-Agent"] == "custom/1.0"

    @patch("paper_tracker.sources.time.sleep")
    @patch("paper_tracker.sources.httpx.get")
    def test_retries_on_429_then_succeeds(self, mock_get, mock_sleep):
        mock_get.side_effect = [
            _resp(raises=_status_error(429)),
            _resp(raises=_status_error(429)),
            _resp("finally"),
        ]
        resp = httpx_get_with_retry("https://export.arxiv.org/api/query")
        assert resp.text == "finally"
        assert mock_get.call_count == 3
        assert mock_sleep.call_count == 2  # backed off before each retry

    @patch("paper_tracker.sources.time.sleep")
    @patch("paper_tracker.sources.httpx.get")
    def test_retries_on_503(self, mock_get, mock_sleep):
        # arXiv historically threw 503 when throttling — must also be retried.
        mock_get.side_effect = [_resp(raises=_status_error(503)), _resp("ok")]
        resp = httpx_get_with_retry("https://export.arxiv.org/api/query")
        assert resp.text == "ok"
        assert mock_get.call_count == 2

    @patch("paper_tracker.sources.time.sleep")
    @patch("paper_tracker.sources.httpx.get")
    def test_honors_retry_after_header(self, mock_get, mock_sleep):
        mock_get.side_effect = [
            _resp(raises=_status_error(429, headers={"Retry-After": "7"})),
            _resp("ok"),
        ]
        httpx_get_with_retry("https://export.arxiv.org/api/query")
        # First (and only) backoff should wait exactly the server-asked 7s.
        assert mock_sleep.call_args_list[0][0][0] == 7.0

    @patch("paper_tracker.sources.time.sleep")
    @patch("paper_tracker.sources.httpx.get")
    def test_raises_after_exhausting_retries(self, mock_get, mock_sleep):
        mock_get.side_effect = [_resp(raises=_status_error(429)) for _ in range(5)]
        with pytest.raises(httpx.HTTPStatusError):
            httpx_get_with_retry("https://export.arxiv.org/api/query", retries=5)
        assert mock_get.call_count == 5

    @patch("paper_tracker.sources.time.sleep")
    @patch("paper_tracker.sources.httpx.get")
    def test_does_not_retry_client_errors(self, mock_get, mock_sleep):
        mock_get.side_effect = [_resp(raises=_status_error(404))]
        with pytest.raises(httpx.HTTPStatusError):
            httpx_get_with_retry("https://example.com")
        assert mock_get.call_count == 1
        assert mock_sleep.call_count == 0


# ---------------------------------------------------------------
# arxiv.search — distinguishes total failure from empty result
# ---------------------------------------------------------------

class TestArxivSearchFailureSignal:
    def _cfg(self) -> dict:
        return {"search": {
            "arxiv_keywords": ["world model"],
            "arxiv_categories": ["cs.CV"],
            "arxiv_lookback_days": 30,
        }}

    @patch("paper_tracker.sources.arxiv.httpx_get_with_retry")
    def test_total_failure_raises(self, mock_fetch):
        """First request fails → raise so the pipeline can flag the source."""
        mock_fetch.side_effect = httpx.HTTPError("arXiv 429/timeout")
        with pytest.raises(httpx.HTTPError):
            arxiv_search(self._cfg())

    @patch("paper_tracker.sources.arxiv.time.sleep")
    @patch("paper_tracker.sources.arxiv.httpx_get_with_retry")
    def test_partial_failure_returns_what_was_fetched(self, mock_fetch, mock_sleep):
        """Later page fails after page 1 succeeded → return the partial result."""
        page1 = _resp(_make_arxiv_xml([{"id": f"2603.{i:05d}"} for i in range(100)]))
        mock_fetch.side_effect = [page1, httpx.HTTPError("page 2 failed")]
        papers = arxiv_search(self._cfg())
        assert len(papers) == 100  # page 1 kept, no exception raised

    @patch("paper_tracker.sources.arxiv.httpx_get_with_retry")
    def test_genuine_empty_returns_empty(self, mock_fetch):
        """API responds fine with zero entries → empty list, no exception."""
        mock_fetch.return_value = _resp(_make_arxiv_xml([]))
        assert arxiv_search(self._cfg()) == []


# ---------------------------------------------------------------
# OpenReview — venue listing + local keyword filter (no date floor)
# ---------------------------------------------------------------

class TestOpenReviewVenueListing:
    def _notes_resp(self, notes):
        r = MagicMock()
        r.json.return_value = {"notes": notes}
        return r

    def _note(self, title, abstract="", fid="f1"):
        return {"forum": fid, "id": fid, "cdate": 1700000000000,
                "content": {"title": {"value": title}, "abstract": {"value": abstract}}}

    @patch("paper_tracker.sources.openreview_api.time.sleep")
    @patch("paper_tracker.sources.openreview_api.httpx_get_with_retry")
    def test_keyword_filters_venue_notes_locally(self, mock_get, mock_sleep):
        from paper_tracker.sources.openreview_api import search
        notes = [
            self._note("A video world model for control", fid="f1"),
            self._note("Unrelated parsing paper", abstract="syntax trees", fid="f2"),
            self._note("Controllable video generation", fid="f3"),
        ]
        mock_get.return_value = self._notes_resp(notes)  # <100 notes → single page
        cfg = {"search": {"openreview_venues": ["iclr2025"],
                          "openreview_keywords": ["world model", "video generation"],
                          "openreview_max_results": 10}}
        titles = [p["title"] for p in search(cfg)]
        assert "A video world model for control" in titles
        assert "Controllable video generation" in titles
        assert "Unrelated parsing paper" not in titles

    @patch("paper_tracker.sources.openreview_api.time.sleep")
    @patch("paper_tracker.sources.openreview_api.httpx_get_with_retry")
    def test_queries_by_venueid_and_ignores_date_floor(self, mock_get, mock_sleep):
        from paper_tracker.sources.openreview_api import search
        # cdate ~2023 (well before the date floor) must NOT be filtered out.
        old = {"forum": "f9", "id": "f9", "cdate": 1690000000000,
               "content": {"title": {"value": "world model paper"}, "abstract": {"value": ""}}}
        mock_get.return_value = self._notes_resp([old])
        cfg = {"search": {"openreview_venues": ["neurips2025"],
                          "openreview_keywords": ["world model"],
                          "openreview_max_results": 10,
                          "search_date_from": "2025-06-01"}}  # floor must be ignored here
        papers = search(cfg)
        assert len(papers) == 1
        # venueid-based query, not the cross-venue /search endpoint
        assert mock_get.call_args[1]["params"]["content.venueid"] == "NeurIPS.cc/2025/Conference"

    @patch("paper_tracker.sources.openreview_api.time.sleep")
    @patch("paper_tracker.sources.openreview_api.httpx_get_with_retry")
    def test_no_keywords_keeps_all(self, mock_get, mock_sleep):
        from paper_tracker.sources.openreview_api import search
        mock_get.return_value = self._notes_resp([self._note("anything", fid="f1")])
        cfg = {"search": {"openreview_venues": ["icml2025"],
                          "openreview_keywords": [], "arxiv_keywords": [],
                          "openreview_max_results": 10}}
        assert len(search(cfg)) == 1

    def test_no_venues_returns_empty(self):
        from paper_tracker.sources.openreview_api import search
        cfg = {"search": {"openreview_venues": [], "openreview_keywords": ["x"]}}
        assert search(cfg) == []


# ---------------------------------------------------------------
# extract_arxiv_id — normalize raw IDs / URLs to a bare arXiv ID
# ---------------------------------------------------------------

class TestExtractArxivId:
    @pytest.mark.parametrize("raw,expected", [
        ("2401.12345", "2401.12345"),
        ("2401.12345v3", "2401.12345"),
        ("  2401.12345  ", "2401.12345"),
        ("arXiv:2401.12345", "2401.12345"),
        ("ARXIV: 2401.12345", "2401.12345"),
        ("https://arxiv.org/abs/2401.12345", "2401.12345"),
        ("https://arxiv.org/abs/2401.12345v2", "2401.12345"),
        ("http://export.arxiv.org/abs/2401.12345", "2401.12345"),
        ("https://arxiv.org/pdf/2401.12345.pdf", "2401.12345"),
        ("https://arxiv.org/pdf/2401.12345", "2401.12345"),
        ("2401.1234", "2401.1234"),          # 4-digit (older) sequence
        ("hep-th/9901001", "hep-th/9901001"),  # legacy id scheme
        ("cs.AI/0501001", "cs.AI/0501001"),
        ("not a paper", ""),
        ("", ""),
    ])
    def test_extract(self, raw, expected):
        assert extract_arxiv_id(raw) == expected


# ---------------------------------------------------------------
# fetch_by_id — single-paper metadata lookup via id_list
# ---------------------------------------------------------------

class TestFetchById:
    @patch("paper_tracker.sources.arxiv.httpx_get_with_retry")
    def test_returns_normalized_paper(self, mock_get):
        xml = _make_arxiv_xml([{
            "id": "2401.12345",
            "title": "A Manual Paper",
            "abstract": "Some abstract.",
            "author": "Jane Doe",
            "published": "2024-01-15T00:00:00Z",
        }])
        mock_get.return_value = _resp(xml)

        paper = fetch_by_id("https://arxiv.org/abs/2401.12345")

        assert paper is not None
        assert paper["arxiv_id"] == "2401.12345"
        assert paper["title"] == "A Manual Paper"
        assert paper["authors"] == "Jane Doe"
        assert paper["published"] == "2024-01-15"  # normalized to YYYY-MM-DD
        assert paper["url"] == "https://arxiv.org/abs/2401.12345"
        # request used the cleaned id, not the raw URL
        assert mock_get.call_args.kwargs["params"]["id_list"] == "2401.12345"

    @patch("paper_tracker.sources.arxiv.httpx_get_with_retry")
    def test_not_found_returns_none(self, mock_get):
        mock_get.return_value = _resp(_make_arxiv_xml([]))
        assert fetch_by_id("2401.00000") is None

    @patch("paper_tracker.sources.arxiv.httpx_get_with_retry")
    def test_id_mismatch_returns_none(self, mock_get):
        # arXiv echoes a different/error id for unknown requests
        mock_get.return_value = _resp(_make_arxiv_xml([{"id": "2401.99999"}]))
        assert fetch_by_id("2401.12345") is None

    @patch("paper_tracker.sources.arxiv.httpx_get_with_retry")
    def test_unparseable_input_skips_request(self, mock_get):
        assert fetch_by_id("just some text") is None
        mock_get.assert_not_called()

    @patch("paper_tracker.sources.arxiv.httpx_get_with_retry",
           side_effect=httpx.HTTPError("503"))
    def test_http_error_propagates(self, mock_get):
        with pytest.raises(httpx.HTTPError):
            fetch_by_id("2401.12345")


class TestFetchManyById:
    @patch("paper_tracker.sources.arxiv.httpx_get_with_retry")
    def test_fetches_in_one_request(self, mock_get):
        xml = _make_arxiv_xml([
            {"id": "2401.11111", "title": "P1", "published": "2024-01-01T00:00:00Z"},
            {"id": "2401.22222", "title": "P2", "published": "2024-02-02T00:00:00Z"},
        ])
        mock_get.return_value = _resp(xml)
        out = fetch_many_by_id(["https://arxiv.org/abs/2401.11111", "arXiv:2401.22222v3"])
        assert set(out.keys()) == {"2401.11111", "2401.22222"}
        assert out["2401.11111"]["title"] == "P1"
        assert out["2401.22222"]["published"] == "2024-02-02"  # normalized
        assert mock_get.call_count == 1
        assert mock_get.call_args.kwargs["params"]["id_list"] == "2401.11111,2401.22222"

    @patch("paper_tracker.sources.arxiv.httpx_get_with_retry")
    def test_dedups_and_skips_invalid(self, mock_get):
        mock_get.return_value = _resp(_make_arxiv_xml([{"id": "2401.11111"}]))
        out = fetch_many_by_id(["2401.11111", "2401.11111", "not-an-id", ""])
        assert list(out.keys()) == ["2401.11111"]
        assert mock_get.call_args.kwargs["params"]["id_list"] == "2401.11111"

    def test_all_invalid_makes_no_request(self):
        with patch("paper_tracker.sources.arxiv.httpx_get_with_retry") as mock_get:
            assert fetch_many_by_id(["nope", "also nope", ""]) == {}
            mock_get.assert_not_called()

    @patch("paper_tracker.sources.arxiv.httpx_get_with_retry")
    def test_unknown_id_absent(self, mock_get):
        mock_get.return_value = _resp(_make_arxiv_xml([{"id": "2401.11111"}]))
        out = fetch_many_by_id(["2401.11111", "2401.99999"])
        assert "2401.11111" in out and "2401.99999" not in out
