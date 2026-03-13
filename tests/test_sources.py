"""Unit tests for sources — arxiv.search_broad, huggingface, paperswithcode."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

import pytest

from paper_tracker.sources.arxiv import search_broad, _parse_entries
from paper_tracker.sources.huggingface import fetch_daily_papers
from paper_tracker.sources.paperswithcode import fetch_trending


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _make_arxiv_xml(entries: list[dict]) -> str:
    """Build a minimal arXiv Atom XML response."""
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<feed xmlns="http://www.w3.org/2005/Atom">']
    for e in entries:
        lines.append(f"""
  <entry>
    <id>http://arxiv.org/abs/{e['id']}v1</id>
    <published>{e.get('published', '2026-03-04T00:00:00Z')}</published>
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

    def test_filters_old_papers(self):
        xml = _make_arxiv_xml([
            {"id": "2603.01234", "published": "2026-03-04T00:00:00Z"},
            {"id": "2401.99999", "published": "2024-01-01T00:00:00Z"},
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
