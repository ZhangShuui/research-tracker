"""Tests for sources/pdf.py (PDF-URL ingestion)."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import httpx
import pytest

from paper_tracker.sources import pdf


def _resp(content: bytes) -> MagicMock:
    r = MagicMock()
    r.content = content
    return r


class TestIsPdfUrl:
    @pytest.mark.parametrize("u,exp", [
        ("https://maths.ed.ac.uk/foo.pdf", True),
        ("http://example.org/paper", True),
        ("  https://x/a.pdf ", True),
        ("2401.12345", False),
        ("arXiv:2401.12345", False),
        ("", False),
    ])
    def test_is_pdf_url(self, u, exp):
        assert pdf.is_pdf_url(u) is exp


class TestFetchPdfPaper:
    @patch("paper_tracker.sources.pdf.call_cli",
           return_value='{"title":"A Great Theorem","authors":"L. Euler","abstract":"We prove it."}')
    @patch("paper_tracker.sources.pdf._extract_text", return_value="lots of text")
    @patch("paper_tracker.sources.pdf.httpx_get_with_retry")
    def test_success_and_deterministic_id(self, mock_get, mock_text, mock_cli):
        mock_get.return_value = _resp(b"%PDF-1.5\nbinary...")
        p = pdf.fetch_pdf_paper("https://maths.ed.ac.uk/foo.pdf", {})
        assert p is not None
        assert p["title"] == "A Great Theorem"
        assert p["authors"] == "L. Euler"
        assert p["abstract"] == "We prove it."
        assert p["source"] == "pdf"
        assert p["url"] == "https://maths.ed.ac.uk/foo.pdf"
        assert p["paper_id"].startswith("pdf:") and p["paper_id"] == p["arxiv_id"]
        # same URL → same id (dedups on re-add)
        p2 = pdf.fetch_pdf_paper("https://maths.ed.ac.uk/foo.pdf", {})
        assert p2["paper_id"] == p["paper_id"]
        # sonnet used for the extraction
        assert mock_cli.call_args.kwargs.get("model") == "sonnet"

    @patch("paper_tracker.sources.pdf.httpx_get_with_retry")
    def test_not_a_pdf_returns_none(self, mock_get):
        mock_get.return_value = _resp(b"<html>not a pdf</html>")
        assert pdf.fetch_pdf_paper("https://x/page", {}) is None

    @patch("paper_tracker.sources.pdf.httpx_get_with_retry", side_effect=httpx.HTTPError("boom"))
    def test_download_failure_returns_none(self, mock_get):
        assert pdf.fetch_pdf_paper("https://x/a.pdf", {}) is None

    @patch("paper_tracker.sources.pdf.call_cli", return_value="{}")
    @patch("paper_tracker.sources.pdf._extract_text", return_value="text")
    @patch("paper_tracker.sources.pdf.httpx_get_with_retry")
    def test_no_title_returns_none(self, mock_get, mock_text, mock_cli):
        mock_get.return_value = _resp(b"%PDF-1.5")
        assert pdf.fetch_pdf_paper("https://x/a.pdf", {}) is None

    @patch("paper_tracker.sources.pdf._extract_text", return_value="")
    @patch("paper_tracker.sources.pdf.httpx_get_with_retry")
    def test_no_text_returns_none(self, mock_get, mock_text):
        mock_get.return_value = _resp(b"%PDF-1.5")
        assert pdf.fetch_pdf_paper("https://x/a.pdf", {}) is None
