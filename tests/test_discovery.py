"""Unit tests for discovery.py — trending and math pipelines.

LLM calls and source fetchers are mocked to isolate pipeline logic.
"""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest

from paper_tracker.discovery import (
    run_trending,
    run_math_insights,
    _format_papers_for_prompt,
)
from paper_tracker.registry import Registry


_CFG = {
    "paths": {"data_dir": "/tmp/test"},
    "summarizer": {"claude_path": "claude", "claude_model": "opus"},
}


def _fake_paper(arxiv_id: str, source: str = "arxiv") -> dict:
    return {
        "arxiv_id": arxiv_id,
        "title": f"Paper {arxiv_id}",
        "authors": "Author A",
        "abstract": "An abstract about AI.",
        "url": f"https://arxiv.org/abs/{arxiv_id}",
        "published": "2026-03-04",
        "source": source,
    }


@pytest.fixture()
def reg(tmp_path):
    r = Registry(str(tmp_path))
    yield r
    r.close()


# ---------------------------------------------------------------
# _format_papers_for_prompt
# ---------------------------------------------------------------

class TestFormatPapersForPrompt:
    def test_basic_formatting(self):
        papers = [_fake_paper("2603.001"), _fake_paper("2603.002", "huggingface")]
        text = _format_papers_for_prompt(papers)
        assert "2603.001" in text
        assert "(arxiv," in text
        assert "(huggingface," in text

    def test_max_papers_limit(self):
        papers = [_fake_paper(f"2603.{i:05d}") for i in range(200)]
        text = _format_papers_for_prompt(papers, max_papers=5)
        # Only first 5 should appear
        assert "2603.00004" in text
        assert "2603.00005" not in text

    def test_empty_list(self):
        text = _format_papers_for_prompt([])
        assert text == ""


# ---------------------------------------------------------------
# run_trending
# ---------------------------------------------------------------

class TestRunTrending:
    @patch("paper_tracker.discovery.call_cli")
    @patch("paper_tracker.discovery.fetch_pwc_trending")
    @patch("paper_tracker.discovery.fetch_daily_papers")
    @patch("paper_tracker.discovery.search_broad")
    def test_happy_path(self, mock_arxiv, mock_hf, mock_pwc, mock_cli, reg):
        mock_arxiv.return_value = [_fake_paper("001"), _fake_paper("002")]
        mock_hf.return_value = [_fake_paper("003", "huggingface")]
        mock_pwc.return_value = [_fake_paper("004", "paperswithcode")]
        mock_cli.return_value = "# Trending Themes\n\n## Theme 1: Test"

        result = run_trending(reg, _CFG)

        assert "report_id" in result
        assert result["paper_count"] == 4

        report = reg.get_discovery_report(result["report_id"])
        assert report["status"] == "completed"
        assert report["paper_count"] == 4
        assert "Trending" in report["content"]
        assert len(report["papers_json"]) == 4

    @patch("paper_tracker.discovery.call_cli")
    @patch("paper_tracker.discovery.fetch_pwc_trending")
    @patch("paper_tracker.discovery.fetch_daily_papers")
    @patch("paper_tracker.discovery.search_broad")
    def test_deduplicates_papers(self, mock_arxiv, mock_hf, mock_pwc, mock_cli, reg):
        """Papers with same arxiv_id from different sources should be deduplicated."""
        mock_arxiv.return_value = [_fake_paper("001")]
        mock_hf.return_value = [_fake_paper("001", "huggingface")]  # same ID
        mock_pwc.return_value = [_fake_paper("002", "paperswithcode")]
        mock_cli.return_value = "Themes content"

        result = run_trending(reg, _CFG)
        assert result["paper_count"] == 2  # 001 deduplicated

    @patch("paper_tracker.discovery.call_cli")
    @patch("paper_tracker.discovery.fetch_pwc_trending")
    @patch("paper_tracker.discovery.fetch_daily_papers")
    @patch("paper_tracker.discovery.search_broad")
    def test_source_stats_tracked(self, mock_arxiv, mock_hf, mock_pwc, mock_cli, reg):
        mock_arxiv.return_value = [_fake_paper("001"), _fake_paper("002")]
        mock_hf.return_value = [_fake_paper("003", "huggingface")]
        mock_pwc.return_value = []
        mock_cli.return_value = "Themes"

        result = run_trending(reg, _CFG)
        report = reg.get_discovery_report(result["report_id"])
        stats = report["source_stats"]
        assert stats.get("arxiv", 0) == 2
        assert stats.get("huggingface", 0) == 1

    @patch("paper_tracker.discovery.call_cli")
    @patch("paper_tracker.discovery.fetch_pwc_trending")
    @patch("paper_tracker.discovery.fetch_daily_papers")
    @patch("paper_tracker.discovery.search_broad")
    def test_llm_failure_still_completes(self, mock_arxiv, mock_hf, mock_pwc, mock_cli, reg):
        """If LLM returns None, report should still complete with fallback content."""
        mock_arxiv.return_value = [_fake_paper("001")]
        mock_hf.return_value = []
        mock_pwc.return_value = []
        mock_cli.return_value = None

        result = run_trending(reg, _CFG)
        report = reg.get_discovery_report(result["report_id"])
        assert report["status"] == "completed"
        assert "No trending themes" in report["content"]

    @patch("paper_tracker.discovery.call_cli")
    @patch("paper_tracker.discovery.fetch_pwc_trending")
    @patch("paper_tracker.discovery.fetch_daily_papers")
    @patch("paper_tracker.discovery.search_broad")
    def test_all_sources_fail_gracefully(self, mock_arxiv, mock_hf, mock_pwc, mock_cli, reg):
        """If all sources return empty, should still produce a report."""
        mock_arxiv.return_value = []
        mock_hf.return_value = []
        mock_pwc.return_value = []
        mock_cli.return_value = "No papers to analyze."

        result = run_trending(reg, _CFG)
        report = reg.get_discovery_report(result["report_id"])
        assert report["status"] == "completed"
        assert report["paper_count"] == 0

    @patch("paper_tracker.discovery.call_cli")
    @patch("paper_tracker.discovery.fetch_pwc_trending")
    @patch("paper_tracker.discovery.fetch_daily_papers")
    @patch("paper_tracker.discovery.search_broad")
    def test_calls_llm_with_opus_timeout(self, mock_arxiv, mock_hf, mock_pwc, mock_cli, reg):
        mock_arxiv.return_value = [_fake_paper("001")]
        mock_hf.return_value = []
        mock_pwc.return_value = []
        mock_cli.return_value = "Themes"

        run_trending(reg, _CFG)
        mock_cli.assert_called_once()
        assert mock_cli.call_args[1]["timeout"] == 360

    @patch("paper_tracker.discovery.fetch_pwc_trending", side_effect=Exception("Network error"))
    @patch("paper_tracker.discovery.fetch_daily_papers")
    @patch("paper_tracker.discovery.search_broad")
    def test_source_exception_fails_report(self, mock_arxiv, mock_hf, mock_pwc, reg):
        """If a source raises an exception, the report should be marked failed."""
        mock_arxiv.return_value = [_fake_paper("001")]
        mock_hf.return_value = []

        result = run_trending(reg, _CFG)
        report = reg.get_discovery_report(result["report_id"])
        assert report["status"] == "failed"


# ---------------------------------------------------------------
# run_math_insights
# ---------------------------------------------------------------

class TestRunMathInsights:
    @patch("paper_tracker.discovery.call_cli")
    @patch("paper_tracker.discovery.search_random_era")
    @patch("paper_tracker.discovery.search_broad")
    def test_happy_path(self, mock_broad, mock_era, mock_cli, reg):
        recent = [_fake_paper(f"2603.{i:05d}") for i in range(30)]
        mock_broad.return_value = recent
        mock_era.return_value = []  # no historical/wildcard papers
        mock_cli.return_value = "# Math Insights\n\nSome insights"

        result = run_math_insights(reg, _CFG)

        assert "report_id" in result
        assert result["paper_count"] == 30

        report = reg.get_discovery_report(result["report_id"])
        assert report["status"] == "completed"
        assert "Math" in report["content"]
        assert report["paper_count"] == 30

    @patch("paper_tracker.discovery.call_cli")
    @patch("paper_tracker.discovery.search_random_era")
    @patch("paper_tracker.discovery.search_broad")
    def test_samples_25_from_large_set(self, mock_broad, mock_era, mock_cli, reg):
        """Should sample ~25 papers when >25 are available."""
        recent = [_fake_paper(f"2603.{i:05d}") for i in range(100)]
        historical = [_fake_paper(f"9901.{i:05d}") for i in range(30)]
        wildcard = [_fake_paper(f"0501.{i:05d}") for i in range(15)]
        mock_broad.return_value = recent
        mock_era.side_effect = [historical, wildcard]
        mock_cli.return_value = "Insights"

        run_math_insights(reg, _CFG)

        # The prompt should only contain ~25 papers, not all 145
        prompt = mock_cli.call_args[0][0]
        # Count paper references in the prompt (each has [ID])
        import re
        paper_refs = len(re.findall(r'\[\d{4}\.\d{5}\]', prompt))
        assert paper_refs == 25

    @patch("paper_tracker.discovery.call_cli")
    @patch("paper_tracker.discovery.search_random_era")
    @patch("paper_tracker.discovery.search_broad")
    def test_fewer_than_25_uses_all(self, mock_broad, mock_era, mock_cli, reg):
        """With <25 papers, should use all without sampling."""
        papers = [_fake_paper(f"2603.{i:05d}") for i in range(10)]
        mock_broad.return_value = papers
        mock_era.return_value = []
        mock_cli.return_value = "Insights"

        run_math_insights(reg, _CFG)

        prompt = mock_cli.call_args[0][0]
        paper_count_in_prompt = prompt.count("[2603.")
        assert paper_count_in_prompt == 10

    @patch("paper_tracker.discovery.call_cli")
    @patch("paper_tracker.discovery.search_random_era")
    @patch("paper_tracker.discovery.search_broad")
    def test_uses_math_categories(self, mock_broad, mock_era, mock_cli, reg):
        """Should search core math/stats categories for recent papers."""
        mock_broad.return_value = [_fake_paper("001")]
        mock_era.return_value = []
        mock_cli.return_value = "Insights"

        run_math_insights(reg, _CFG)

        categories = mock_broad.call_args[0][0]
        assert "math.ST" in categories
        assert "stat.ML" in categories
        assert "math.OC" in categories

    @patch("paper_tracker.discovery.call_cli")
    @patch("paper_tracker.discovery.search_random_era")
    @patch("paper_tracker.discovery.search_broad")
    def test_14_day_lookback(self, mock_broad, mock_era, mock_cli, reg):
        mock_broad.return_value = []
        mock_era.return_value = []
        mock_cli.return_value = "No papers"

        run_math_insights(reg, _CFG)

        assert mock_broad.call_args[1]["lookback_days"] == 14

    @patch("paper_tracker.discovery.call_cli")
    @patch("paper_tracker.discovery.search_random_era")
    @patch("paper_tracker.discovery.search_broad")
    def test_llm_failure(self, mock_broad, mock_era, mock_cli, reg):
        mock_broad.return_value = [_fake_paper("001")]
        mock_era.return_value = []
        mock_cli.return_value = None

        result = run_math_insights(reg, _CFG)
        report = reg.get_discovery_report(result["report_id"])
        assert report["status"] == "completed"
        assert "No math insights" in report["content"]

    @patch("paper_tracker.discovery.search_random_era", return_value=[])
    @patch("paper_tracker.discovery.search_broad", side_effect=Exception("arXiv down"))
    def test_arxiv_exception(self, mock_broad, mock_era, reg):
        result = run_math_insights(reg, _CFG)
        report = reg.get_discovery_report(result["report_id"])
        assert report["status"] == "failed"
        assert "error" in result
