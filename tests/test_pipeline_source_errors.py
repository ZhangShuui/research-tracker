"""run_pipeline should surface source failures instead of silently reporting 0.

These tests exercise the early-return path (no new papers/repos), so they never
reach summarize/report/insights and need no network or LLM — only the two
source ``search`` functions are mocked.
"""

from __future__ import annotations

from unittest.mock import patch

import httpx

from paper_tracker.main import run_pipeline


def _cfg() -> dict:
    return {"search": {
        "arxiv_keywords": ["world model"],
        "arxiv_categories": ["cs.CV"],
        "prefilter_enabled": False,
    }}


@patch("paper_tracker.sources.github.search", return_value=[])
@patch("paper_tracker.sources.arxiv.search", side_effect=httpx.HTTPError("429 rate limited"))
def test_marks_partial_when_a_source_fails(mock_arxiv, mock_github, tmp_path):
    result = run_pipeline(
        topic_cfg=_cfg(),
        session_id="2026-06-01_001",
        topic_id="t",
        topic_name="T",
        data_dir=str(tmp_path),
        session_dir=tmp_path / "sess",
    )
    assert result["status"] == "partial"
    assert "arxiv" in result["source_errors"]
    assert "HTTPError" in result["source_errors"]["arxiv"]
    assert "incomplete" in result["error_message"].lower()
    assert result["paper_count"] == 0


@patch("paper_tracker.sources.github.search", return_value=[])
@patch("paper_tracker.sources.arxiv.search", return_value=[])
def test_completed_when_genuinely_empty(mock_arxiv, mock_github, tmp_path):
    """No errors + no results → a clean 'completed', not 'partial'."""
    result = run_pipeline(
        topic_cfg=_cfg(),
        session_id="2026-06-01_002",
        topic_id="t",
        topic_name="T",
        data_dir=str(tmp_path),
        session_dir=tmp_path / "sess",
    )
    assert result["status"] == "completed"
    assert result["source_errors"] == {}
    assert result["error_message"] == ""
