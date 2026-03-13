"""Unit tests for summarizer.py — filter_papers_by_quality & refilter_papers.

LLM calls are mocked to isolate logic from external dependencies.
"""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest

from paper_tracker.summarizer import (
    filter_papers_by_quality,
    refilter_papers,
    _parse_json_array,
)


def _paper(arxiv_id: str, **kw) -> dict:
    base = {
        "arxiv_id": arxiv_id,
        "title": f"Paper {arxiv_id}",
        "abstract": "Abstract text " * 50,
        "method": "Some method",
        "contribution": "Some contribution",
    }
    base.update(kw)
    return base


_CFG = {"summarizer": {"claude_path": "claude", "claude_model": "opus"}}


# ---------------------------------------------------------------
# _parse_json_array
# ---------------------------------------------------------------

class TestParseJsonArray:
    def test_plain_array(self):
        assert _parse_json_array('[{"id": "a", "quality": 3}]') == [{"id": "a", "quality": 3}]

    def test_markdown_fenced(self):
        raw = '```json\n[{"id": "a"}]\n```'
        assert _parse_json_array(raw) == [{"id": "a"}]

    def test_text_before_array(self):
        raw = 'Here is the result:\n[{"id": "a"}]'
        assert _parse_json_array(raw) == [{"id": "a"}]

    def test_invalid_json(self):
        assert _parse_json_array("not json at all") == []

    def test_empty_array(self):
        assert _parse_json_array("[]") == []

    def test_nested_brackets(self):
        raw = '[{"id": "a", "list": [1, 2]}]'
        parsed = _parse_json_array(raw)
        assert len(parsed) == 1
        assert parsed[0]["list"] == [1, 2]


# ---------------------------------------------------------------
# filter_papers_by_quality
# ---------------------------------------------------------------

class TestFilterPapersByQuality:
    @patch("paper_tracker.summarizer.llm_call_cli" if False else "paper_tracker.llm.call_cli")
    def test_filters_low_quality(self, mock_cli):
        """Papers scored below min_quality should be removed."""
        papers = [_paper("001"), _paper("002"), _paper("003")]

        mock_cli.return_value = json.dumps([
            {"id": "001", "quality": 5, "rationale": "excellent"},
            {"id": "002", "quality": 1, "rationale": "irrelevant"},
            {"id": "003", "quality": 3, "rationale": "okay"},
        ])

        result = filter_papers_by_quality(papers, _CFG, "test topic", min_quality=3)

        assert len(result) == 2
        ids = {p["arxiv_id"] for p in result}
        assert ids == {"001", "003"}
        assert result[0]["quality_score"] == 5
        assert result[1]["quality_score"] == 3

    @patch("paper_tracker.llm.call_cli")
    def test_unscored_papers_get_default_3(self, mock_cli):
        """If LLM doesn't return a score for a paper, default to 3 (benefit of doubt)."""
        papers = [_paper("001"), _paper("002")]

        # Only return score for one paper
        mock_cli.return_value = json.dumps([
            {"id": "001", "quality": 5, "rationale": "great"},
        ])

        result = filter_papers_by_quality(papers, _CFG, "test topic", min_quality=3)
        assert len(result) == 2  # both kept
        p002 = [p for p in result if p["arxiv_id"] == "002"][0]
        assert p002["quality_score"] == 3

    @patch("paper_tracker.llm.call_cli")
    def test_cli_failure_keeps_all(self, mock_cli):
        """If CLI fails entirely, all papers get default score 3 and are kept."""
        papers = [_paper("001"), _paper("002")]
        mock_cli.return_value = None

        result = filter_papers_by_quality(papers, _CFG, "test topic", min_quality=3)
        assert len(result) == 2
        assert all(p["quality_score"] == 3 for p in result)

    @patch("paper_tracker.llm.call_cli")
    def test_empty_input(self, mock_cli):
        result = filter_papers_by_quality([], _CFG, "test topic")
        assert result == []
        mock_cli.assert_not_called()

    @patch("paper_tracker.llm.call_cli")
    def test_quality_clamped_to_1_5(self, mock_cli):
        """Scores outside 1-5 range should be clamped."""
        papers = [_paper("001"), _paper("002")]
        mock_cli.return_value = json.dumps([
            {"id": "001", "quality": 10, "rationale": "extreme"},
            {"id": "002", "quality": -1, "rationale": "negative"},
        ])

        result = filter_papers_by_quality(papers, _CFG, "test topic", min_quality=1)
        scores = {p["arxiv_id"]: p["quality_score"] for p in result}
        assert scores["001"] == 5
        assert scores["002"] == 1

    @patch("paper_tracker.llm.call_cli")
    def test_keywords_passed_to_prompt(self, mock_cli):
        """Keywords should appear in the prompt sent to CLI."""
        papers = [_paper("001")]
        mock_cli.return_value = json.dumps([{"id": "001", "quality": 4, "rationale": "ok"}])

        filter_papers_by_quality(
            papers, _CFG, "test topic",
            keywords=["diffusion", "world model"],
        )

        prompt = mock_cli.call_args[0][0]
        assert "diffusion" in prompt
        assert "world model" in prompt

    @patch("paper_tracker.llm.call_cli")
    def test_uses_opus_model(self, mock_cli):
        """Should call LLM with opus model."""
        papers = [_paper("001")]
        mock_cli.return_value = json.dumps([{"id": "001", "quality": 4, "rationale": "ok"}])

        filter_papers_by_quality(papers, _CFG, "test topic")
        mock_cli.assert_called_once()
        assert mock_cli.call_args[1].get("model") == "opus"

    @patch("paper_tracker.llm.call_cli")
    def test_batching_10_per_batch(self, mock_cli):
        """Should process 10 papers per batch."""
        papers = [_paper(f"{i:03d}") for i in range(25)]

        def _fake_response(prompt, cfg, **kw):
            # Extract IDs from prompt and return scores
            results = []
            for p in papers:
                if p["arxiv_id"] in prompt:
                    results.append({"id": p["arxiv_id"], "quality": 4, "rationale": "ok"})
            return json.dumps(results)

        mock_cli.side_effect = _fake_response

        result = filter_papers_by_quality(papers, _CFG, "test topic")
        assert mock_cli.call_count == 3  # 10 + 10 + 5
        assert len(result) == 25


# ---------------------------------------------------------------
# refilter_papers
# ---------------------------------------------------------------

class TestRefilterPapers:
    @patch("paper_tracker.llm.call_cli")
    def test_updates_scores(self, mock_cli):
        """refilter should update quality_score on each paper."""
        papers = [_paper("001", quality_score=3), _paper("002", quality_score=4)]

        mock_cli.return_value = json.dumps([
            {"id": "001", "quality": 1, "rationale": "not relevant"},
            {"id": "002", "quality": 5, "rationale": "excellent"},
        ])

        result = refilter_papers(papers, _CFG, "test topic")
        assert len(result) == 2  # refilter does NOT filter
        scores = {p["arxiv_id"]: p["quality_score"] for p in result}
        assert scores["001"] == 1
        assert scores["002"] == 5

    @patch("paper_tracker.llm.call_cli")
    def test_does_not_filter(self, mock_cli):
        """refilter should return ALL papers, even low-scored ones."""
        papers = [_paper("001"), _paper("002")]
        mock_cli.return_value = json.dumps([
            {"id": "001", "quality": 1, "rationale": "bad"},
            {"id": "002", "quality": 1, "rationale": "bad"},
        ])

        result = refilter_papers(papers, _CFG, "test topic")
        assert len(result) == 2

    @patch("paper_tracker.llm.call_cli")
    def test_custom_instructions_in_prompt(self, mock_cli):
        """Custom instructions should appear in the prompt."""
        papers = [_paper("001")]
        mock_cli.return_value = json.dumps([{"id": "001", "quality": 2, "rationale": "nlp"}])

        refilter_papers(
            papers, _CFG, "test topic",
            custom_instructions="Remove all NLP papers",
        )

        prompt = mock_cli.call_args[0][0]
        assert "Remove all NLP papers" in prompt
        assert "CUSTOM INSTRUCTIONS" in prompt

    @patch("paper_tracker.llm.call_cli")
    def test_no_custom_instructions(self, mock_cli):
        """Without custom instructions, CUSTOM INSTRUCTIONS section should not appear."""
        papers = [_paper("001")]
        mock_cli.return_value = json.dumps([{"id": "001", "quality": 4, "rationale": "ok"}])

        refilter_papers(papers, _CFG, "test topic")

        prompt = mock_cli.call_args[0][0]
        assert "CUSTOM INSTRUCTIONS" not in prompt

    @patch("paper_tracker.llm.call_cli")
    def test_cli_failure_preserves_scores(self, mock_cli):
        """If CLI fails, existing scores should be preserved."""
        papers = [_paper("001", quality_score=4)]
        mock_cli.return_value = None

        result = refilter_papers(papers, _CFG, "test topic")
        assert result[0]["quality_score"] == 4

    @patch("paper_tracker.llm.call_cli")
    def test_empty_input(self, mock_cli):
        result = refilter_papers([], _CFG, "test topic")
        assert result == []
        mock_cli.assert_not_called()

    @patch("paper_tracker.llm.call_cli")
    def test_uses_opus_model(self, mock_cli):
        papers = [_paper("001")]
        mock_cli.return_value = json.dumps([{"id": "001", "quality": 3, "rationale": "ok"}])

        refilter_papers(papers, _CFG, "test topic")
        assert mock_cli.call_args[1].get("model") == "opus"

    @patch("paper_tracker.llm.call_cli")
    def test_partial_response_preserves_unscored(self, mock_cli):
        """If LLM only returns scores for some papers, others keep existing score."""
        papers = [_paper("001", quality_score=4), _paper("002", quality_score=2)]
        mock_cli.return_value = json.dumps([
            {"id": "001", "quality": 5, "rationale": "better"},
        ])

        result = refilter_papers(papers, _CFG, "test topic")
        scores = {p["arxiv_id"]: p["quality_score"] for p in result}
        assert scores["001"] == 5
        assert scores["002"] == 2  # unchanged
