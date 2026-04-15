"""Unit tests for deep_read.py — generate_deep_read and generate_deep_read_qa.

LLM calls are mocked so no actual CLI invocations happen.
"""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest

from paper_tracker.deep_read import (
    _build_paper_context,
    _extract_repo_urls,
    generate_deep_read,
    generate_deep_read_qa,
)


_FAKE_PAPER = {
    "arxiv_id": "2501.00001",
    "title": "Test Paper",
    "authors": "Alice, Bob",
    "abstract": "An abstract about transformers.",
    "url": "https://arxiv.org/abs/2501.00001",
    "published": "2025-01-01",
    "summary": "Summary of the paper.",
    "key_insight": "Attention is all you need.",
    "method": "Transformer architecture.",
    "contribution": "Novel contribution.",
    "math_concepts": ["self-attention", "softmax"],
    "venue": "NeurIPS",
    "cited_works": ["Vaswani 2017"],
    "quality_score": 4,
}

_FAKE_CFG = {
    "summarizer": {"claude_path": "claude", "claude_model": "opus"},
}


# ---------------------------------------------------------------
# _build_paper_context
# ---------------------------------------------------------------

class TestBuildPaperContext:
    def test_basic_fields(self):
        ctx = _build_paper_context(_FAKE_PAPER)
        assert "Test Paper" in ctx
        assert "Alice, Bob" in ctx
        assert "NeurIPS" in ctx
        assert "An abstract about transformers" in ctx
        assert "self-attention" in ctx
        assert "Vaswani 2017" in ctx

    def test_minimal_paper(self):
        ctx = _build_paper_context({"title": "T", "authors": "A", "published": "2025"})
        assert "T" in ctx
        assert "A" in ctx
        assert "Venue" not in ctx  # no venue

    def test_math_concepts_as_string(self):
        """If math_concepts is a string (corrupted data), should not crash."""
        ctx = _build_paper_context({
            "title": "T", "authors": "A", "published": "2025",
            "math_concepts": "not a list",
        })
        assert "Mathematical Concepts" not in ctx  # skips non-list


# ---------------------------------------------------------------
# generate_deep_read (with mocked LLM)
# ---------------------------------------------------------------

class TestGenerateDeepRead:
    @patch("paper_tracker.deep_read.Storage")
    @patch("paper_tracker.deep_read.call_cli")
    def test_five_stages(self, mock_call_cli, MockStorage):
        # Setup
        mock_store = MagicMock()
        mock_store.get_arxiv.return_value = _FAKE_PAPER
        MockStorage.return_value = mock_store

        responses = [
            "Motivation analysis text",
            "Method analysis text",
            "Experiment analysis text",
            "Contribution analysis text",
            json.dumps({
                "key_innovations": ["Fast training"],
                "strengths": ["Simple"],
                "weaknesses": ["Limited"],
                "field_comparison": ["Better than X"],
                "inspiration_points": ["Could extend to Y"],
            }),
        ]
        mock_call_cli.side_effect = responses

        sections_collected = []
        def on_section(name, content):
            sections_collected.append(name)

        result = generate_deep_read(
            topic_id="test-topic",
            topic_name="Test",
            data_dir="/tmp/data",
            cfg=_FAKE_CFG,
            paper_id="2501.00001",
            on_section_done=on_section,
        )

        assert mock_call_cli.call_count == 5
        assert result["motivation_analysis"] == "Motivation analysis text"
        assert result["method_analysis"] == "Method analysis text"
        assert result["experiment_analysis"] == "Experiment analysis text"
        assert result["contribution_analysis"] == "Contribution analysis text"

        card = result["summary_card_json"]
        assert isinstance(card, dict)
        assert card["key_innovations"] == ["Fast training"]

        assert sections_collected == [
            "motivation_analysis",
            "method_analysis",
            "experiment_analysis",
            "contribution_analysis",
            "summary_card_json",
            "code_review",
        ]
        # No repos in paper — code_review should be empty, no extra LLM call
        assert result["code_review"] == ""
        assert result["repo_urls"] == []

        mock_store.close.assert_called_once()

    @patch("paper_tracker.deep_read.Storage")
    @patch("paper_tracker.deep_read.call_cli")
    def test_paper_not_found_raises(self, mock_call_cli, MockStorage):
        mock_store = MagicMock()
        mock_store.get_arxiv.return_value = None
        MockStorage.return_value = mock_store

        with pytest.raises(ValueError, match="Paper not found"):
            generate_deep_read(
                topic_id="test-topic",
                topic_name="Test",
                data_dir="/tmp/data",
                cfg=_FAKE_CFG,
                paper_id="nonexistent",
            )

    @patch("paper_tracker.deep_read.Storage")
    @patch("paper_tracker.deep_read.call_cli")
    def test_llm_returns_none(self, mock_call_cli, MockStorage):
        """If LLM returns None for a stage, it should be stored as empty string."""
        mock_store = MagicMock()
        mock_store.get_arxiv.return_value = _FAKE_PAPER
        MockStorage.return_value = mock_store
        mock_call_cli.return_value = None

        result = generate_deep_read(
            topic_id="test-topic",
            topic_name="Test",
            data_dir="/tmp/data",
            cfg=_FAKE_CFG,
            paper_id="2501.00001",
        )
        assert result["motivation_analysis"] == ""
        assert result["method_analysis"] == ""

    @patch("paper_tracker.deep_read.Storage")
    @patch("paper_tracker.deep_read.call_cli")
    def test_json_parse_failure_fallback(self, mock_call_cli, MockStorage):
        """If summary card JSON is invalid, should fall back gracefully."""
        mock_store = MagicMock()
        mock_store.get_arxiv.return_value = _FAKE_PAPER
        MockStorage.return_value = mock_store

        mock_call_cli.side_effect = [
            "M", "Me", "E", "C",
            "not valid json at all",
        ]

        result = generate_deep_read(
            topic_id="test-topic",
            topic_name="Test",
            data_dir="/tmp/data",
            cfg=_FAKE_CFG,
            paper_id="2501.00001",
        )

        card = result["summary_card_json"]
        assert isinstance(card, dict)
        assert card["key_innovations"] == []
        assert "_raw" in card  # raw text preserved

    @patch("paper_tracker.deep_read.Storage")
    @patch("paper_tracker.deep_read.call_cli")
    def test_json_with_markdown_fences(self, mock_call_cli, MockStorage):
        """LLM sometimes wraps JSON in ```json ... ``` — should be cleaned."""
        mock_store = MagicMock()
        mock_store.get_arxiv.return_value = _FAKE_PAPER
        MockStorage.return_value = mock_store

        fenced_json = '```json\n{"key_innovations": ["X"], "strengths": [], "weaknesses": [], "field_comparison": [], "inspiration_points": []}\n```'
        mock_call_cli.side_effect = ["M", "Me", "E", "C", fenced_json]

        result = generate_deep_read(
            topic_id="test-topic",
            topic_name="Test",
            data_dir="/tmp/data",
            cfg=_FAKE_CFG,
            paper_id="2501.00001",
        )
        assert result["summary_card_json"]["key_innovations"] == ["X"]

    @patch("paper_tracker.deep_read.Storage")
    @patch("paper_tracker.deep_read.call_cli")
    def test_no_callback(self, mock_call_cli, MockStorage):
        """Should work fine without on_section_done callback."""
        mock_store = MagicMock()
        mock_store.get_arxiv.return_value = _FAKE_PAPER
        MockStorage.return_value = mock_store
        mock_call_cli.return_value = "text"

        result = generate_deep_read(
            topic_id="test-topic",
            topic_name="Test",
            data_dir="/tmp/data",
            cfg=_FAKE_CFG,
            paper_id="2501.00001",
            on_section_done=None,
        )
        assert result["motivation_analysis"] == "text"

    @patch("paper_tracker.deep_read.Storage")
    @patch("paper_tracker.deep_read.call_cli")
    def test_uses_opus_model(self, mock_call_cli, MockStorage):
        """All stages should request opus model."""
        mock_store = MagicMock()
        mock_store.get_arxiv.return_value = _FAKE_PAPER
        MockStorage.return_value = mock_store
        mock_call_cli.return_value = "text"

        generate_deep_read(
            topic_id="test-topic",
            topic_name="Test",
            data_dir="/tmp/data",
            cfg=_FAKE_CFG,
            paper_id="2501.00001",
        )

        for call in mock_call_cli.call_args_list:
            assert call.kwargs.get("model") == "opus"


# ---------------------------------------------------------------
# generate_deep_read_qa (with mocked LLM)
# ---------------------------------------------------------------

class TestGenerateDeepReadQA:
    @patch("paper_tracker.deep_read.call_cli")
    def test_basic_qa(self, mock_call_cli):
        mock_call_cli.return_value = "The method uses a transformer."

        result = generate_deep_read_qa(
            paper=_FAKE_PAPER,
            existing_analysis={
                "motivation_analysis": "M",
                "method_analysis": "Me",
                "experiment_analysis": "E",
                "contribution_analysis": "C",
            },
            prior_messages=[],
            user_question="What method does this use?",
            cfg=_FAKE_CFG,
        )

        assert result["content"] == "The method uses a transformer."
        mock_call_cli.assert_called_once()

        # Check the prompt includes paper context and analysis
        prompt = mock_call_cli.call_args[0][0]
        assert "Test Paper" in prompt
        assert "Motivation Analysis" in prompt
        assert "What method does this use?" in prompt

    @patch("paper_tracker.deep_read.call_cli")
    def test_includes_history(self, mock_call_cli):
        mock_call_cli.return_value = "Yes it does."

        generate_deep_read_qa(
            paper=_FAKE_PAPER,
            existing_analysis={"motivation_analysis": "M"},
            prior_messages=[
                {"role": "user", "content": "Does it use attention?"},
                {"role": "assistant", "content": "Yes."},
            ],
            user_question="Can you elaborate?",
            cfg=_FAKE_CFG,
        )

        prompt = mock_call_cli.call_args[0][0]
        assert "Does it use attention?" in prompt
        assert "Can you elaborate?" in prompt

    @patch("paper_tracker.deep_read.call_cli")
    def test_llm_failure_returns_fallback(self, mock_call_cli):
        mock_call_cli.return_value = None

        result = generate_deep_read_qa(
            paper=_FAKE_PAPER,
            existing_analysis={},
            prior_messages=[],
            user_question="Q?",
            cfg=_FAKE_CFG,
        )
        assert "unable to generate" in result["content"].lower()

    @patch("paper_tracker.deep_read.call_cli")
    def test_uses_opus_model(self, mock_call_cli):
        mock_call_cli.return_value = "Answer"

        generate_deep_read_qa(
            paper=_FAKE_PAPER,
            existing_analysis={},
            prior_messages=[],
            user_question="Q?",
            cfg=_FAKE_CFG,
        )

        assert mock_call_cli.call_args.kwargs.get("model") == "opus"


# ---------------------------------------------------------------
# _extract_repo_urls
# ---------------------------------------------------------------

class TestExtractRepoUrls:
    def test_no_urls(self):
        assert _extract_repo_urls(_FAKE_PAPER) == []

    def test_url_in_abstract(self):
        paper = {**_FAKE_PAPER, "abstract": "Code at https://github.com/alice/cool-repo"}
        urls = _extract_repo_urls(paper)
        assert urls == ["https://github.com/alice/cool-repo"]

    def test_url_in_summary(self):
        paper = {**_FAKE_PAPER, "summary": "See https://github.com/org/project for code."}
        assert _extract_repo_urls(paper) == ["https://github.com/org/project"]

    def test_multiple_dedup(self):
        paper = {
            **_FAKE_PAPER,
            "abstract": "https://github.com/a/b and https://github.com/c/d",
            "summary": "Also https://github.com/a/b again",
        }
        urls = _extract_repo_urls(paper)
        assert urls == ["https://github.com/a/b", "https://github.com/c/d"]

    def test_strips_trailing_git(self):
        paper = {**_FAKE_PAPER, "abstract": "https://github.com/a/b.git"}
        assert _extract_repo_urls(paper) == ["https://github.com/a/b"]

    def test_url_field(self):
        paper = {**_FAKE_PAPER, "url": "https://github.com/org/paper-code"}
        assert _extract_repo_urls(paper) == ["https://github.com/org/paper-code"]


# ---------------------------------------------------------------
# Code review stage (with mocked LLM + network)
# ---------------------------------------------------------------

_FAKE_PAPER_WITH_REPO = {
    **_FAKE_PAPER,
    "abstract": "We release code at https://github.com/alice/transformer-impl",
}


class TestCodeReviewStage:
    @patch("paper_tracker.deep_read._build_repo_context")
    @patch("paper_tracker.deep_read._clone_repo")
    @patch("paper_tracker.deep_read.Storage")
    @patch("paper_tracker.deep_read.call_cli")
    def test_code_review_generated_when_repo_found(self, mock_call_cli, MockStorage, mock_clone, mock_build):
        mock_store = MagicMock()
        mock_store.get_arxiv.return_value = _FAKE_PAPER_WITH_REPO
        MockStorage.return_value = mock_store
        mock_clone.return_value = "/tmp/d/t/repos/alice/transformer-impl"
        mock_build.return_value = "## Repo context here"

        # 5 stages + 1 code review = 6 LLM calls
        mock_call_cli.side_effect = ["M", "Me", "E", "C", '{"key_innovations":[],"strengths":[],"weaknesses":[],"field_comparison":[],"inspiration_points":[]}', "Code review text"]

        result = generate_deep_read(
            topic_id="t", topic_name="T", data_dir="/tmp/d",
            cfg=_FAKE_CFG, paper_id="2501.00001",
        )

        assert mock_call_cli.call_count == 6
        assert result["code_review"] == "Code review text"
        assert result["repo_urls"] == ["https://github.com/alice/transformer-impl"]
        assert result["repo_local_paths"] == ["/tmp/d/t/repos/alice/transformer-impl"]
        mock_clone.assert_called_once()
        mock_build.assert_called_once_with("/tmp/d/t/repos/alice/transformer-impl")

    @patch("paper_tracker.deep_read._clone_repo")
    @patch("paper_tracker.deep_read.Storage")
    @patch("paper_tracker.deep_read.call_cli")
    def test_code_review_skipped_when_no_repos(self, mock_call_cli, MockStorage, mock_clone):
        mock_store = MagicMock()
        mock_store.get_arxiv.return_value = _FAKE_PAPER  # no GitHub URL
        MockStorage.return_value = mock_store
        mock_call_cli.return_value = "text"

        result = generate_deep_read(
            topic_id="t", topic_name="T", data_dir="/tmp/d",
            cfg=_FAKE_CFG, paper_id="2501.00001",
        )

        assert result["code_review"] == ""
        assert result["repo_urls"] == []
        assert result["repo_local_paths"] == []
        mock_clone.assert_not_called()
        assert mock_call_cli.call_count == 5  # no code review LLM call

    @patch("paper_tracker.deep_read._build_repo_context")
    @patch("paper_tracker.deep_read._clone_repo")
    @patch("paper_tracker.deep_read.Storage")
    @patch("paper_tracker.deep_read.call_cli")
    def test_code_review_callback_fires(self, mock_call_cli, MockStorage, mock_clone, mock_build):
        mock_store = MagicMock()
        mock_store.get_arxiv.return_value = _FAKE_PAPER_WITH_REPO
        MockStorage.return_value = mock_store
        mock_clone.return_value = "/tmp/d/t/repos/alice/transformer-impl"
        mock_build.return_value = "repo ctx"
        mock_call_cli.side_effect = ["M", "Me", "E", "C", '{}', "CR"]

        sections = []
        result = generate_deep_read(
            topic_id="t", topic_name="T", data_dir="/tmp/d",
            cfg=_FAKE_CFG, paper_id="2501.00001",
            on_section_done=lambda name, _: sections.append(name),
        )

        assert "code_review" in sections
        assert sections[-1] == "code_review"

    @patch("paper_tracker.deep_read._build_repo_context")
    @patch("paper_tracker.deep_read._clone_repo")
    @patch("paper_tracker.deep_read.Storage")
    @patch("paper_tracker.deep_read.call_cli")
    def test_caps_at_three_repos(self, mock_call_cli, MockStorage, mock_clone, mock_build):
        paper = {
            **_FAKE_PAPER,
            "abstract": " ".join(f"https://github.com/org/repo{i}" for i in range(5)),
        }
        mock_store = MagicMock()
        mock_store.get_arxiv.return_value = paper
        MockStorage.return_value = mock_store
        mock_clone.side_effect = lambda url, dest, **kw: f"/tmp/cloned/{url.split('/')[-1]}"
        mock_build.return_value = "ctx"
        mock_call_cli.return_value = "text"

        result = generate_deep_read(
            topic_id="t", topic_name="T", data_dir="/tmp/d",
            cfg=_FAKE_CFG, paper_id="2501.00001",
        )

        assert len(result["repo_urls"]) == 5  # all detected
        assert mock_clone.call_count == 3  # but only 3 cloned

    @patch("paper_tracker.deep_read._build_repo_context")
    @patch("paper_tracker.deep_read._clone_repo")
    @patch("paper_tracker.deep_read.Storage")
    @patch("paper_tracker.deep_read.call_cli")
    def test_clone_failure_skips_code_review(self, mock_call_cli, MockStorage, mock_clone, mock_build):
        """If clone fails, code_review should be empty (no LLM call)."""
        mock_store = MagicMock()
        mock_store.get_arxiv.return_value = _FAKE_PAPER_WITH_REPO
        MockStorage.return_value = mock_store
        mock_clone.return_value = None  # clone failed
        mock_call_cli.return_value = "text"

        result = generate_deep_read(
            topic_id="t", topic_name="T", data_dir="/tmp/d",
            cfg=_FAKE_CFG, paper_id="2501.00001",
        )

        assert result["code_review"] == ""
        assert result["repo_local_paths"] == []
        mock_build.assert_not_called()
        assert mock_call_cli.call_count == 5  # no code review LLM call
