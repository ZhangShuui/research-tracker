"""Unit tests for brainstorm.py — discovery context integration.

Tests _load_discovery_context and its injection into idea generation.
"""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest

from paper_tracker.brainstorm import (
    _load_discovery_context,
    run_brainstorm,
)
from paper_tracker.registry import Registry


_CFG = {"summarizer": {"claude_path": "claude", "claude_model": "opus"}}


@pytest.fixture()
def reg(tmp_path):
    r = Registry(str(tmp_path))
    yield r
    r.close()


# ---------------------------------------------------------------
# _load_discovery_context
# ---------------------------------------------------------------

class TestLoadDiscoveryContext:
    def test_no_registry_returns_empty(self):
        assert _load_discovery_context(None) == ""

    def test_no_reports_returns_empty(self, reg):
        assert _load_discovery_context(reg) == ""

    def test_loads_trending_context(self, reg):
        dr = reg.create_discovery_report("trending")
        reg.update_discovery_report(dr["id"], {
            "status": "completed",
            "finished_at": "2026-03-05T10:00:00",
            "content": "# Trending\n\nTheme 1: LLM Agents",
        })
        ctx = _load_discovery_context(reg)
        assert "Trending Themes" in ctx
        assert "LLM Agents" in ctx

    def test_loads_math_context(self, reg):
        dr = reg.create_discovery_report("math")
        reg.update_discovery_report(dr["id"], {
            "status": "completed",
            "finished_at": "2026-03-05T10:00:00",
            "content": "# Math\n\nOptimal Transport for ML",
        })
        ctx = _load_discovery_context(reg)
        assert "Math Insights" in ctx
        assert "Optimal Transport" in ctx

    def test_loads_both_trending_and_math(self, reg):
        dr_t = reg.create_discovery_report("trending")
        reg.update_discovery_report(dr_t["id"], {
            "status": "completed",
            "finished_at": "2026-03-05T10:00:00",
            "content": "Trending content here",
        })
        dr_m = reg.create_discovery_report("math")
        reg.update_discovery_report(dr_m["id"], {
            "status": "completed",
            "finished_at": "2026-03-05T11:00:00",
            "content": "Math content here",
        })
        ctx = _load_discovery_context(reg)
        assert "Trending" in ctx
        assert "Math" in ctx
        assert "Discovery Context" in ctx

    def test_truncates_long_content(self, reg):
        dr = reg.create_discovery_report("trending")
        long_content = "A" * 10000
        reg.update_discovery_report(dr["id"], {
            "status": "completed",
            "finished_at": "2026-03-05T10:00:00",
            "content": long_content,
        })
        ctx = _load_discovery_context(reg)
        # Content should be truncated to 3000 chars per report
        assert len(ctx) < 7000  # some overhead from headers

    def test_ignores_running_reports(self, reg):
        reg.create_discovery_report("trending")  # status=running
        ctx = _load_discovery_context(reg)
        assert ctx == ""

    def test_ignores_empty_content(self, reg):
        dr = reg.create_discovery_report("trending")
        reg.update_discovery_report(dr["id"], {
            "status": "completed",
            "finished_at": "2026-03-05T10:00:00",
            "content": "",
        })
        ctx = _load_discovery_context(reg)
        assert ctx == ""


# ---------------------------------------------------------------
# run_brainstorm — discovery context injection
# ---------------------------------------------------------------

class TestBrainstormDiscoveryInjection:
    @patch("paper_tracker.brainstorm.call_cli")
    @patch("paper_tracker.brainstorm.Storage")
    def test_auto_mode_includes_context(self, MockStorage, mock_cli, reg):
        """Auto brainstorm should include discovery context in the prompt."""
        # Set up discovery report
        dr = reg.create_discovery_report("trending")
        reg.update_discovery_report(dr["id"], {
            "status": "completed",
            "finished_at": "2026-03-05T10:00:00",
            "content": "UNIQUE_TRENDING_MARKER_XYZ",
        })

        mock_store = MagicMock()
        mock_store.get_all_arxiv.return_value = ([], 0)
        MockStorage.return_value = mock_store

        mock_cli.return_value = json.dumps([{
            "title": "Test Idea",
            "problem": "A problem",
            "motivation": "Important",
            "method": "A method",
            "experiment_plan": "Run experiments",
            "novelty_score": 7,
            "feasibility_score": 8,
        }])

        run_brainstorm(
            topic_id="test-topic",
            topic_name="Test Topic",
            data_dir="/tmp/test",
            cfg=_CFG,
            mode="auto",
            registry=reg,
        )

        # Check the first call (idea generation)
        prompt = mock_cli.call_args_list[0][0][0]
        assert "UNIQUE_TRENDING_MARKER_XYZ" in prompt
        assert "Discovery Context" in prompt

    @patch("paper_tracker.brainstorm.call_cli")
    @patch("paper_tracker.brainstorm.Storage")
    def test_no_registry_no_context(self, MockStorage, mock_cli):
        """Without registry, brainstorm should still work (no context)."""
        mock_store = MagicMock()
        mock_store.get_all_arxiv.return_value = ([], 0)
        MockStorage.return_value = mock_store

        mock_cli.return_value = json.dumps([{
            "title": "Idea",
            "problem": "P",
            "motivation": "M",
            "method": "Me",
            "experiment_plan": "E",
            "novelty_score": 5,
            "feasibility_score": 5,
        }])

        result = run_brainstorm(
            topic_id="test",
            topic_name="Test",
            data_dir="/tmp/test",
            cfg=_CFG,
            mode="auto",
            registry=None,
        )

        assert len(result["ideas"]) == 1
        prompt = mock_cli.call_args_list[0][0][0]
        # No discovery context markers
        assert "Discovery Context" not in prompt

    @patch("paper_tracker.brainstorm.call_cli")
    @patch("paper_tracker.brainstorm.Storage")
    def test_user_mode_not_affected(self, MockStorage, mock_cli, reg):
        """User mode uses a different prompt — discovery context is only in auto mode."""
        dr = reg.create_discovery_report("trending")
        reg.update_discovery_report(dr["id"], {
            "status": "completed",
            "finished_at": "2026-03-05T10:00:00",
            "content": "Trending stuff",
        })

        mock_store = MagicMock()
        mock_store.get_all_arxiv.return_value = ([], 0)
        MockStorage.return_value = mock_store

        mock_cli.return_value = json.dumps({
            "title": "User Idea Refined",
            "problem": "P",
            "motivation": "M",
            "method": "Me",
            "experiment_plan": "E",
            "novelty_score": 5,
            "feasibility_score": 5,
        })

        run_brainstorm(
            topic_id="test",
            topic_name="Test",
            data_dir="/tmp/test",
            cfg=_CFG,
            mode="user",
            user_idea="My great idea",
            registry=reg,
        )

        # User mode prompt should NOT contain discovery context
        prompt = mock_cli.call_args_list[0][0][0]
        assert "Discovery Context" not in prompt
