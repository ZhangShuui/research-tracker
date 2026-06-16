"""Tests for insights.suggest_cross_domain (bridge proposal + arXiv search)."""

from __future__ import annotations

import json
from unittest.mock import patch

from paper_tracker import insights


def _sel(arxiv_id: str, title: str = "Sel", **kw) -> dict:
    return {"arxiv_id": arxiv_id, "paper_id": arxiv_id, "title": title, **kw}


def _hit(arxiv_id: str, title: str = "Hit") -> dict:
    return {
        "arxiv_id": arxiv_id, "title": title, "authors": "Auth",
        "abstract": "abs", "url": f"https://arxiv.org/abs/{arxiv_id}",
        "published": "2020-05-05T00:00:00Z",
    }


_BRIDGES = json.dumps([
    {"query": "optimal transport", "domain": "math.OC", "rationale": "geometry of distributions"},
    {"query": "category theory ml", "domain": "math.CT", "rationale": "compositional structure"},
])


class TestSuggestCrossDomain:
    def test_empty_selection_returns_empty(self):
        assert insights.suggest_cross_domain([], "T", {}) == []

    @patch("paper_tracker.insights.call_cli", return_value="not json at all")
    def test_unparseable_llm_returns_empty(self, _cli):
        assert insights.suggest_cross_domain([_sel("1")], "T", {}) == []

    @patch("paper_tracker.sources.arxiv.search_by_query")
    @patch("paper_tracker.insights.call_cli", return_value=_BRIDGES)
    def test_builds_candidates_with_domain_and_rationale(self, _cli, mock_search):
        mock_search.side_effect = [[_hit("2401.001")], [_hit("2401.002")]]
        out = insights.suggest_cross_domain([_sel("9")], "T", {})
        assert [c["arxiv_id"] for c in out] == ["2401.001", "2401.002"]
        assert out[0]["domain"] == "math.OC"
        assert out[0]["rationale"] == "geometry of distributions"
        assert out[0]["published"] == "2020-05-05"  # normalized

    @patch("paper_tracker.sources.arxiv.search_by_query")
    @patch("paper_tracker.insights.call_cli", return_value=_BRIDGES)
    def test_dedups_and_excludes_selected(self, _cli, mock_search):
        # both bridges return an overlapping hit + the already-selected paper
        mock_search.side_effect = [
            [_hit("dup"), _hit("sel-1")],
            [_hit("dup"), _hit("fresh")],
        ]
        out = insights.suggest_cross_domain([_sel("sel-1")], "T", {})
        ids = [c["arxiv_id"] for c in out]
        assert ids == ["dup", "fresh"]   # dup once, selected excluded

    @patch("paper_tracker.sources.arxiv.search_by_query")
    @patch("paper_tracker.insights.call_cli", return_value=_BRIDGES)
    def test_caps_at_max_candidates(self, _cli, mock_search):
        mock_search.side_effect = [
            [_hit(f"a{i}") for i in range(10)],
            [_hit(f"b{i}") for i in range(10)],
        ]
        out = insights.suggest_cross_domain([_sel("9")], "T", {}, max_candidates=3)
        assert len(out) == 3

    @patch("paper_tracker.sources.arxiv.search_by_query")
    @patch("paper_tracker.insights.call_cli", return_value=_BRIDGES)
    def test_failed_bridge_search_is_skipped(self, _cli, mock_search):
        mock_search.side_effect = [Exception("boom"), [_hit("ok")]]
        out = insights.suggest_cross_domain([_sel("9")], "T", {})
        assert [c["arxiv_id"] for c in out] == ["ok"]
