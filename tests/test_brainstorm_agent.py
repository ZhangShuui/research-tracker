"""Tests for brainstorm_agent (agentic idea generation), the manifest round-trip,
and the brainstorm Stage-1 dispatcher. The Agent SDK loop is mocked — no real
claude calls."""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

from paper_tracker import brainstorm_agent, brainstorm, insights


# ---------------------------------------------------------------
# Tool backends
# ---------------------------------------------------------------

class TestToolBackends:
    def test_format_card(self):
        p = {"index": "P1", "title": "OT", "venue": "math.OC", "summary": "studies transport",
             "math_concepts": ["Wasserstein"], "abstract": "abc", "url": "u"}
        t = brainstorm_agent._format_card(p)
        assert "[P1] OT" in t and "Wasserstein" in t and "studies transport" in t

    def test_search_arxiv_formats(self):
        with patch("paper_tracker.brainstorm_agent.arxiv.search_by_query",
                   return_value=[{"title": "T", "arxiv_id": "2401.00001", "abstract": "abs"}]):
            out = brainstorm_agent._search_arxiv("query")
        assert "2401.00001" in out and "T" in out

    def test_search_arxiv_handles_failure(self):
        with patch("paper_tracker.brainstorm_agent.arxiv.search_by_query",
                   side_effect=Exception("boom")):
            out = brainstorm_agent._search_arxiv("q")
        assert "failed" in out.lower()

    def test_search_local_ranks_library_and_kb(self):
        lib = [{"title": "deep transport nets", "abstract": "x", "arxiv_id": "L1"}]
        kb = [{"title": "optimal transport theory", "summary": "y", "arxiv_id": "K1"}]

        def fake_storage(data_dir, topic_id):
            m = MagicMock()
            m.get_all_arxiv.return_value = (kb if topic_id == "__crossdomain__" else lib, 1)
            return m

        with patch("paper_tracker.brainstorm_agent.Storage", side_effect=fake_storage):
            out = brainstorm_agent._search_local("transport", "/d", "t1")
        assert "L1" in out and "K1" in out          # both sources searched
        assert "library" in out and "knowledge_base" in out

    def test_fetch_fulltext_falls_back_to_abstract(self):
        # non-arXiv, no url -> no fetch, returns abstract
        out = brainstorm_agent._fetch_fulltext({"id": "x", "abstract": "ABS", "source": "manual"})
        assert "ABS" in out

    def test_fetch_fulltext_arxiv(self):
        with patch("paper_tracker.sources.pdf.download_and_extract_text",
                   return_value="FULLTEXT BODY"):
            out = brainstorm_agent._fetch_fulltext({"id": "2401.00001", "source": "arxiv"})
        assert "FULLTEXT BODY" in out


# ---------------------------------------------------------------
# generate_ideas (agent loop mocked at _run_agent)
# ---------------------------------------------------------------

class TestGenerateIdeas:
    def test_returns_text_on_success(self):
        async def fake_run(prompt, tools, model, max_turns):
            return '[{"title":"X"}]'
        with patch.object(brainstorm_agent, "_build_tools", return_value=[]), \
             patch.object(brainstorm_agent, "_run_agent", fake_run):
            out = brainstorm_agent.generate_ideas(
                "task", manifest=[{"index": "P1", "title": "A"}],
                data_dir="/d", topic_id="t1", cfg={})
        assert out == '[{"title":"X"}]'

    def test_returns_empty_on_failure(self):
        async def boom(*a, **k):
            raise RuntimeError("sdk down")
        with patch.object(brainstorm_agent, "_build_tools", return_value=[]), \
             patch.object(brainstorm_agent, "_run_agent", boom):
            out = brainstorm_agent.generate_ideas(
                "task", manifest=[], data_dir="/d", topic_id="t1", cfg={})
        assert out == ""        # failure -> "" so caller falls back


# ---------------------------------------------------------------
# Manifest round-trip (insights writer -> brainstorm loader)
# ---------------------------------------------------------------

class TestManifestRoundTrip:
    def test_write_and_load(self, tmp_path):
        papers = [{"arxiv_id": "a1", "title": "A", "abstract": "x"},
                  {"paper_id": "pdf:zz", "title": "B", "summary": "s"}]
        sdir = tmp_path / "t1" / "sessions" / "S1"
        sdir.mkdir(parents=True)
        insights._write_papers_manifest(sdir, papers)

        reg = MagicMock()
        reg.get_latest_session.return_value = {"id": "S1"}
        out = brainstorm._load_insights_manifest(str(tmp_path), "t1", reg)
        assert [p["index"] for p in out] == ["P1", "P2"]
        assert out[0]["id"] == "a1" and out[1]["id"] == "pdf:zz"
        assert out[0]["title"] == "A"

    def test_load_missing_returns_empty(self, tmp_path):
        reg = MagicMock()
        reg.get_latest_session.return_value = {"id": "ghost"}
        assert brainstorm._load_insights_manifest(str(tmp_path), "t1", reg) == []


# ---------------------------------------------------------------
# Stage-1 dispatcher: agentic vs one-shot + fallback
# ---------------------------------------------------------------

class TestIdeaGenDispatch:
    def test_agentic_off_uses_call_cli(self):
        with patch("paper_tracker.brainstorm.call_cli", return_value="ONESHOT") as cc:
            out = brainstorm._idea_gen_raw(
                "p", {}, ctx_opts={"agentic_retrieval": False},
                data_dir="/d", topic_id="t1", registry=None)
        assert out == "ONESHOT"
        cc.assert_called_once()

    def test_agentic_on_uses_agent(self):
        with patch("paper_tracker.brainstorm._load_insights_manifest",
                   return_value=[{"index": "P1"}]), \
             patch("paper_tracker.brainstorm_agent.generate_ideas", return_value="AGENT") as ga, \
             patch("paper_tracker.brainstorm.call_cli") as cc:
            out = brainstorm._idea_gen_raw(
                "p", {}, ctx_opts={"agentic_retrieval": True},
                data_dir="/d", topic_id="t1", registry=MagicMock())
        assert out == "AGENT"
        ga.assert_called_once()
        cc.assert_not_called()

    def test_agentic_empty_falls_back_to_oneshot(self):
        with patch("paper_tracker.brainstorm._load_insights_manifest", return_value=[]), \
             patch("paper_tracker.brainstorm_agent.generate_ideas", return_value=""), \
             patch("paper_tracker.brainstorm.call_cli", return_value="FALLBACK") as cc:
            out = brainstorm._idea_gen_raw(
                "p", {}, ctx_opts={"agentic_retrieval": True},
                data_dir="/d", topic_id="t1", registry=MagicMock())
        assert out == "FALLBACK"
        cc.assert_called_once()


# ---------------------------------------------------------------
# Theory-focused review: normalize, bottleneck, dispatch, agent
# ---------------------------------------------------------------

class TestNormalizeReview:
    def test_overall_and_accept(self):
        r = brainstorm._normalize_review({"idea_title": "X", "novelty": 8, "rigor": 8, "significance": 7})
        assert r["verdict"] == "ACCEPT"
        assert r["overall"] >= 7.0
        assert r["novelty"] == 8 and r["rigor"] == 8 and r["significance"] == 7

    def test_redflag_forces_drop(self):
        r = brainstorm._normalize_review({"novelty": 9, "rigor": 9, "significance": 9, "feasibility_redflag": True})
        assert r["verdict"] == "DROP"

    def test_impact_alias_for_significance(self):
        r = brainstorm._normalize_review({"novelty": 6, "rigor": 6, "impact": 6})
        assert r["significance"] == 6

    def test_low_score_drops(self):
        r = brainstorm._normalize_review({"novelty": 2, "rigor": 4, "significance": 4})
        assert r["verdict"] == "DROP"          # any dim <= 2

    def test_high_novelty_conditional(self):
        r = brainstorm._normalize_review({"novelty": 9, "rigor": 5, "significance": 5})
        assert r["verdict"] == "CONDITIONAL_ACCEPT"

    def test_revise_autogenerates_instructions(self):
        r = brainstorm._normalize_review({"novelty": 5, "rigor": 5, "significance": 5})
        assert r["verdict"] == "REVISE"
        assert r["revision_instructions"]      # filled in when LLM gave none


class TestIdentifyBottleneck:
    def test_rigor_bottleneck_uses_diagnosis(self):
        b, hint = brainstorm._identify_bottleneck(
            {"novelty": 8, "rigor": 4, "significance": 7, "rigor_diagnosis": "assumption Z unproven"})
        assert b == "rigor"
        assert "rigor" in hint.lower() and "assumption Z" in hint

    def test_novelty_bottleneck(self):
        b, _ = brainstorm._identify_bottleneck({"novelty": 3, "rigor": 7, "significance": 7})
        assert b == "novelty"


class TestReviewIdeasDispatch:
    _JSON = '[{"idea_title":"X","novelty":7,"rigor":6,"significance":7,"weaknesses":[],"verdict":"REVISE"}]'

    def test_oneshot_when_agentic_off(self):
        with patch("paper_tracker.brainstorm.call_cli", return_value=self._JSON) as cc:
            out = brainstorm._review_ideas([{"title": "X"}], "T", "papers", 1, {})
        assert cc.called
        assert out[0]["idea_title"] == "X" and out[0]["novelty"] == 7 and out[0]["rigor"] == 6
        assert out[0]["overall"]               # derived

    def test_agentic_used_when_on(self):
        with patch("paper_tracker.brainstorm_agent.review_ideas", return_value=self._JSON) as ga, \
             patch("paper_tracker.brainstorm.call_cli") as cc:
            out = brainstorm._review_ideas([{"title": "X"}], "T", "papers", 1, {},
                                           ctx_opts={"agentic_review": True}, data_dir="/d", topic_id="t1")
        ga.assert_called_once()
        cc.assert_not_called()
        assert out[0]["idea_title"] == "X"

    def test_agentic_empty_falls_back(self):
        with patch("paper_tracker.brainstorm_agent.review_ideas", return_value="") as _ga, \
             patch("paper_tracker.brainstorm.call_cli", return_value=self._JSON) as cc:
            out = brainstorm._review_ideas([{"title": "X"}], "T", "papers", 1, {},
                                           ctx_opts={"agentic_review": True}, data_dir="/d", topic_id="t1")
        cc.assert_called_once()
        assert out[0]["idea_title"] == "X"

    def test_empty_ideas(self):
        assert brainstorm._review_ideas([], "T", "p", 0, {}) == []


class TestReviewIdeasAgent:
    def test_returns_text_on_success(self):
        async def fake_run(prompt, tools, model, max_turns):
            return '[{"idea_title":"X"}]'
        with patch.object(brainstorm_agent, "_build_tools", return_value=[]), \
             patch.object(brainstorm_agent, "_run_agent", fake_run):
            out = brainstorm_agent.review_ideas("task", data_dir="/d", topic_id="t1", cfg={})
        assert out == '[{"idea_title":"X"}]'

    def test_empty_on_failure(self):
        async def boom(*a, **k):
            raise RuntimeError("down")
        with patch.object(brainstorm_agent, "_build_tools", return_value=[]), \
             patch.object(brainstorm_agent, "_run_agent", boom):
            out = brainstorm_agent.review_ideas("task", data_dir="/d", topic_id="t1", cfg={})
        assert out == ""
