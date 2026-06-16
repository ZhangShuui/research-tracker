"""Tests for the manual insights endpoints: suggest-cross-domain + generate.

Uses a REAL Registry (tmp) + Storage (tmp) so session creation/update round-trips,
with the executor patched to run synchronously and the LLM calls patched out.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from paper_tracker.server import app
from paper_tracker.registry import Registry
from paper_tracker.storage import Storage


class _SyncExecutor:
    """Runs submitted jobs inline so background work completes within the request."""
    def submit(self, fn, *args, **kwargs):
        fn(*args, **kwargs)
        return MagicMock()


@pytest.fixture(autouse=True)
def _server_globals(tmp_path):
    import paper_tracker.server as srv

    reg = Registry(str(tmp_path))
    reg.create_topic({
        "id": "t1", "name": "Topic One",
        "arxiv_keywords": ["x"], "arxiv_categories": [], "github_keywords": [],
    })
    srv._registry = reg
    srv._data_dir = str(tmp_path)
    srv._base_cfg = {
        "paths": {"data_dir": str(tmp_path)},
        "summarizer": {"claude_path": "claude", "claude_model": "opus"},
    }
    srv._scheduler = MagicMock()

    orig_exec = srv._brainstorm_executor
    srv._brainstorm_executor = _SyncExecutor()
    yield
    srv._brainstorm_executor = orig_exec
    reg.close()
    srv._registry = None
    srv._scheduler = None


@pytest.fixture()
def client():
    return TestClient(app, raise_server_exceptions=False)


def _seed_paper(tmp_path, arxiv_id="2401.00001"):
    store = Storage(str(tmp_path), "t1")
    try:
        store.insert_arxiv({
            "arxiv_id": arxiv_id, "title": f"Paper {arxiv_id}", "authors": "A",
            "abstract": "abs", "url": f"https://arxiv.org/abs/{arxiv_id}",
            "published": "2025-01-01", "summary": "s", "key_insight": "k",
            "method": "m", "contribution": "c", "math_concepts": ["x"],
            "venue": "v", "cited_works": [],
        })
    finally:
        store.close()


def _fake_generate(papers, topic_name, session_dir, cfg):
    d = Path(session_dir)
    d.mkdir(parents=True, exist_ok=True)
    p = d / "insights.md"
    p.write_text(f"## Memo\n\n{len(papers)} papers\n", encoding="utf-8")
    return p


def _fake_generate_agentic(papers, topic_name, session_dir, cfg, *, progress_cb=None):
    d = Path(session_dir)
    d.mkdir(parents=True, exist_ok=True)
    p = d / "insights.md"
    p.write_text(f"## Agentic Memo\n\n{len(papers)} papers\n", encoding="utf-8")
    return p


# ---------------------------------------------------------------
# suggest-cross-domain
# ---------------------------------------------------------------

class TestSuggest:
    def test_topic_not_found(self, client):
        r = client.post("/api/topics/nope/insights/suggest-cross-domain",
                        json={"paper_ids": ["x"]})
        assert r.status_code == 404

    def test_empty_paper_ids_400(self, client):
        r = client.post("/api/topics/t1/insights/suggest-cross-domain",
                        json={"paper_ids": []})
        assert r.status_code == 400

    def test_no_papers_found_400(self, client):
        r = client.post("/api/topics/t1/insights/suggest-cross-domain",
                        json={"paper_ids": ["ghost"]})
        assert r.status_code == 400

    @patch("paper_tracker.insights.suggest_cross_domain")
    @patch("paper_tracker.crossdomain.retrieve",
           return_value=[{"arxiv_id": "kb1", "title": "KB paper", "domain": "math.OC",
                          "rationale": "foundational", "source": "kb", "score": 0.9,
                          "authors": "", "abstract": "", "url": "", "published": ""}])
    def test_default_uses_knowledge_base(self, mock_retrieve, mock_live, client, tmp_path):
        _seed_paper(tmp_path)
        r = client.post("/api/topics/t1/insights/suggest-cross-domain",
                        json={"paper_ids": ["2401.00001"]})
        assert r.status_code == 200
        body = r.json()
        assert body["candidates"][0]["arxiv_id"] == "kb1"
        assert body["candidates"][0]["source"] == "kb"
        assert body["kb_count"] == 1 and body["live"] is False
        mock_retrieve.assert_called_once()
        mock_live.assert_not_called()  # live search NOT run by default

    @patch("paper_tracker.insights.suggest_cross_domain",
           return_value=[{"arxiv_id": "live1", "title": "Live", "domain": "physics",
                          "rationale": "r", "authors": "", "abstract": "",
                          "url": "", "published": ""}])
    @patch("paper_tracker.crossdomain.retrieve", return_value=[])
    def test_live_search_toggle_merges(self, mock_retrieve, mock_live, client, tmp_path):
        _seed_paper(tmp_path)
        r = client.post("/api/topics/t1/insights/suggest-cross-domain",
                        json={"paper_ids": ["2401.00001"], "live_search": True})
        assert r.status_code == 200
        body = r.json()
        assert body["live"] is True
        assert body["candidates"][0]["arxiv_id"] == "live1"
        assert body["candidates"][0]["source"] == "live"  # tagged
        mock_live.assert_called_once()


# ---------------------------------------------------------------
# generate
# ---------------------------------------------------------------

class TestGenerate:
    def test_topic_not_found(self, client):
        r = client.post("/api/topics/nope/insights/generate",
                        json={"paper_ids": ["x"]})
        assert r.status_code == 404

    def test_empty_paper_ids_400(self, client):
        r = client.post("/api/topics/t1/insights/generate", json={"paper_ids": []})
        assert r.status_code == 400

    @patch("paper_tracker.insights.generate", side_effect=_fake_generate)
    def test_generates_manual_session(self, mock_gen, client, tmp_path):
        _seed_paper(tmp_path)
        r = client.post("/api/topics/t1/insights/generate",
                        json={"paper_ids": ["2401.00001"]})
        assert r.status_code == 202, r.text
        sid = r.json()["session_id"]

        # poll target: the manual session is completed with insights inlined
        s = client.get(f"/api/topics/t1/sessions/{sid}").json()
        assert s["kind"] == "manual"
        assert s["status"] == "completed"
        assert s["paper_count"] == 1
        assert "## Memo" in s["insights_content"]
        mock_gen.assert_called_once()

    @patch("paper_tracker.insights.generate", side_effect=_fake_generate)
    def test_cross_domain_papers_counted(self, mock_gen, client, tmp_path):
        _seed_paper(tmp_path)
        r = client.post("/api/topics/t1/insights/generate", json={
            "paper_ids": ["2401.00001"],
            "cross_domain_papers": [
                {"arxiv_id": "m1", "title": "Optimal Transport", "abstract": "ot",
                 "domain": "math.OC"},
            ],
        })
        sid = r.json()["session_id"]
        s = client.get(f"/api/topics/t1/sessions/{sid}").json()
        assert s["paper_count"] == 2  # 1 selected + 1 cross-domain

    @patch("paper_tracker.insights.generate", side_effect=_fake_generate)
    def test_no_papers_found_marks_failed(self, mock_gen, client):
        r = client.post("/api/topics/t1/insights/generate",
                        json={"paper_ids": ["ghost"]})
        assert r.status_code == 202  # session created
        sid = r.json()["session_id"]
        s = client.get(f"/api/topics/t1/sessions/{sid}").json()
        assert s["status"] == "failed"
        mock_gen.assert_not_called()

    @patch("paper_tracker.insights.generate_agentic", side_effect=_fake_generate_agentic)
    @patch("paper_tracker.insights.generate", side_effect=_fake_generate)
    def test_agentic_mode_uses_pipeline(self, mock_single, mock_agentic, client, tmp_path):
        _seed_paper(tmp_path)
        r = client.post("/api/topics/t1/insights/generate",
                        json={"paper_ids": ["2401.00001"], "mode": "agentic"})
        assert r.status_code == 202, r.text
        sid = r.json()["session_id"]
        s = client.get(f"/api/topics/t1/sessions/{sid}").json()
        assert s["status"] == "completed"
        assert "## Agentic Memo" in s["insights_content"]
        mock_agentic.assert_called_once()
        mock_single.assert_not_called()              # agentic path, not the single-call one

    @patch("paper_tracker.insights.generate_agentic", side_effect=_fake_generate_agentic)
    @patch("paper_tracker.insights.generate", side_effect=_fake_generate)
    def test_default_mode_uses_single(self, mock_single, mock_agentic, client, tmp_path):
        _seed_paper(tmp_path)
        r = client.post("/api/topics/t1/insights/generate",
                        json={"paper_ids": ["2401.00001"]})   # no mode -> single
        assert r.status_code == 202
        mock_single.assert_called_once()
        mock_agentic.assert_not_called()

    @patch("paper_tracker.insights.generate", side_effect=_fake_generate)
    def test_manual_session_excluded_default_not_in_runs(self, mock_gen, client, tmp_path):
        """Manual sessions carry kind='manual' so the UI can keep them out of run history."""
        _seed_paper(tmp_path)
        sid = client.post("/api/topics/t1/insights/generate",
                          json={"paper_ids": ["2401.00001"]}).json()["session_id"]
        sessions = client.get("/api/topics/t1/sessions").json()["sessions"]
        manual = [s for s in sessions if s["id"] == sid]
        assert manual and manual[0]["kind"] == "manual"
