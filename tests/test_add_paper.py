"""Tests for the manual add-paper endpoints: POST .../papers/lookup + .../papers.

Mirrors test_server.py's harness: TestClient with mocked registry, but a real
temp data dir so the create path actually writes/reads tracker.db.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient

from paper_tracker.server import app
from paper_tracker.storage import Storage


_FAKE_TOPIC = {
    "id": "test-topic",
    "name": "Test Topic",
    "description": "A test",
    "arxiv_keywords": ["test"],
    "arxiv_categories": [],
    "arxiv_lookback_days": 2,
    "github_keywords": [],
    "github_lookback_days": 7,
    "schedule_cron": "",
    "enabled": True,
    "created_at": "2025-01-01",
}


@pytest.fixture(autouse=True)
def _server_globals(tmp_path):
    import paper_tracker.server as srv

    mock_reg = MagicMock()
    mock_reg.get_topic.return_value = _FAKE_TOPIC
    srv._registry = mock_reg
    srv._data_dir = str(tmp_path)
    srv._base_cfg = {
        "paths": {"data_dir": str(tmp_path)},
        "summarizer": {"claude_path": "claude", "claude_model": "opus"},
    }
    srv._scheduler = MagicMock()
    srv._summary_backfill_jobs.clear()
    yield
    srv._registry = None
    srv._scheduler = None


@pytest.fixture()
def client():
    return TestClient(app, raise_server_exceptions=False)


def _arxiv_paper(arxiv_id: str = "2401.12345") -> dict:
    return {
        "arxiv_id": arxiv_id,
        "title": "Fetched Title",
        "authors": "A. Author",
        "abstract": "Fetched abstract.",
        "url": f"https://arxiv.org/abs/{arxiv_id}",
        "published": "2024-01-15",
        "summary": "", "key_insight": "", "method": "", "contribution": "",
        "math_concepts": [], "venue": "", "cited_works": [],
        "doi": f"10.48550/arxiv.{arxiv_id}",
    }


# ---------------------------------------------------------------
# POST .../papers/lookup
# ---------------------------------------------------------------

class TestLookup:
    def test_topic_not_found(self, client):
        import paper_tracker.server as srv
        srv._registry.get_topic.return_value = None
        r = client.post("/api/topics/x/papers/lookup", json={"query": "2401.12345"})
        assert r.status_code == 404

    @patch("paper_tracker.sources.arxiv.fetch_by_id", return_value=_arxiv_paper())
    def test_found(self, mock_fetch, client):
        r = client.post("/api/topics/test-topic/papers/lookup",
                        json={"query": "https://arxiv.org/abs/2401.12345"})
        assert r.status_code == 200
        body = r.json()
        assert body["found"] is True
        assert body["paper"]["arxiv_id"] == "2401.12345"

    @patch("paper_tracker.sources.arxiv.fetch_by_id", return_value=None)
    def test_not_found(self, mock_fetch, client):
        r = client.post("/api/topics/test-topic/papers/lookup",
                        json={"query": "2401.00000"})
        assert r.status_code == 200
        assert r.json() == {"found": False, "paper": None}

    @patch("paper_tracker.sources.arxiv.fetch_by_id",
           side_effect=httpx.HTTPError("503"))
    def test_arxiv_error_returns_502(self, mock_fetch, client):
        r = client.post("/api/topics/test-topic/papers/lookup",
                        json={"query": "2401.12345"})
        assert r.status_code == 502

    def test_empty_query_400(self, client):
        r = client.post("/api/topics/test-topic/papers/lookup", json={"query": "  "})
        assert r.status_code == 400


# ---------------------------------------------------------------
# POST .../papers  (create)
# ---------------------------------------------------------------

class TestAddPaper:
    def test_add_arxiv_paper_persists(self, client, tmp_path):
        r = client.post("/api/topics/test-topic/papers", json={
            "arxiv_id": "https://arxiv.org/abs/2401.12345",
            "title": "My Paper", "authors": "X", "abstract": "Abs",
            "published": "2024-01-15", "venue": "ICLR 2024",
        })
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["summarizing"] is False
        p = body["paper"]
        assert p["arxiv_id"] == "2401.12345"   # extracted from URL
        assert p["source"] == "manual"
        assert p["venue"] == "ICLR 2024"
        assert p["url"] == "https://arxiv.org/abs/2401.12345"

        store = Storage(str(tmp_path), "test-topic")
        try:
            assert store.get_arxiv("2401.12345") is not None
        finally:
            store.close()

    def test_manual_paper_gets_synthetic_key(self, client):
        r = client.post("/api/topics/test-topic/papers", json={
            "title": "A blog-post idea", "url": "https://example.com/post",
        })
        assert r.status_code == 201, r.text
        p = r.json()["paper"]
        assert p["arxiv_id"].startswith("manual:")
        assert p["source"] == "manual"

    def test_duplicate_arxiv_returns_409(self, client):
        payload = {"arxiv_id": "2401.55555", "title": "Dup"}
        assert client.post("/api/topics/test-topic/papers", json=payload).status_code == 201
        assert client.post("/api/topics/test-topic/papers", json=payload).status_code == 409

    def test_manual_duplicate_dedups_by_title_url(self, client):
        payload = {"title": "Same", "url": "https://e.com/x"}
        assert client.post("/api/topics/test-topic/papers", json=payload).status_code == 201
        assert client.post("/api/topics/test-topic/papers", json=payload).status_code == 409

    def test_unparseable_arxiv_id_400(self, client):
        r = client.post("/api/topics/test-topic/papers",
                        json={"arxiv_id": "not-an-id", "title": "T"})
        assert r.status_code == 400

    def test_missing_title_422(self, client):
        r = client.post("/api/topics/test-topic/papers",
                        json={"arxiv_id": "2401.12345"})
        assert r.status_code == 422

    def test_blank_title_422(self, client):
        r = client.post("/api/topics/test-topic/papers",
                        json={"arxiv_id": "2401.12345", "title": "   "})
        assert r.status_code == 422

    @patch("paper_tracker.server._submit_summarize_job")
    def test_summarize_flag_triggers_job(self, mock_job, client):
        r = client.post("/api/topics/test-topic/papers", json={
            "arxiv_id": "2401.77777", "title": "Sum me", "summarize": True,
        })
        assert r.status_code == 201
        assert r.json()["summarizing"] is True
        mock_job.assert_called_once()

    def test_no_summarize_does_not_trigger_job(self, client):
        with patch("paper_tracker.server._submit_summarize_job") as mock_job:
            r = client.post("/api/topics/test-topic/papers",
                            json={"arxiv_id": "2401.88888", "title": "No sum"})
            assert r.status_code == 201
            mock_job.assert_not_called()


# ---------------------------------------------------------------
# POST .../papers/batch  (bulk add by URL/ID)
# ---------------------------------------------------------------

def _meta(arxiv_id: str, title: str = "T") -> dict:
    return {
        "arxiv_id": arxiv_id, "title": title, "authors": "A", "abstract": "abs",
        "url": f"https://arxiv.org/abs/{arxiv_id}", "published": "2024-01-01",
        "doi": f"10.48550/arxiv.{arxiv_id}",
    }


class TestBatchAdd:
    def test_topic_not_found(self, client):
        import paper_tracker.server as srv
        srv._registry.get_topic.return_value = None
        r = client.post("/api/topics/x/papers/batch", json={"text": "2401.11111"})
        assert r.status_code == 404

    def test_empty_text_400(self, client):
        r = client.post("/api/topics/test-topic/papers/batch", json={"text": "   "})
        assert r.status_code == 400

    @patch("paper_tracker.sources.arxiv.fetch_many_by_id")
    def test_adds_multiple(self, mock_fetch, client, tmp_path):
        mock_fetch.return_value = {
            "2401.11111": _meta("2401.11111", "P1"),
            "2401.22222": _meta("2401.22222", "P2"),
        }
        r = client.post("/api/topics/test-topic/papers/batch", json={
            "text": "https://arxiv.org/abs/2401.11111, arXiv:2401.22222"})
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["counts"]["added"] == 2
        assert body["summarizing"] is False
        store = Storage(str(tmp_path), "test-topic")
        try:
            assert store.get_arxiv("2401.11111") is not None
            assert store.get_arxiv("2401.22222")["source"] == "manual"
        finally:
            store.close()

    @patch("paper_tracker.sources.arxiv.fetch_many_by_id")
    def test_invalid_notfound_duplicate_mixed(self, mock_fetch, client, tmp_path):
        store = Storage(str(tmp_path), "test-topic")
        try:
            store.insert_arxiv({
                "arxiv_id": "2401.33333", "paper_id": "2401.33333", "source": "manual",
                "title": "dup", "authors": "", "abstract": "", "url": "", "published": "",
                "summary": "", "key_insight": "", "method": "", "contribution": "",
                "math_concepts": [], "venue": "", "cited_works": [], "quality_score": 0,
                "citation_count": 0, "doi": "10.48550/arxiv.2401.33333",
            })
        finally:
            store.close()
        mock_fetch.return_value = {"2401.11111": _meta("2401.11111")}  # 99999 absent

        r = client.post("/api/topics/test-topic/papers/batch", json={
            "text": "2401.11111, 2401.99999, not-a-paper, 2401.33333"})
        assert r.status_code == 201
        counts = r.json()["counts"]
        assert counts == {"added": 1, "duplicate": 1, "not_found": 1, "invalid": 1, "error": 0}
        # only the valid, non-duplicate ids are fetched
        assert mock_fetch.call_args.args[0] == ["2401.11111", "2401.99999"]

    @patch("paper_tracker.sources.arxiv.fetch_many_by_id",
           side_effect=httpx.HTTPError("503"))
    def test_fetch_error_marks_error(self, mock_fetch, client):
        r = client.post("/api/topics/test-topic/papers/batch", json={"text": "2401.11111"})
        assert r.status_code == 201
        assert r.json()["counts"]["error"] == 1

    @patch("paper_tracker.server._submit_batch_summarize_job")
    @patch("paper_tracker.sources.arxiv.fetch_many_by_id")
    def test_summarize_flag_triggers_one_job(self, mock_fetch, mock_job, client):
        mock_fetch.return_value = {"2401.11111": _meta("2401.11111")}
        r = client.post("/api/topics/test-topic/papers/batch",
                        json={"text": "2401.11111", "summarize": True})
        assert r.json()["summarizing"] is True
        mock_job.assert_called_once()

    @patch("paper_tracker.sources.arxiv.fetch_many_by_id")
    def test_same_id_twice_dedups_in_batch(self, mock_fetch, client):
        mock_fetch.return_value = {"2401.11111": _meta("2401.11111")}
        r = client.post("/api/topics/test-topic/papers/batch", json={
            "text": "2401.11111, https://arxiv.org/abs/2401.11111"})
        counts = r.json()["counts"]
        assert counts["added"] == 1
        assert counts["duplicate"] == 1  # second form is an in-batch duplicate


# ---------------------------------------------------------------
# POST/GET .../papers/backfill-summaries
# ---------------------------------------------------------------

class _SyncExecutor:
    def submit(self, fn, *a, **k):
        fn(*a, **k)
        return MagicMock()


def _seed_paper(tmp_path, arxiv_id, key_insight=""):
    store = Storage(str(tmp_path), "test-topic")
    try:
        store.insert_arxiv({
            "arxiv_id": arxiv_id, "paper_id": arxiv_id, "source": "manual",
            "title": f"Paper {arxiv_id}", "authors": "A", "abstract": "some abstract",
            "url": "", "published": "2024-01-01", "summary": "", "key_insight": key_insight,
            "method": "", "contribution": "", "math_concepts": [], "venue": "",
            "cited_works": [], "quality_score": 0, "citation_count": 0,
            "doi": f"10.48550/arxiv.{arxiv_id}",
        })
    finally:
        store.close()


class TestBackfillSummaries:
    def test_topic_not_found(self, client):
        import paper_tracker.server as srv
        srv._registry.get_topic.return_value = None
        r = client.post("/api/topics/x/papers/backfill-summaries")
        assert r.status_code == 404

    def test_no_missing_completes_with_zero(self, client, tmp_path):
        _seed_paper(tmp_path, "2401.00001", key_insight="already summarized")
        r = client.post("/api/topics/test-topic/papers/backfill-summaries")
        assert r.status_code == 202
        assert r.json()["total"] == 0

    def test_backfills_only_missing(self, client, tmp_path):
        _seed_paper(tmp_path, "2401.00002", key_insight="")           # missing
        _seed_paper(tmp_path, "2401.00003", key_insight="present")    # has one
        import paper_tracker.server as srv

        def _fake_summarize(papers, cfg):
            for p in papers:
                p["key_insight"] = "GENERATED"
                p["summary"] = "S"

        with patch.object(srv, "_brainstorm_executor", _SyncExecutor()), \
             patch("paper_tracker.server.summarize_papers", side_effect=_fake_summarize):
            r = client.post("/api/topics/test-topic/papers/backfill-summaries")
        assert r.status_code == 202
        assert r.json()["total"] == 1

        s = client.get("/api/topics/test-topic/papers/backfill-summaries").json()
        assert s["status"] == "completed" and s["processed"] == 1

        store = Storage(str(tmp_path), "test-topic")
        try:
            assert store.get_arxiv("2401.00002")["key_insight"] == "GENERATED"
            assert store.get_arxiv("2401.00003")["key_insight"] == "present"  # untouched
        finally:
            store.close()

    def test_409_when_running(self, client):
        import paper_tracker.server as srv
        srv._summary_backfill_jobs["test-topic"] = {"status": "running", "total": 5, "processed": 0}
        r = client.post("/api/topics/test-topic/papers/backfill-summaries")
        assert r.status_code == 409
