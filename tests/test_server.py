"""Unit tests for server.py — DELETE paper, POST/GET refilter endpoints.

Uses FastAPI TestClient with mocked registry and storage.
"""

from __future__ import annotations

import json
import time
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from paper_tracker.server import app, _refilter_jobs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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
def _setup_server_globals():
    """Set up server globals so endpoints don't hit 503."""
    import paper_tracker.server as srv

    mock_reg = MagicMock()
    mock_reg.get_topic.return_value = _FAKE_TOPIC
    mock_reg.list_topics.return_value = [_FAKE_TOPIC]

    srv._registry = mock_reg
    srv._data_dir = "/tmp/test-paper-tracker"
    srv._base_cfg = {
        "paths": {"data_dir": "/tmp/test-paper-tracker"},
        "summarizer": {"claude_path": "claude", "claude_model": "opus"},
    }
    srv._scheduler = MagicMock()

    # Clear refilter jobs between tests
    _refilter_jobs.clear()

    yield

    srv._registry = None
    srv._scheduler = None


@pytest.fixture()
def client():
    return TestClient(app, raise_server_exceptions=False)


def _fake_paper(arxiv_id: str, quality_score: int = 3) -> dict:
    return {
        "arxiv_id": arxiv_id,
        "title": f"Paper {arxiv_id}",
        "authors": "A, B",
        "abstract": "Abstract",
        "url": f"https://arxiv.org/abs/{arxiv_id}",
        "published": "2025-01-01",
        "summary": "Summary",
        "key_insight": "Insight",
        "method": "Method",
        "contribution": "Contribution",
        "math_concepts": [],
        "venue": "",
        "cited_works": [],
        "quality_score": quality_score,
        "added_at": "2025-01-01 00:00:00",
        "notified": 0,
    }


# ---------------------------------------------------------------------------
# DELETE /api/topics/{topic_id}/papers/{arxiv_id}
# ---------------------------------------------------------------------------

class TestDeletePaper:
    @patch("paper_tracker.server.Storage")
    def test_delete_success(self, MockStorage, client):
        mock_store = MagicMock()
        mock_store.delete_arxiv.return_value = True
        MockStorage.return_value = mock_store

        resp = client.delete("/api/topics/test-topic/papers/2501.00001")
        assert resp.status_code == 204
        mock_store.delete_arxiv.assert_called_once_with("2501.00001")
        mock_store.close.assert_called_once()

    @patch("paper_tracker.server.Storage")
    def test_delete_not_found(self, MockStorage, client):
        mock_store = MagicMock()
        mock_store.delete_arxiv.return_value = False
        MockStorage.return_value = mock_store

        resp = client.delete("/api/topics/test-topic/papers/9999.99999")
        assert resp.status_code == 404

    def test_delete_topic_not_found(self, client):
        import paper_tracker.server as srv
        srv._registry.get_topic.return_value = None

        resp = client.delete("/api/topics/nonexistent/papers/2501.00001")
        assert resp.status_code == 404
        assert "Topic not found" in resp.json()["detail"]

    @patch("paper_tracker.server.Storage")
    def test_delete_store_always_closed(self, MockStorage, client):
        """Storage.close() must be called even if delete raises."""
        mock_store = MagicMock()
        mock_store.delete_arxiv.side_effect = Exception("db error")
        MockStorage.return_value = mock_store

        resp = client.delete("/api/topics/test-topic/papers/2501.00001")
        assert resp.status_code == 500
        mock_store.close.assert_called_once()


# ---------------------------------------------------------------------------
# POST /api/topics/{topic_id}/papers/refilter
# ---------------------------------------------------------------------------

class TestStartRefilter:
    @patch("paper_tracker.server.refilter_papers")
    @patch("paper_tracker.server.Storage")
    def test_start_refilter_202(self, MockStorage, mock_refilter, client):
        mock_store = MagicMock()
        mock_store.get_all_arxiv.return_value = ([_fake_paper("001")], 1)
        MockStorage.return_value = mock_store

        resp = client.post(
            "/api/topics/test-topic/papers/refilter",
            json={"custom_instructions": "keep ML only", "min_quality": 3, "auto_delete": False},
        )
        assert resp.status_code == 202
        data = resp.json()
        assert data["status"] == "started"
        assert data["topic_id"] == "test-topic"

    def test_refilter_topic_not_found(self, client):
        import paper_tracker.server as srv
        srv._registry.get_topic.return_value = None

        resp = client.post(
            "/api/topics/nonexistent/papers/refilter",
            json={},
        )
        assert resp.status_code == 404

    def test_refilter_conflict_when_running(self, client):
        _refilter_jobs["test-topic"] = {"status": "running", "total": 10, "processed": 0, "removed": 0}

        resp = client.post(
            "/api/topics/test-topic/papers/refilter",
            json={},
        )
        assert resp.status_code == 409


# ---------------------------------------------------------------------------
# GET /api/topics/{topic_id}/papers/refilter
# ---------------------------------------------------------------------------

class TestGetRefilterStatus:
    def test_idle_when_no_job(self, client):
        resp = client.get("/api/topics/test-topic/papers/refilter")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "idle"
        assert data["topic_id"] == "test-topic"

    def test_returns_running_status(self, client):
        _refilter_jobs["test-topic"] = {
            "status": "running",
            "total": 50,
            "processed": 0,
            "removed": 0,
        }
        resp = client.get("/api/topics/test-topic/papers/refilter")
        data = resp.json()
        assert data["status"] == "running"
        assert data["total"] == 50

    def test_returns_completed_status(self, client):
        _refilter_jobs["test-topic"] = {
            "status": "completed",
            "total": 50,
            "processed": 50,
            "removed": 5,
        }
        resp = client.get("/api/topics/test-topic/papers/refilter")
        data = resp.json()
        assert data["status"] == "completed"
        assert data["removed"] == 5

    def test_topic_not_found(self, client):
        import paper_tracker.server as srv
        srv._registry.get_topic.return_value = None

        resp = client.get("/api/topics/nonexistent/papers/refilter")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Route ordering: refilter before {arxiv_id:path}
# ---------------------------------------------------------------------------

class TestRouteOrdering:
    """Ensure /papers/refilter is not captured by /papers/{arxiv_id:path}."""

    def test_get_refilter_not_captured_as_paper(self, client):
        """GET /papers/refilter should hit the refilter endpoint, not get_paper."""
        resp = client.get("/api/topics/test-topic/papers/refilter")
        # If it were captured by get_paper, it would try to find arxiv_id="refilter"
        # and return 404 with "Paper not found", not the refilter status
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "topic_id" in data

    @patch("paper_tracker.server.Storage")
    def test_get_paper_still_works(self, MockStorage, client):
        """GET /papers/{real_arxiv_id} should still work."""
        mock_store = MagicMock()
        mock_store.get_arxiv.return_value = _fake_paper("2501.00001")
        MockStorage.return_value = mock_store

        resp = client.get("/api/topics/test-topic/papers/2501.00001")
        assert resp.status_code == 200
        assert resp.json()["arxiv_id"] == "2501.00001"


# ---------------------------------------------------------------------------
# GET /api/topics/{topic_id}/papers (list with quality scores)
# ---------------------------------------------------------------------------

class TestListPapers:
    @patch("paper_tracker.server.Storage")
    def test_list_includes_quality_score(self, MockStorage, client):
        mock_store = MagicMock()
        mock_store.get_all_arxiv.return_value = (
            [_fake_paper("001", quality_score=5), _fake_paper("002", quality_score=2)],
            2,
        )
        MockStorage.return_value = mock_store

        resp = client.get("/api/topics/test-topic/papers")
        assert resp.status_code == 200
        papers = resp.json()["papers"]
        assert papers[0]["quality_score"] == 5
        assert papers[1]["quality_score"] == 2
