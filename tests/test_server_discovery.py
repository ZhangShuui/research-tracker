"""Unit tests for server.py — discovery endpoints.

Uses FastAPI TestClient with mocked registry.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from paper_tracker.server import app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _setup_server_globals():
    """Set up server globals so endpoints don't hit 503."""
    import paper_tracker.server as srv

    mock_reg = MagicMock()
    srv._registry = mock_reg
    srv._data_dir = "/tmp/test-paper-tracker"
    srv._base_cfg = {
        "paths": {"data_dir": "/tmp/test-paper-tracker"},
        "summarizer": {"claude_path": "claude", "claude_model": "opus"},
    }
    srv._scheduler = MagicMock()

    yield

    srv._registry = None
    srv._scheduler = None


@pytest.fixture()
def client():
    return TestClient(app, raise_server_exceptions=False)


def _fake_report(report_id: str = "dr-abc12345", rtype: str = "trending", status: str = "completed") -> dict:
    return {
        "id": report_id,
        "type": rtype,
        "status": status,
        "started_at": "2026-03-05T10:00:00",
        "finished_at": "2026-03-05T10:05:00" if status == "completed" else None,
        "content": "# Report\nSome content" if status == "completed" else "",
        "papers_json": [{"arxiv_id": "2603.001", "title": "Test", "source": "arxiv"}],
        "paper_count": 42,
        "source_stats": {"arxiv": 30, "huggingface": 12},
    }


# ---------------------------------------------------------------------------
# POST /api/discovery
# ---------------------------------------------------------------------------

class TestStartDiscovery:
    @patch("paper_tracker.server._brainstorm_executor")
    def test_start_trending_202(self, mock_exec, client):
        import paper_tracker.server as srv
        srv._registry.list_discovery_reports.return_value = []

        resp = client.post("/api/discovery", json={"type": "trending"})
        assert resp.status_code == 202
        data = resp.json()
        assert data["status"] == "started"
        assert data["type"] == "trending"
        mock_exec.submit.assert_called_once()

    @patch("paper_tracker.server._brainstorm_executor")
    def test_start_math_202(self, mock_exec, client):
        import paper_tracker.server as srv
        srv._registry.list_discovery_reports.return_value = []

        resp = client.post("/api/discovery", json={"type": "math"})
        assert resp.status_code == 202
        assert resp.json()["type"] == "math"

    def test_invalid_type_400(self, client):
        resp = client.post("/api/discovery", json={"type": "invalid"})
        assert resp.status_code == 400

    @patch("paper_tracker.server._brainstorm_executor")
    def test_conflict_when_running_409(self, mock_exec, client):
        import paper_tracker.server as srv
        srv._registry.list_discovery_reports.return_value = [
            _fake_report(status="running"),
        ]

        resp = client.post("/api/discovery", json={"type": "trending"})
        assert resp.status_code == 409

    @patch("paper_tracker.server._brainstorm_executor")
    def test_allows_start_when_previous_completed(self, mock_exec, client):
        import paper_tracker.server as srv
        srv._registry.list_discovery_reports.return_value = [
            _fake_report(status="completed"),
        ]

        resp = client.post("/api/discovery", json={"type": "trending"})
        assert resp.status_code == 202


# ---------------------------------------------------------------------------
# GET /api/discovery
# ---------------------------------------------------------------------------

class TestListDiscovery:
    def test_list_all(self, client):
        import paper_tracker.server as srv
        srv._registry.list_discovery_reports.return_value = [
            _fake_report("dr-001", "trending"),
            _fake_report("dr-002", "math"),
        ]

        resp = client.get("/api/discovery")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["reports"]) == 2

    def test_filter_by_type(self, client):
        import paper_tracker.server as srv
        srv._registry.list_discovery_reports.return_value = [
            _fake_report("dr-001", "trending"),
        ]

        resp = client.get("/api/discovery?type=trending")
        assert resp.status_code == 200
        srv._registry.list_discovery_reports.assert_called_with(
            report_type="trending", limit=20,
        )

    def test_empty_list(self, client):
        import paper_tracker.server as srv
        srv._registry.list_discovery_reports.return_value = []

        resp = client.get("/api/discovery")
        assert resp.status_code == 200
        assert resp.json()["reports"] == []


# ---------------------------------------------------------------------------
# GET /api/discovery/latest/{type}
# ---------------------------------------------------------------------------

class TestGetLatestDiscovery:
    def test_latest_trending(self, client):
        import paper_tracker.server as srv
        srv._registry.get_latest_discovery_report.return_value = _fake_report()

        resp = client.get("/api/discovery/latest/trending")
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "trending"
        assert data["status"] == "completed"

    def test_latest_math(self, client):
        import paper_tracker.server as srv
        srv._registry.get_latest_discovery_report.return_value = _fake_report(rtype="math")

        resp = client.get("/api/discovery/latest/math")
        assert resp.status_code == 200

    def test_invalid_type_400(self, client):
        resp = client.get("/api/discovery/latest/invalid")
        assert resp.status_code == 400

    def test_no_completed_but_running(self, client):
        """If no completed report exists but one is running, return the running one."""
        import paper_tracker.server as srv
        srv._registry.get_latest_discovery_report.return_value = None
        srv._registry.list_discovery_reports.return_value = [
            _fake_report(status="running"),
        ]

        resp = client.get("/api/discovery/latest/trending")
        assert resp.status_code == 200
        assert resp.json()["status"] == "running"

    def test_no_reports_at_all_404(self, client):
        import paper_tracker.server as srv
        srv._registry.get_latest_discovery_report.return_value = None
        srv._registry.list_discovery_reports.return_value = []

        resp = client.get("/api/discovery/latest/trending")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/discovery/{report_id}
# ---------------------------------------------------------------------------

class TestGetDiscoveryReport:
    def test_get_existing(self, client):
        import paper_tracker.server as srv
        srv._registry.get_discovery_report.return_value = _fake_report("dr-abc12345")

        resp = client.get("/api/discovery/dr-abc12345")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "dr-abc12345"
        assert data["paper_count"] == 42

    def test_get_nonexistent_404(self, client):
        import paper_tracker.server as srv
        srv._registry.get_discovery_report.return_value = None

        resp = client.get("/api/discovery/dr-nonexistent")
        assert resp.status_code == 404

    def test_response_includes_all_fields(self, client):
        import paper_tracker.server as srv
        srv._registry.get_discovery_report.return_value = _fake_report()

        resp = client.get("/api/discovery/dr-abc12345")
        data = resp.json()
        assert "id" in data
        assert "type" in data
        assert "status" in data
        assert "content" in data
        assert "papers_json" in data
        assert "paper_count" in data
        assert "source_stats" in data


# ---------------------------------------------------------------------------
# Route ordering: latest/{type} before {report_id}
# ---------------------------------------------------------------------------

class TestDiscoveryRouteOrdering:
    def test_latest_not_captured_as_report_id(self, client):
        """GET /api/discovery/latest/trending should not be captured by /{report_id}."""
        import paper_tracker.server as srv
        srv._registry.get_latest_discovery_report.return_value = _fake_report()

        resp = client.get("/api/discovery/latest/trending")
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "trending"
        # If captured by /{report_id}, it would call get_discovery_report("latest")
        # and likely return 404
        srv._registry.get_latest_discovery_report.assert_called_once_with("trending")
