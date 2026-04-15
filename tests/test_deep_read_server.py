"""Unit tests for server.py — Deep Read endpoints.

Uses FastAPI TestClient with mocked registry and storage.
"""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from paper_tracker.server import app, _deep_read_progress, _deep_read_qa_progress


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

_FAKE_PAPER = {
    "arxiv_id": "2501.00001",
    "title": "Test Paper",
    "authors": "Alice, Bob",
    "abstract": "An abstract",
    "url": "https://arxiv.org/abs/2501.00001",
    "published": "2025-01-01",
    "summary": "Summary",
    "key_insight": "Insight",
    "method": "Method",
    "contribution": "Contribution",
    "math_concepts": [],
    "venue": "NeurIPS",
    "cited_works": [],
    "quality_score": 4,
    "added_at": "2025-01-01",
    "notified": 0,
}

_FAKE_DR_SESSION = {
    "id": "dr-abc12345",
    "topic_id": "test-topic",
    "paper_id": "2501.00001",
    "paper_title": "Test Paper",
    "status": "running",
    "started_at": "2025-01-01T00:00:00",
    "finished_at": None,
    "motivation_analysis": "",
    "method_analysis": "",
    "experiment_analysis": "",
    "contribution_analysis": "",
    "summary_card_json": {},
    "user_notes": "",
}

_FAKE_DR_SESSION_COMPLETED = {
    **_FAKE_DR_SESSION,
    "status": "completed",
    "finished_at": "2025-01-01T01:00:00",
    "motivation_analysis": "M",
    "method_analysis": "Me",
    "experiment_analysis": "E",
    "contribution_analysis": "C",
    "summary_card_json": {"key_innovations": ["A"]},
}


@pytest.fixture(autouse=True)
def _setup_server_globals():
    """Set up server globals so endpoints don't hit 503."""
    import paper_tracker.server as srv

    mock_reg = MagicMock()
    mock_reg.get_topic.return_value = _FAKE_TOPIC

    srv._registry = mock_reg
    srv._data_dir = "/tmp/test-paper-tracker"
    srv._base_cfg = {
        "paths": {"data_dir": "/tmp/test-paper-tracker"},
        "summarizer": {"claude_path": "claude", "claude_model": "opus"},
    }
    srv._scheduler = MagicMock()

    # Clear progress dicts between tests
    _deep_read_progress.clear()
    _deep_read_qa_progress.clear()

    yield

    srv._registry = None
    srv._scheduler = None


@pytest.fixture()
def client():
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# POST /api/topics/{topic_id}/deep-read
# ---------------------------------------------------------------------------

class TestStartDeepRead:
    @patch("paper_tracker.server._brainstorm_executor")
    @patch("paper_tracker.server.Storage")
    def test_start_202(self, MockStorage, mock_executor, client):
        import paper_tracker.server as srv

        mock_store = MagicMock()
        mock_store.get_arxiv.return_value = _FAKE_PAPER
        MockStorage.return_value = mock_store

        srv._registry.create_deep_read_session.return_value = _FAKE_DR_SESSION

        resp = client.post(
            "/api/topics/test-topic/deep-read",
            json={"paper_id": "2501.00001"},
        )
        assert resp.status_code == 202
        data = resp.json()
        assert data["status"] == "started"
        assert data["session_id"] == "dr-abc12345"
        mock_executor.submit.assert_called_once()
        mock_store.close.assert_called_once()

    def test_topic_not_found(self, client):
        import paper_tracker.server as srv
        srv._registry.get_topic.return_value = None

        resp = client.post(
            "/api/topics/nonexistent/deep-read",
            json={"paper_id": "2501.00001"},
        )
        assert resp.status_code == 404
        assert "Topic not found" in resp.json()["detail"]

    @patch("paper_tracker.server.Storage")
    def test_paper_not_found(self, MockStorage, client):
        mock_store = MagicMock()
        mock_store.get_arxiv.return_value = None
        MockStorage.return_value = mock_store

        resp = client.post(
            "/api/topics/test-topic/deep-read",
            json={"paper_id": "nonexistent"},
        )
        assert resp.status_code == 404
        assert "Paper not found" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# GET /api/topics/{topic_id}/deep-read
# ---------------------------------------------------------------------------

class TestListDeepReadSessions:
    def test_list_all(self, client):
        import paper_tracker.server as srv
        srv._registry.list_deep_read_sessions.return_value = [_FAKE_DR_SESSION]

        resp = client.get("/api/topics/test-topic/deep-read")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["sessions"]) == 1
        assert data["sessions"][0]["id"] == "dr-abc12345"

    def test_filter_by_paper_id(self, client):
        import paper_tracker.server as srv
        srv._registry.list_deep_read_sessions.return_value = [_FAKE_DR_SESSION]

        resp = client.get("/api/topics/test-topic/deep-read?paper_id=2501.00001")
        assert resp.status_code == 200
        srv._registry.list_deep_read_sessions.assert_called_with("test-topic", paper_id="2501.00001")

    def test_topic_not_found(self, client):
        import paper_tracker.server as srv
        srv._registry.get_topic.return_value = None

        resp = client.get("/api/topics/nonexistent/deep-read")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/topics/{topic_id}/deep-read/{session_id}
# ---------------------------------------------------------------------------

class TestGetDeepReadSession:
    def test_get_with_messages(self, client):
        import paper_tracker.server as srv
        srv._registry.get_deep_read_session.return_value = _FAKE_DR_SESSION_COMPLETED
        srv._registry.list_deep_read_messages.return_value = [
            {"id": "dm-1", "role": "user", "content": "Q?", "status": "completed"},
        ]

        resp = client.get("/api/topics/test-topic/deep-read/dr-abc12345")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert len(data["messages"]) == 1

    def test_not_found(self, client):
        import paper_tracker.server as srv
        srv._registry.get_deep_read_session.return_value = None

        resp = client.get("/api/topics/test-topic/deep-read/dr-nonexistent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/topics/{topic_id}/deep-read/{session_id}/progress
# ---------------------------------------------------------------------------

class TestGetDeepReadProgress:
    def test_from_in_memory(self, client):
        _deep_read_progress["dr-abc12345"] = {
            "status": "running",
            "sections_done": ["motivation_analysis"],
            "current_section": "method_analysis",
        }

        resp = client.get("/api/topics/test-topic/deep-read/dr-abc12345/progress")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "running"
        assert "motivation_analysis" in data["sections_done"]
        assert data["current_section"] == "method_analysis"

    def test_completed_clears_cache(self, client):
        _deep_read_progress["dr-abc12345"] = {
            "status": "completed",
            "sections_done": ["motivation_analysis", "method_analysis"],
            "current_section": None,
        }

        resp = client.get("/api/topics/test-topic/deep-read/dr-abc12345/progress")
        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"
        # Should be cleared from cache
        assert "dr-abc12345" not in _deep_read_progress

    def test_fallback_to_db(self, client):
        import paper_tracker.server as srv
        srv._registry.get_deep_read_session.return_value = {
            **_FAKE_DR_SESSION_COMPLETED,
            "summary_card_json": {"key_innovations": ["A"]},
        }

        resp = client.get("/api/topics/test-topic/deep-read/dr-abc12345/progress")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        # Should derive sections_done from non-empty fields
        assert "motivation_analysis" in data["sections_done"]
        assert "summary_card_json" in data["sections_done"]

    def test_fallback_not_found(self, client):
        import paper_tracker.server as srv
        srv._registry.get_deep_read_session.return_value = None

        resp = client.get("/api/topics/test-topic/deep-read/dr-nonexistent/progress")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /api/topics/{topic_id}/deep-read/{session_id}
# ---------------------------------------------------------------------------

class TestDeleteDeepReadSession:
    def test_delete_success(self, client):
        import paper_tracker.server as srv
        srv._registry.delete_deep_read_session.return_value = True

        resp = client.delete("/api/topics/test-topic/deep-read/dr-abc12345")
        assert resp.status_code == 204

    def test_delete_not_found(self, client):
        import paper_tracker.server as srv
        srv._registry.delete_deep_read_session.return_value = False

        resp = client.delete("/api/topics/test-topic/deep-read/dr-nonexistent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# PUT /api/topics/{topic_id}/deep-read/{session_id}/notes
# ---------------------------------------------------------------------------

class TestUpdateDeepReadNotes:
    def test_save_notes(self, client):
        import paper_tracker.server as srv
        srv._registry.get_deep_read_session.return_value = _FAKE_DR_SESSION_COMPLETED

        resp = client.put(
            "/api/topics/test-topic/deep-read/dr-abc12345/notes",
            json={"notes": "Great paper!"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "saved"
        srv._registry.update_deep_read_session.assert_called_once_with(
            "test-topic", "dr-abc12345", {"user_notes": "Great paper!"},
        )

    def test_session_not_found(self, client):
        import paper_tracker.server as srv
        srv._registry.get_deep_read_session.return_value = None

        resp = client.put(
            "/api/topics/test-topic/deep-read/dr-nonexistent/notes",
            json={"notes": "X"},
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /api/topics/{topic_id}/deep-read/{session_id}/messages
# ---------------------------------------------------------------------------

class TestSendDeepReadMessage:
    @patch("paper_tracker.server._brainstorm_executor")
    def test_send_message_202(self, mock_executor, client):
        import paper_tracker.server as srv
        srv._registry.get_deep_read_session.return_value = _FAKE_DR_SESSION_COMPLETED
        srv._registry.add_deep_read_message.side_effect = [
            {"id": "dm-user1", "role": "user", "content": "Q?", "status": "completed"},
            {"id": "dm-asst1", "role": "assistant", "content": "", "status": "pending"},
        ]

        resp = client.post(
            "/api/topics/test-topic/deep-read/dr-abc12345/messages",
            json={"content": "What is the key insight?"},
        )
        assert resp.status_code == 202
        data = resp.json()
        assert data["user_msg_id"] == "dm-user1"
        assert data["assistant_msg_id"] == "dm-asst1"
        mock_executor.submit.assert_called_once()

    def test_send_on_running_session_409(self, client):
        import paper_tracker.server as srv
        srv._registry.get_deep_read_session.return_value = _FAKE_DR_SESSION  # status=running

        resp = client.post(
            "/api/topics/test-topic/deep-read/dr-abc12345/messages",
            json={"content": "Q?"},
        )
        assert resp.status_code == 409
        assert "completed" in resp.json()["detail"]

    def test_send_empty_content_422(self, client):
        import paper_tracker.server as srv
        srv._registry.get_deep_read_session.return_value = _FAKE_DR_SESSION_COMPLETED

        resp = client.post(
            "/api/topics/test-topic/deep-read/dr-abc12345/messages",
            json={"content": "   "},
        )
        assert resp.status_code == 422

    def test_session_not_found(self, client):
        import paper_tracker.server as srv
        srv._registry.get_deep_read_session.return_value = None

        resp = client.post(
            "/api/topics/test-topic/deep-read/dr-nonexistent/messages",
            json={"content": "Q?"},
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/topics/{id}/deep-read/{sid}/messages/{mid}/progress
# ---------------------------------------------------------------------------

class TestGetDeepReadMessageProgress:
    def test_from_in_memory(self, client):
        _deep_read_qa_progress["dm-asst1"] = {"status": "generating"}

        resp = client.get(
            "/api/topics/test-topic/deep-read/dr-abc12345/messages/dm-asst1/progress"
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "generating"

    def test_completed_clears_cache(self, client):
        _deep_read_qa_progress["dm-asst1"] = {"status": "completed"}

        resp = client.get(
            "/api/topics/test-topic/deep-read/dr-abc12345/messages/dm-asst1/progress"
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"
        assert "dm-asst1" not in _deep_read_qa_progress

    def test_fallback_to_db(self, client):
        import paper_tracker.server as srv
        srv._registry.get_deep_read_message.return_value = {
            "id": "dm-asst1", "status": "completed",
        }

        resp = client.get(
            "/api/topics/test-topic/deep-read/dr-abc12345/messages/dm-asst1/progress"
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"

    def test_not_found(self, client):
        import paper_tracker.server as srv
        srv._registry.get_deep_read_message.return_value = None

        resp = client.get(
            "/api/topics/test-topic/deep-read/dr-abc12345/messages/dm-nonexistent/progress"
        )
        assert resp.status_code == 404
