"""Unit tests for registry.py — deep_read_sessions / deep_read_messages CRUD."""

from __future__ import annotations

import json

import pytest

from paper_tracker.registry import Registry


@pytest.fixture()
def reg(tmp_path):
    r = Registry(str(tmp_path))
    # Create a topic to satisfy foreign key references
    r.create_topic({
        "id": "tp-test",
        "name": "Test Topic",
        "arxiv_keywords": ["test"],
        "arxiv_categories": [],
        "github_keywords": [],
    })
    yield r
    r.close()


# ---------------------------------------------------------------
# create_deep_read_session
# ---------------------------------------------------------------

class TestCreateDeepReadSession:
    def test_basic_create(self, reg: Registry):
        s = reg.create_deep_read_session("tp-test", "2501.00001", "My Paper Title")
        assert s is not None
        assert s["id"].startswith("dr-")
        assert s["topic_id"] == "tp-test"
        assert s["paper_id"] == "2501.00001"
        assert s["paper_title"] == "My Paper Title"
        assert s["status"] == "running"
        assert s["started_at"] is not None

    def test_default_fields_empty(self, reg: Registry):
        s = reg.create_deep_read_session("tp-test", "2501.00001")
        assert s["paper_title"] == ""
        assert s["motivation_analysis"] == ""
        assert s["method_analysis"] == ""
        assert s["experiment_analysis"] == ""
        assert s["contribution_analysis"] == ""
        assert s["summary_card_json"] == {}
        assert s["user_notes"] == ""

    def test_unique_ids(self, reg: Registry):
        s1 = reg.create_deep_read_session("tp-test", "2501.00001")
        s2 = reg.create_deep_read_session("tp-test", "2501.00001")
        assert s1["id"] != s2["id"]


# ---------------------------------------------------------------
# get_deep_read_session
# ---------------------------------------------------------------

class TestGetDeepReadSession:
    def test_get_existing(self, reg: Registry):
        s = reg.create_deep_read_session("tp-test", "2501.00001", "Title")
        got = reg.get_deep_read_session("tp-test", s["id"])
        assert got is not None
        assert got["id"] == s["id"]
        assert got["paper_title"] == "Title"

    def test_get_nonexistent(self, reg: Registry):
        assert reg.get_deep_read_session("tp-test", "dr-nonexistent") is None

    def test_wrong_topic_returns_none(self, reg: Registry):
        s = reg.create_deep_read_session("tp-test", "2501.00001")
        assert reg.get_deep_read_session("other-topic", s["id"]) is None

    def test_summary_card_json_parsed(self, reg: Registry):
        s = reg.create_deep_read_session("tp-test", "2501.00001")
        card = {"key_innovations": ["Fast training"], "strengths": ["Simple"]}
        reg.update_deep_read_session("tp-test", s["id"], {
            "summary_card_json": json.dumps(card),
        })
        got = reg.get_deep_read_session("tp-test", s["id"])
        assert isinstance(got["summary_card_json"], dict)
        assert got["summary_card_json"]["key_innovations"] == ["Fast training"]


# ---------------------------------------------------------------
# list_deep_read_sessions
# ---------------------------------------------------------------

class TestListDeepReadSessions:
    def test_empty(self, reg: Registry):
        sessions = reg.list_deep_read_sessions("tp-test")
        assert sessions == []

    def test_list_all(self, reg: Registry):
        reg.create_deep_read_session("tp-test", "2501.00001")
        reg.create_deep_read_session("tp-test", "2501.00002")
        sessions = reg.list_deep_read_sessions("tp-test")
        assert len(sessions) == 2

    def test_filter_by_paper_id(self, reg: Registry):
        reg.create_deep_read_session("tp-test", "2501.00001")
        reg.create_deep_read_session("tp-test", "2501.00002")
        reg.create_deep_read_session("tp-test", "2501.00001")

        filtered = reg.list_deep_read_sessions("tp-test", paper_id="2501.00001")
        assert len(filtered) == 2
        assert all(s["paper_id"] == "2501.00001" for s in filtered)

    def test_ordered_desc(self, reg: Registry):
        s1 = reg.create_deep_read_session("tp-test", "2501.00001")
        s2 = reg.create_deep_read_session("tp-test", "2501.00002")
        sessions = reg.list_deep_read_sessions("tp-test")
        # Most recent first
        assert sessions[0]["id"] == s2["id"]
        assert sessions[1]["id"] == s1["id"]

    def test_summary_card_parsed_in_list(self, reg: Registry):
        s = reg.create_deep_read_session("tp-test", "2501.00001")
        card = {"strengths": ["Good"]}
        reg.update_deep_read_session("tp-test", s["id"], {
            "summary_card_json": json.dumps(card),
        })
        sessions = reg.list_deep_read_sessions("tp-test")
        assert isinstance(sessions[0]["summary_card_json"], dict)


# ---------------------------------------------------------------
# update_deep_read_session
# ---------------------------------------------------------------

class TestUpdateDeepReadSession:
    def test_update_status(self, reg: Registry):
        s = reg.create_deep_read_session("tp-test", "2501.00001")
        reg.update_deep_read_session("tp-test", s["id"], {"status": "completed"})
        got = reg.get_deep_read_session("tp-test", s["id"])
        assert got["status"] == "completed"

    def test_update_analysis_fields(self, reg: Registry):
        s = reg.create_deep_read_session("tp-test", "2501.00001")
        reg.update_deep_read_session("tp-test", s["id"], {
            "motivation_analysis": "# Motivation\nGood question.",
            "method_analysis": "# Method\nNovel approach.",
        })
        got = reg.get_deep_read_session("tp-test", s["id"])
        assert "Good question" in got["motivation_analysis"]
        assert "Novel approach" in got["method_analysis"]

    def test_update_user_notes(self, reg: Registry):
        s = reg.create_deep_read_session("tp-test", "2501.00001")
        reg.update_deep_read_session("tp-test", s["id"], {"user_notes": "Interesting paper!"})
        got = reg.get_deep_read_session("tp-test", s["id"])
        assert got["user_notes"] == "Interesting paper!"

    def test_update_summary_card_as_dict(self, reg: Registry):
        s = reg.create_deep_read_session("tp-test", "2501.00001")
        card = {"key_innovations": ["A"], "strengths": ["B"]}
        reg.update_deep_read_session("tp-test", s["id"], {"summary_card_json": card})
        got = reg.get_deep_read_session("tp-test", s["id"])
        assert got["summary_card_json"] == card

    def test_update_ignores_unknown_fields(self, reg: Registry):
        s = reg.create_deep_read_session("tp-test", "2501.00001")
        reg.update_deep_read_session("tp-test", s["id"], {"unknown_field": "value"})
        got = reg.get_deep_read_session("tp-test", s["id"])
        assert "unknown_field" not in got

    def test_update_multiple_fields(self, reg: Registry):
        s = reg.create_deep_read_session("tp-test", "2501.00001")
        reg.update_deep_read_session("tp-test", s["id"], {
            "status": "completed",
            "finished_at": "2026-03-16T12:00:00",
            "motivation_analysis": "M",
            "method_analysis": "Me",
            "experiment_analysis": "E",
            "contribution_analysis": "C",
        })
        got = reg.get_deep_read_session("tp-test", s["id"])
        assert got["status"] == "completed"
        assert got["finished_at"] == "2026-03-16T12:00:00"
        assert got["motivation_analysis"] == "M"
        assert got["contribution_analysis"] == "C"


# ---------------------------------------------------------------
# delete_deep_read_session (cascade delete messages)
# ---------------------------------------------------------------

class TestDeleteDeepReadSession:
    def test_delete_existing(self, reg: Registry):
        s = reg.create_deep_read_session("tp-test", "2501.00001")
        assert reg.delete_deep_read_session("tp-test", s["id"]) is True
        assert reg.get_deep_read_session("tp-test", s["id"]) is None

    def test_delete_nonexistent(self, reg: Registry):
        assert reg.delete_deep_read_session("tp-test", "dr-nonexistent") is False

    def test_cascade_deletes_messages(self, reg: Registry):
        s = reg.create_deep_read_session("tp-test", "2501.00001")
        reg.add_deep_read_message("tp-test", s["id"], "user", "Question?")
        reg.add_deep_read_message("tp-test", s["id"], "assistant", "Answer.")
        messages_before = reg.list_deep_read_messages("tp-test", s["id"])
        assert len(messages_before) == 2

        reg.delete_deep_read_session("tp-test", s["id"])
        messages_after = reg.list_deep_read_messages("tp-test", s["id"])
        assert len(messages_after) == 0


# ---------------------------------------------------------------
# add_deep_read_message
# ---------------------------------------------------------------

class TestAddDeepReadMessage:
    def test_add_user_message(self, reg: Registry):
        s = reg.create_deep_read_session("tp-test", "2501.00001")
        msg = reg.add_deep_read_message("tp-test", s["id"], "user", "What is the method?")
        assert msg is not None
        assert msg["id"].startswith("dm-")
        assert msg["role"] == "user"
        assert msg["content"] == "What is the method?"
        assert msg["status"] == "completed"
        assert msg["created_at"] is not None

    def test_add_assistant_message_pending(self, reg: Registry):
        s = reg.create_deep_read_session("tp-test", "2501.00001")
        msg = reg.add_deep_read_message("tp-test", s["id"], "assistant", "", status="pending")
        assert msg["role"] == "assistant"
        assert msg["status"] == "pending"
        assert msg["content"] == ""


# ---------------------------------------------------------------
# list_deep_read_messages
# ---------------------------------------------------------------

class TestListDeepReadMessages:
    def test_empty(self, reg: Registry):
        s = reg.create_deep_read_session("tp-test", "2501.00001")
        messages = reg.list_deep_read_messages("tp-test", s["id"])
        assert messages == []

    def test_ordered_asc(self, reg: Registry):
        s = reg.create_deep_read_session("tp-test", "2501.00001")
        m1 = reg.add_deep_read_message("tp-test", s["id"], "user", "Q1")
        m2 = reg.add_deep_read_message("tp-test", s["id"], "assistant", "A1")
        m3 = reg.add_deep_read_message("tp-test", s["id"], "user", "Q2")
        messages = reg.list_deep_read_messages("tp-test", s["id"])
        assert len(messages) == 3
        assert messages[0]["id"] == m1["id"]
        assert messages[1]["id"] == m2["id"]
        assert messages[2]["id"] == m3["id"]


# ---------------------------------------------------------------
# update_deep_read_message
# ---------------------------------------------------------------

class TestUpdateDeepReadMessage:
    def test_update_content_and_status(self, reg: Registry):
        s = reg.create_deep_read_session("tp-test", "2501.00001")
        msg = reg.add_deep_read_message("tp-test", s["id"], "assistant", "", status="pending")
        reg.update_deep_read_message(msg["id"], {
            "content": "Here is the answer.",
            "status": "completed",
        })
        got = reg.get_deep_read_message(msg["id"])
        assert got["content"] == "Here is the answer."
        assert got["status"] == "completed"

    def test_update_ignores_unknown(self, reg: Registry):
        s = reg.create_deep_read_session("tp-test", "2501.00001")
        msg = reg.add_deep_read_message("tp-test", s["id"], "user", "Q")
        reg.update_deep_read_message(msg["id"], {"role": "admin"})
        got = reg.get_deep_read_message(msg["id"])
        assert got["role"] == "user"  # unchanged


# ---------------------------------------------------------------
# get_deep_read_message
# ---------------------------------------------------------------

class TestGetDeepReadMessage:
    def test_get_existing(self, reg: Registry):
        s = reg.create_deep_read_session("tp-test", "2501.00001")
        msg = reg.add_deep_read_message("tp-test", s["id"], "user", "Hello")
        got = reg.get_deep_read_message(msg["id"])
        assert got is not None
        assert got["content"] == "Hello"

    def test_get_nonexistent(self, reg: Registry):
        assert reg.get_deep_read_message("dm-nonexistent") is None


# ---------------------------------------------------------------
# delete_topic cascade
# ---------------------------------------------------------------

class TestDeleteTopicCascade:
    def test_delete_topic_removes_deep_read(self, reg: Registry):
        s = reg.create_deep_read_session("tp-test", "2501.00001")
        reg.add_deep_read_message("tp-test", s["id"], "user", "Q")
        reg.add_deep_read_message("tp-test", s["id"], "assistant", "A")

        reg.delete_topic("tp-test")

        # Everything should be gone
        assert reg.get_deep_read_session("tp-test", s["id"]) is None
        assert reg.list_deep_read_sessions("tp-test") == []
        assert reg.list_deep_read_messages("tp-test", s["id"]) == []


# ---------------------------------------------------------------
# recover_stale_tasks
# ---------------------------------------------------------------

class TestRecoverStaleTasks:
    def test_recovers_running_sessions(self, reg: Registry):
        s = reg.create_deep_read_session("tp-test", "2501.00001")
        assert s["status"] == "running"

        counts = reg.recover_stale_tasks()
        assert counts.get("deep_read_sessions", 0) >= 1

        got = reg.get_deep_read_session("tp-test", s["id"])
        assert got["status"] == "failed"
        assert got["finished_at"] is not None

    def test_recovers_pending_messages(self, reg: Registry):
        s = reg.create_deep_read_session("tp-test", "2501.00001")
        msg = reg.add_deep_read_message("tp-test", s["id"], "assistant", "", status="pending")

        counts = reg.recover_stale_tasks()
        assert counts.get("deep_read_messages", 0) >= 1

        got = reg.get_deep_read_message(msg["id"])
        assert got["status"] == "failed"

    def test_no_recovery_when_completed(self, reg: Registry):
        s = reg.create_deep_read_session("tp-test", "2501.00001")
        reg.update_deep_read_session("tp-test", s["id"], {"status": "completed"})
        msg = reg.add_deep_read_message("tp-test", s["id"], "user", "Q")  # status=completed by default

        counts = reg.recover_stale_tasks()
        assert counts.get("deep_read_sessions", 0) == 0
        assert counts.get("deep_read_messages", 0) == 0
