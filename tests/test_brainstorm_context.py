"""Tests for brainstorm._load_topic_insights choosing a specific session."""

from __future__ import annotations

from unittest.mock import MagicMock

from paper_tracker.brainstorm import _load_topic_insights


def _session_dir(root, topic, sid, text):
    d = root / topic / "sessions" / sid
    d.mkdir(parents=True, exist_ok=True)
    (d / "insights.md").write_text(text, encoding="utf-8")
    return str(d / "insights.md")


def test_defaults_to_latest_session(tmp_path):
    latest = _session_dir(tmp_path, "t1", "2026-01-01_001", "LATEST MEMO")
    reg = MagicMock()
    reg.get_latest_session.return_value = {"id": "2026-01-01_001", "insights_path": latest}

    out = _load_topic_insights(str(tmp_path), "t1", reg)
    assert "LATEST MEMO" in out
    reg.get_session.assert_not_called()


def test_loads_chosen_session(tmp_path):
    _session_dir(tmp_path, "t1", "2026-01-01_001", "LATEST MEMO")
    chosen = _session_dir(tmp_path, "t1", "2026-01-01_002", "CHOSEN MEMO")
    reg = MagicMock()
    reg.get_session.return_value = {"id": "2026-01-01_002", "insights_path": chosen}

    out = _load_topic_insights(str(tmp_path), "t1", reg, "2026-01-01_002")
    assert "CHOSEN MEMO" in out
    assert "selected session" in out
    reg.get_session.assert_called_once_with("t1", "2026-01-01_002")


def test_no_registry_returns_empty():
    assert _load_topic_insights("/x", "t1", None) == ""


def test_missing_insights_path_returns_empty(tmp_path):
    reg = MagicMock()
    reg.get_session.return_value = {"id": "s", "insights_path": ""}
    assert _load_topic_insights(str(tmp_path), "t1", reg, "s") == ""
