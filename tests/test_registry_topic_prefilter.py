"""Registry tests for the topic prefilter fields (prefilter_enabled, prefilter_criteria)."""

from __future__ import annotations

import sqlite3

import pytest

from paper_tracker.registry import Registry


@pytest.fixture()
def reg(tmp_path):
    r = Registry(str(tmp_path))
    yield r
    r.close()


def _base_topic(**overrides) -> dict:
    base = {
        "id": "t1",
        "name": "Topic 1",
        "arxiv_keywords": ["x"],
        "arxiv_categories": ["cs.CL"],
        "github_keywords": [],
    }
    base.update(overrides)
    return base


class TestCreateTopicPrefilter:
    def test_default_prefilter_enabled_true(self, reg: Registry):
        t = reg.create_topic(_base_topic())
        assert t["prefilter_enabled"] is True

    def test_default_prefilter_criteria_empty(self, reg: Registry):
        t = reg.create_topic(_base_topic())
        assert t["prefilter_criteria"] == ""

    def test_explicit_prefilter_disabled(self, reg: Registry):
        t = reg.create_topic(_base_topic(prefilter_enabled=False))
        assert t["prefilter_enabled"] is False

    def test_explicit_prefilter_criteria_stored(self, reg: Registry):
        t = reg.create_topic(_base_topic(
            prefilter_criteria="Exclude GAN papers.",
        ))
        assert t["prefilter_criteria"] == "Exclude GAN papers."


class TestUpdateTopicPrefilter:
    def test_update_prefilter_enabled(self, reg: Registry):
        reg.create_topic(_base_topic())
        reg.update_topic("t1", {"prefilter_enabled": False})
        assert reg.get_topic("t1")["prefilter_enabled"] is False

    def test_update_prefilter_criteria(self, reg: Registry):
        reg.create_topic(_base_topic())
        reg.update_topic("t1", {"prefilter_criteria": "New rule."})
        assert reg.get_topic("t1")["prefilter_criteria"] == "New rule."


class TestMigration:
    def test_migration_adds_prefilter_columns(self, tmp_path):
        """A DB without prefilter columns should get them added on open."""
        db_path = tmp_path / "registry.db"
        conn = sqlite3.connect(str(db_path))
        # Minimal schema lacking prefilter fields
        conn.execute("""CREATE TABLE IF NOT EXISTS topics (
            id TEXT PRIMARY KEY, name TEXT, description TEXT DEFAULT '',
            arxiv_keywords TEXT, arxiv_categories TEXT,
            arxiv_lookback_days INTEGER DEFAULT 2,
            github_keywords TEXT, github_lookback_days INTEGER DEFAULT 7,
            schedule_cron TEXT DEFAULT '', enabled INTEGER DEFAULT 1,
            created_at TEXT
        )""")
        conn.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, topic_id TEXT, started_at TEXT, finished_at TEXT, paper_count INTEGER DEFAULT 0, repo_count INTEGER DEFAULT 0, status TEXT, report_path TEXT, insights_path TEXT)")
        conn.execute("CREATE TABLE IF NOT EXISTS brainstorm_sessions (id TEXT PRIMARY KEY, topic_id TEXT, mode TEXT, user_idea TEXT, status TEXT, started_at TEXT, finished_at TEXT, ideas_json TEXT, literature_result TEXT, logic_result TEXT, code_result TEXT, run_code_verification INTEGER DEFAULT 0)")
        conn.execute("CREATE TABLE IF NOT EXISTS research_plans (id TEXT PRIMARY KEY, topic_id TEXT, brainstorm_session_id TEXT, idea_title TEXT, idea_json TEXT, status TEXT, started_at TEXT, finished_at TEXT, introduction TEXT, related_work TEXT, methodology TEXT, experimental_design TEXT, expected_results TEXT, timeline TEXT, review TEXT, full_markdown TEXT)")
        conn.commit()
        conn.close()

        # Opening via Registry triggers auto-migration
        r = Registry(str(tmp_path))
        try:
            cols = {row[1] for row in r._conn.execute("PRAGMA table_info(topics)").fetchall()}
            assert "prefilter_enabled" in cols
            assert "prefilter_criteria" in cols
        finally:
            r.close()
