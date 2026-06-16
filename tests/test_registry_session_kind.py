"""Registry tests for the sessions.kind field ('run' | 'manual')."""

from __future__ import annotations

import sqlite3

import pytest

from paper_tracker.registry import Registry


@pytest.fixture()
def reg(tmp_path):
    r = Registry(str(tmp_path))
    yield r
    r.close()


def _topic(reg: Registry) -> None:
    reg.create_topic({
        "id": "t1", "name": "Topic 1",
        "arxiv_keywords": ["x"], "arxiv_categories": ["cs.CL"],
        "github_keywords": [],
    })


class TestSessionKind:
    def test_default_kind_is_run(self, reg: Registry):
        _topic(reg)
        s = reg.create_session("t1")
        assert s["kind"] == "run"

    def test_manual_kind_stored(self, reg: Registry):
        _topic(reg)
        s = reg.create_session("t1", kind="manual")
        assert s["kind"] == "manual"

    def test_list_sessions_includes_kind(self, reg: Registry):
        _topic(reg)
        reg.create_session("t1", kind="manual")
        rows = reg.list_sessions("t1")
        assert rows and rows[0]["kind"] == "manual"


class TestMigration:
    def test_migration_adds_kind_column(self, tmp_path):
        """A sessions table without `kind` should get it on open (default 'run')."""
        conn = sqlite3.connect(str(tmp_path / "registry.db"))
        conn.execute(
            "CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, topic_id TEXT, "
            "started_at TEXT, finished_at TEXT, paper_count INTEGER DEFAULT 0, "
            "repo_count INTEGER DEFAULT 0, status TEXT, report_path TEXT, insights_path TEXT)"
        )
        conn.execute("INSERT INTO sessions (id, topic_id, status) VALUES ('2026-01-01_001','t1','completed')")
        conn.commit()
        conn.close()

        r = Registry(str(tmp_path))
        try:
            cols = {row[1] for row in r._conn.execute("PRAGMA table_info(sessions)").fetchall()}
            assert "kind" in cols
            row = r._conn.execute("SELECT kind FROM sessions WHERE id='2026-01-01_001'").fetchone()
            assert row[0] == "run"  # existing rows default to 'run'
        finally:
            r.close()
