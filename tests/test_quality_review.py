"""Unit tests for discovery quality review and registry quality fields."""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest

from paper_tracker.discovery import review_discovery_report
from paper_tracker.registry import Registry


_CFG = {
    "paths": {"data_dir": "/tmp/test"},
    "summarizer": {"claude_path": "claude", "claude_model": "opus"},
}


@pytest.fixture()
def reg(tmp_path):
    r = Registry(str(tmp_path))
    yield r
    r.close()


# ---------------------------------------------------------------
# Registry: quality fields
# ---------------------------------------------------------------

class TestRegistryQualityFields:
    def test_default_quality_score_negative_one(self, reg: Registry):
        dr = reg.create_discovery_report("trending")
        assert dr["quality_score"] == -1

    def test_default_quality_flags_empty_list(self, reg: Registry):
        dr = reg.create_discovery_report("trending")
        assert dr["quality_flags"] == []

    def test_update_quality_score(self, reg: Registry):
        dr = reg.create_discovery_report("trending")
        reg.update_discovery_report(dr["id"], {"quality_score": 85})
        got = reg.get_discovery_report(dr["id"])
        assert got["quality_score"] == 85

    def test_update_quality_flags_as_list(self, reg: Registry):
        dr = reg.create_discovery_report("trending")
        flags = [{"issue": "Too few themes", "severity": "medium"}]
        reg.update_discovery_report(dr["id"], {"quality_flags": flags})
        got = reg.get_discovery_report(dr["id"])
        assert got["quality_flags"] == flags

    def test_quality_flags_serialized_as_json(self, reg: Registry):
        dr = reg.create_discovery_report("trending")
        flags = [
            {"issue": "Missing cross-theme observations", "severity": "high"},
            {"issue": "Vague descriptions", "severity": "low"},
        ]
        reg.update_discovery_report(dr["id"], {"quality_flags": flags})
        got = reg.get_discovery_report(dr["id"])
        assert len(got["quality_flags"]) == 2
        assert got["quality_flags"][0]["severity"] == "high"

    def test_update_quality_score_and_flags_together(self, reg: Registry):
        dr = reg.create_discovery_report("trending")
        reg.update_discovery_report(dr["id"], {
            "quality_score": 42,
            "quality_flags": [{"issue": "Low quality", "severity": "high"}],
        })
        got = reg.get_discovery_report(dr["id"])
        assert got["quality_score"] == 42
        assert len(got["quality_flags"]) == 1

    def test_list_reports_includes_quality(self, reg: Registry):
        dr = reg.create_discovery_report("trending")
        reg.update_discovery_report(dr["id"], {"quality_score": 75})
        reports = reg.list_discovery_reports()
        assert reports[0]["quality_score"] == 75

    def test_latest_report_includes_quality(self, reg: Registry):
        dr = reg.create_discovery_report("trending")
        reg.update_discovery_report(dr["id"], {
            "status": "completed",
            "finished_at": "2026-03-05T10:00:00",
            "quality_score": 90,
        })
        latest = reg.get_latest_discovery_report("trending")
        assert latest["quality_score"] == 90


# ---------------------------------------------------------------
# Registry: migration
# ---------------------------------------------------------------

class TestRegistryMigration:
    def test_migration_adds_quality_columns(self, tmp_path):
        """If DB exists without quality columns, migration should add them."""
        import sqlite3
        db_path = tmp_path / "registry.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""CREATE TABLE IF NOT EXISTS discovery_reports (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            status TEXT DEFAULT 'running',
            started_at TEXT,
            finished_at TEXT,
            content TEXT DEFAULT '',
            papers_json TEXT DEFAULT '[]',
            paper_count INTEGER DEFAULT 0,
            source_stats TEXT DEFAULT '{}'
        )""")
        # Also create minimal other tables to satisfy schema
        conn.execute("CREATE TABLE IF NOT EXISTS topics (id TEXT PRIMARY KEY, name TEXT, description TEXT DEFAULT '', arxiv_keywords TEXT, arxiv_categories TEXT, arxiv_lookback_days INTEGER DEFAULT 2, github_keywords TEXT, github_lookback_days INTEGER DEFAULT 7, schedule_cron TEXT DEFAULT '', enabled INTEGER DEFAULT 1, created_at TEXT)")
        conn.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, topic_id TEXT, started_at TEXT, finished_at TEXT, paper_count INTEGER DEFAULT 0, repo_count INTEGER DEFAULT 0, status TEXT, report_path TEXT, insights_path TEXT)")
        conn.execute("CREATE TABLE IF NOT EXISTS brainstorm_sessions (id TEXT PRIMARY KEY, topic_id TEXT, mode TEXT, user_idea TEXT, status TEXT, started_at TEXT, finished_at TEXT, ideas_json TEXT, literature_result TEXT, logic_result TEXT, code_result TEXT, run_code_verification INTEGER DEFAULT 0)")
        conn.execute("CREATE TABLE IF NOT EXISTS research_plans (id TEXT PRIMARY KEY, topic_id TEXT, brainstorm_session_id TEXT, idea_title TEXT, idea_json TEXT, status TEXT, started_at TEXT, finished_at TEXT, introduction TEXT, related_work TEXT, methodology TEXT, experimental_design TEXT, expected_results TEXT, timeline TEXT, review TEXT, full_markdown TEXT)")
        conn.commit()
        conn.close()

        # Now open with Registry — should auto-migrate
        r = Registry(str(tmp_path))
        dr = r.create_discovery_report("trending")
        assert dr["quality_score"] == -1
        assert dr["quality_flags"] == []
        r.close()


# ---------------------------------------------------------------
# review_discovery_report
# ---------------------------------------------------------------

class TestReviewDiscoveryReport:
    def test_report_not_found(self, reg: Registry):
        result = review_discovery_report(reg, "dr-nonexistent", _CFG)
        assert result.get("error") == "Report not found"

    def test_not_completed(self, reg: Registry):
        dr = reg.create_discovery_report("trending")
        result = review_discovery_report(reg, dr["id"], _CFG)
        assert result.get("error") == "Can only review completed reports"

    def test_empty_content_scores_zero(self, reg: Registry):
        dr = reg.create_discovery_report("trending")
        reg.update_discovery_report(dr["id"], {
            "status": "completed",
            "finished_at": "2026-03-05T10:00:00",
            "content": "",
        })
        result = review_discovery_report(reg, dr["id"], _CFG)
        assert result["quality_score"] == 0
        got = reg.get_discovery_report(dr["id"])
        assert got["quality_score"] == 0

    @patch("paper_tracker.discovery.call_cli")
    def test_happy_path_trending(self, mock_cli, reg: Registry):
        dr = reg.create_discovery_report("trending")
        reg.update_discovery_report(dr["id"], {
            "status": "completed",
            "finished_at": "2026-03-05T10:00:00",
            "content": "# Trending\n\nSome good content",
            "paper_count": 100,
            "source_stats": {"arxiv": 70, "huggingface": 30},
        })
        mock_cli.return_value = json.dumps({
            "quality_score": 82,
            "flags": [{"issue": "Could use more sources", "severity": "low"}],
            "summary": "Good overall report.",
        })
        result = review_discovery_report(reg, dr["id"], _CFG)
        assert result["quality_score"] == 82
        assert len(result["flags"]) == 1
        assert result["summary"] == "Good overall report."
        # Check persisted
        got = reg.get_discovery_report(dr["id"])
        assert got["quality_score"] == 82

    @patch("paper_tracker.discovery.call_cli")
    def test_happy_path_math(self, mock_cli, reg: Registry):
        dr = reg.create_discovery_report("math")
        reg.update_discovery_report(dr["id"], {
            "status": "completed",
            "finished_at": "2026-03-05T10:00:00",
            "content": "# Math Insights\n\nAnalysis",
            "paper_count": 25,
        })
        mock_cli.return_value = json.dumps({
            "quality_score": 91,
            "flags": [],
            "summary": "Excellent analysis.",
        })
        result = review_discovery_report(reg, dr["id"], _CFG)
        assert result["quality_score"] == 91
        assert result["flags"] == []

    @patch("paper_tracker.discovery.call_cli")
    def test_llm_returns_none(self, mock_cli, reg: Registry):
        dr = reg.create_discovery_report("trending")
        reg.update_discovery_report(dr["id"], {
            "status": "completed",
            "finished_at": "2026-03-05T10:00:00",
            "content": "Some content",
        })
        mock_cli.return_value = None
        result = review_discovery_report(reg, dr["id"], _CFG)
        assert result["quality_score"] == -1

    @patch("paper_tracker.discovery.call_cli")
    def test_llm_returns_malformed_json(self, mock_cli, reg: Registry):
        dr = reg.create_discovery_report("trending")
        reg.update_discovery_report(dr["id"], {
            "status": "completed",
            "finished_at": "2026-03-05T10:00:00",
            "content": "Some content",
        })
        mock_cli.return_value = "This is not JSON at all"
        result = review_discovery_report(reg, dr["id"], _CFG)
        assert result["quality_score"] == -1

    @patch("paper_tracker.discovery.call_cli")
    def test_llm_returns_json_with_markdown_fences(self, mock_cli, reg: Registry):
        dr = reg.create_discovery_report("trending")
        reg.update_discovery_report(dr["id"], {
            "status": "completed",
            "finished_at": "2026-03-05T10:00:00",
            "content": "Some content",
        })
        mock_cli.return_value = '```json\n{"quality_score": 75, "flags": [], "summary": "OK"}\n```'
        result = review_discovery_report(reg, dr["id"], _CFG)
        assert result["quality_score"] == 75

    @patch("paper_tracker.discovery.call_cli")
    def test_low_quality_flags_persisted(self, mock_cli, reg: Registry):
        dr = reg.create_discovery_report("trending")
        reg.update_discovery_report(dr["id"], {
            "status": "completed",
            "finished_at": "2026-03-05T10:00:00",
            "content": "Bad report",
        })
        mock_cli.return_value = json.dumps({
            "quality_score": 35,
            "flags": [
                {"issue": "Only 2 themes identified", "severity": "high"},
                {"issue": "Missing cross-theme observations", "severity": "high"},
                {"issue": "No specific papers cited", "severity": "medium"},
            ],
            "summary": "Very low quality.",
        })
        result = review_discovery_report(reg, dr["id"], _CFG)
        assert result["quality_score"] == 35
        assert len(result["flags"]) == 3
        got = reg.get_discovery_report(dr["id"])
        assert got["quality_score"] == 35
        assert len(got["quality_flags"]) == 3


# ---------------------------------------------------------------
# Server endpoints: review + regenerate
# ---------------------------------------------------------------

class TestServerReviewEndpoint:
    @pytest.fixture(autouse=True)
    def _setup(self):
        from fastapi.testclient import TestClient
        from paper_tracker.server import app
        import paper_tracker.server as srv

        self.mock_reg = MagicMock()
        srv._registry = self.mock_reg
        srv._data_dir = "/tmp/test"
        srv._base_cfg = _CFG
        srv._scheduler = MagicMock()

        self.client = TestClient(app, raise_server_exceptions=False)
        yield
        srv._registry = None
        srv._scheduler = None

    def test_review_not_found_404(self):
        self.mock_reg.get_discovery_report.return_value = None
        resp = self.client.post("/api/discovery/dr-bad/review")
        assert resp.status_code == 404

    def test_review_not_completed_409(self):
        self.mock_reg.get_discovery_report.return_value = {
            "id": "dr-123", "status": "running", "type": "trending",
        }
        resp = self.client.post("/api/discovery/dr-123/review")
        assert resp.status_code == 409

    @patch("paper_tracker.server.review_discovery_report")
    def test_review_success(self, mock_review):
        self.mock_reg.get_discovery_report.return_value = {
            "id": "dr-123", "status": "completed", "type": "trending",
        }
        mock_review.return_value = {
            "quality_score": 85,
            "flags": [],
            "summary": "Good report.",
        }
        resp = self.client.post("/api/discovery/dr-123/review")
        assert resp.status_code == 200
        data = resp.json()
        assert data["quality_score"] == 85


class TestServerRegenerateEndpoint:
    @pytest.fixture(autouse=True)
    def _setup(self):
        from fastapi.testclient import TestClient
        from paper_tracker.server import app
        import paper_tracker.server as srv

        self.mock_reg = MagicMock()
        srv._registry = self.mock_reg
        srv._data_dir = "/tmp/test"
        srv._base_cfg = _CFG
        srv._scheduler = MagicMock()

        self.client = TestClient(app, raise_server_exceptions=False)
        yield
        srv._registry = None
        srv._scheduler = None

    def test_regenerate_not_found_404(self):
        self.mock_reg.get_discovery_report.return_value = None
        resp = self.client.post("/api/discovery/dr-bad/regenerate")
        assert resp.status_code == 404

    @patch("paper_tracker.server._brainstorm_executor")
    def test_regenerate_success(self, mock_exec):
        self.mock_reg.get_discovery_report.return_value = {
            "id": "dr-123", "type": "trending", "status": "completed",
        }
        self.mock_reg.list_discovery_reports.return_value = [
            {"id": "dr-123", "status": "completed"},
        ]
        resp = self.client.post("/api/discovery/dr-123/regenerate")
        assert resp.status_code == 202
        data = resp.json()
        assert data["status"] == "started"
        assert data["replacing"] == "dr-123"
        mock_exec.submit.assert_called_once()

    @patch("paper_tracker.server._brainstorm_executor")
    def test_regenerate_conflict_when_running(self, mock_exec):
        self.mock_reg.get_discovery_report.return_value = {
            "id": "dr-123", "type": "trending", "status": "completed",
        }
        self.mock_reg.list_discovery_reports.return_value = [
            {"id": "dr-456", "status": "running"},
        ]
        resp = self.client.post("/api/discovery/dr-123/regenerate")
        assert resp.status_code == 409
