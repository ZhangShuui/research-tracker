"""Unit tests for registry.py — discovery_reports CRUD."""

from __future__ import annotations

import pytest

from paper_tracker.registry import Registry


@pytest.fixture()
def reg(tmp_path):
    r = Registry(str(tmp_path))
    yield r
    r.close()


# ---------------------------------------------------------------
# create_discovery_report
# ---------------------------------------------------------------

class TestCreateDiscoveryReport:
    def test_create_trending(self, reg: Registry):
        dr = reg.create_discovery_report("trending")
        assert dr is not None
        assert dr["type"] == "trending"
        assert dr["status"] == "running"
        assert dr["id"].startswith("dr-")
        assert dr["paper_count"] == 0
        assert dr["content"] == ""

    def test_create_math(self, reg: Registry):
        dr = reg.create_discovery_report("math")
        assert dr["type"] == "math"
        assert dr["status"] == "running"

    def test_create_unique_ids(self, reg: Registry):
        dr1 = reg.create_discovery_report("trending")
        dr2 = reg.create_discovery_report("trending")
        assert dr1["id"] != dr2["id"]

    def test_started_at_populated(self, reg: Registry):
        dr = reg.create_discovery_report("trending")
        assert dr["started_at"] is not None
        assert len(dr["started_at"]) > 0


# ---------------------------------------------------------------
# get_discovery_report
# ---------------------------------------------------------------

class TestGetDiscoveryReport:
    def test_get_existing(self, reg: Registry):
        dr = reg.create_discovery_report("trending")
        got = reg.get_discovery_report(dr["id"])
        assert got is not None
        assert got["id"] == dr["id"]
        assert got["type"] == "trending"

    def test_get_nonexistent(self, reg: Registry):
        assert reg.get_discovery_report("dr-nonexistent") is None

    def test_papers_json_parsed(self, reg: Registry):
        dr = reg.create_discovery_report("trending")
        reg.update_discovery_report(dr["id"], {
            "papers_json": [{"arxiv_id": "2501.00001", "title": "T", "source": "arxiv"}],
        })
        got = reg.get_discovery_report(dr["id"])
        assert isinstance(got["papers_json"], list)
        assert got["papers_json"][0]["arxiv_id"] == "2501.00001"

    def test_source_stats_parsed(self, reg: Registry):
        dr = reg.create_discovery_report("trending")
        reg.update_discovery_report(dr["id"], {
            "source_stats": {"arxiv": 10, "huggingface": 5},
        })
        got = reg.get_discovery_report(dr["id"])
        assert isinstance(got["source_stats"], dict)
        assert got["source_stats"]["arxiv"] == 10


# ---------------------------------------------------------------
# list_discovery_reports
# ---------------------------------------------------------------

class TestListDiscoveryReports:
    def test_empty(self, reg: Registry):
        reports = reg.list_discovery_reports()
        assert reports == []

    def test_list_all(self, reg: Registry):
        reg.create_discovery_report("trending")
        reg.create_discovery_report("math")
        reports = reg.list_discovery_reports()
        assert len(reports) == 2

    def test_filter_by_type(self, reg: Registry):
        reg.create_discovery_report("trending")
        reg.create_discovery_report("math")
        reg.create_discovery_report("trending")

        trending = reg.list_discovery_reports(report_type="trending")
        assert len(trending) == 2
        assert all(r["type"] == "trending" for r in trending)

        math = reg.list_discovery_reports(report_type="math")
        assert len(math) == 1
        assert math[0]["type"] == "math"

    def test_limit(self, reg: Registry):
        for _ in range(5):
            reg.create_discovery_report("trending")
        reports = reg.list_discovery_reports(limit=3)
        assert len(reports) == 3

    def test_ordered_by_started_at_desc(self, reg: Registry):
        dr1 = reg.create_discovery_report("trending")
        dr2 = reg.create_discovery_report("trending")
        reports = reg.list_discovery_reports(report_type="trending")
        # Most recent first
        assert reports[0]["id"] == dr2["id"]
        assert reports[1]["id"] == dr1["id"]


# ---------------------------------------------------------------
# update_discovery_report
# ---------------------------------------------------------------

class TestUpdateDiscoveryReport:
    def test_update_status(self, reg: Registry):
        dr = reg.create_discovery_report("trending")
        reg.update_discovery_report(dr["id"], {"status": "completed"})
        got = reg.get_discovery_report(dr["id"])
        assert got["status"] == "completed"

    def test_update_content(self, reg: Registry):
        dr = reg.create_discovery_report("trending")
        reg.update_discovery_report(dr["id"], {"content": "# Report\nSome content"})
        got = reg.get_discovery_report(dr["id"])
        assert "# Report" in got["content"]

    def test_update_paper_count(self, reg: Registry):
        dr = reg.create_discovery_report("trending")
        reg.update_discovery_report(dr["id"], {"paper_count": 42})
        got = reg.get_discovery_report(dr["id"])
        assert got["paper_count"] == 42

    def test_update_papers_json_as_list(self, reg: Registry):
        dr = reg.create_discovery_report("trending")
        papers = [{"arxiv_id": "001", "title": "Test", "source": "arxiv"}]
        reg.update_discovery_report(dr["id"], {"papers_json": papers})
        got = reg.get_discovery_report(dr["id"])
        assert got["papers_json"] == papers

    def test_update_source_stats_as_dict(self, reg: Registry):
        dr = reg.create_discovery_report("trending")
        stats = {"arxiv": 30, "huggingface": 12}
        reg.update_discovery_report(dr["id"], {"source_stats": stats})
        got = reg.get_discovery_report(dr["id"])
        assert got["source_stats"] == stats

    def test_update_finished_at(self, reg: Registry):
        dr = reg.create_discovery_report("trending")
        reg.update_discovery_report(dr["id"], {"finished_at": "2026-03-05T12:00:00"})
        got = reg.get_discovery_report(dr["id"])
        assert got["finished_at"] == "2026-03-05T12:00:00"

    def test_update_ignores_unknown_fields(self, reg: Registry):
        dr = reg.create_discovery_report("trending")
        reg.update_discovery_report(dr["id"], {"unknown_field": "value"})
        got = reg.get_discovery_report(dr["id"])
        assert "unknown_field" not in got

    def test_update_multiple_fields(self, reg: Registry):
        dr = reg.create_discovery_report("trending")
        reg.update_discovery_report(dr["id"], {
            "status": "completed",
            "finished_at": "2026-03-05T12:00:00",
            "content": "# Done",
            "paper_count": 100,
        })
        got = reg.get_discovery_report(dr["id"])
        assert got["status"] == "completed"
        assert got["finished_at"] == "2026-03-05T12:00:00"
        assert got["content"] == "# Done"
        assert got["paper_count"] == 100


# ---------------------------------------------------------------
# get_latest_discovery_report
# ---------------------------------------------------------------

class TestGetLatestDiscoveryReport:
    def test_no_reports(self, reg: Registry):
        assert reg.get_latest_discovery_report("trending") is None

    def test_only_running_reports(self, reg: Registry):
        reg.create_discovery_report("trending")  # status=running
        assert reg.get_latest_discovery_report("trending") is None

    def test_returns_completed(self, reg: Registry):
        dr = reg.create_discovery_report("trending")
        reg.update_discovery_report(dr["id"], {
            "status": "completed",
            "finished_at": "2026-03-05T10:00:00",
            "content": "# Report 1",
        })
        latest = reg.get_latest_discovery_report("trending")
        assert latest is not None
        assert latest["id"] == dr["id"]

    def test_returns_most_recent_completed(self, reg: Registry):
        dr1 = reg.create_discovery_report("trending")
        reg.update_discovery_report(dr1["id"], {
            "status": "completed",
            "finished_at": "2026-03-04T10:00:00",
        })
        dr2 = reg.create_discovery_report("trending")
        reg.update_discovery_report(dr2["id"], {
            "status": "completed",
            "finished_at": "2026-03-05T10:00:00",
        })
        latest = reg.get_latest_discovery_report("trending")
        assert latest["id"] == dr2["id"]

    def test_type_filtering(self, reg: Registry):
        dr_t = reg.create_discovery_report("trending")
        reg.update_discovery_report(dr_t["id"], {
            "status": "completed",
            "finished_at": "2026-03-05T10:00:00",
        })
        dr_m = reg.create_discovery_report("math")
        reg.update_discovery_report(dr_m["id"], {
            "status": "completed",
            "finished_at": "2026-03-05T11:00:00",
        })

        latest_t = reg.get_latest_discovery_report("trending")
        assert latest_t["id"] == dr_t["id"]

        latest_m = reg.get_latest_discovery_report("math")
        assert latest_m["id"] == dr_m["id"]

    def test_ignores_failed(self, reg: Registry):
        dr = reg.create_discovery_report("trending")
        reg.update_discovery_report(dr["id"], {
            "status": "failed",
            "finished_at": "2026-03-05T10:00:00",
        })
        assert reg.get_latest_discovery_report("trending") is None
