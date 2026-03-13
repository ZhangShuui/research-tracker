"""SQLite registry for topics and sessions."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS topics (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    arxiv_keywords TEXT NOT NULL,
    arxiv_categories TEXT NOT NULL,
    arxiv_lookback_days INTEGER DEFAULT 2,
    github_keywords TEXT NOT NULL,
    github_lookback_days INTEGER DEFAULT 7,
    schedule_cron TEXT DEFAULT '',
    enabled INTEGER DEFAULT 1,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    topic_id TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    paper_count INTEGER DEFAULT 0,
    repo_count INTEGER DEFAULT 0,
    status TEXT DEFAULT 'running',
    report_path TEXT DEFAULT '',
    insights_path TEXT DEFAULT '',
    FOREIGN KEY (topic_id) REFERENCES topics(id)
);

CREATE TABLE IF NOT EXISTS brainstorm_sessions (
    id TEXT PRIMARY KEY,
    topic_id TEXT NOT NULL,
    mode TEXT DEFAULT 'auto',
    user_idea TEXT DEFAULT '',
    status TEXT DEFAULT 'running',
    started_at TEXT,
    finished_at TEXT,
    ideas_json TEXT DEFAULT '[]',
    literature_result TEXT DEFAULT '',
    logic_result TEXT DEFAULT '',
    code_result TEXT DEFAULT '',
    run_code_verification INTEGER DEFAULT 0,
    FOREIGN KEY (topic_id) REFERENCES topics(id)
);

CREATE TABLE IF NOT EXISTS research_plans (
    id TEXT PRIMARY KEY,
    topic_id TEXT NOT NULL,
    brainstorm_session_id TEXT DEFAULT '',
    idea_title TEXT DEFAULT '',
    idea_json TEXT DEFAULT '{}',
    status TEXT DEFAULT 'running',
    started_at TEXT,
    finished_at TEXT,
    introduction TEXT DEFAULT '',
    related_work TEXT DEFAULT '',
    methodology TEXT DEFAULT '',
    experimental_design TEXT DEFAULT '',
    expected_results TEXT DEFAULT '',
    timeline TEXT DEFAULT '',
    review TEXT DEFAULT '',
    full_markdown TEXT DEFAULT '',
    FOREIGN KEY (topic_id) REFERENCES topics(id)
);

CREATE TABLE IF NOT EXISTS discovery_reports (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    status TEXT DEFAULT 'running',
    started_at TEXT,
    finished_at TEXT,
    content TEXT DEFAULT '',
    papers_json TEXT DEFAULT '[]',
    paper_count INTEGER DEFAULT 0,
    source_stats TEXT DEFAULT '{}',
    quality_score INTEGER DEFAULT -1,
    quality_flags TEXT DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS translations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type TEXT NOT NULL,
    source_id TEXT NOT NULL,
    field TEXT NOT NULL,
    language TEXT NOT NULL DEFAULT 'zh',
    content TEXT DEFAULT '',
    created_at TEXT,
    UNIQUE(source_type, source_id, field, language)
);
"""


class Registry:
    def __init__(self, data_dir: str | Path):
        path = Path(data_dir)
        path.mkdir(parents=True, exist_ok=True)
        self.db_path = path / "registry.db"
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._migrate_discovery_quality()
        self._migrate_brainstorm_review()
        self._migrate_topic_sources()
        self._migrate_research_plan_history()
        self._migrate_chat_tables()
        self._lock = threading.Lock()

    def _migrate_discovery_quality(self) -> None:
        """Add quality_score/quality_flags columns if missing (migration)."""
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(discovery_reports)").fetchall()}
        if "quality_score" not in cols:
            self._conn.execute("ALTER TABLE discovery_reports ADD COLUMN quality_score INTEGER DEFAULT -1")
        if "quality_flags" not in cols:
            self._conn.execute("ALTER TABLE discovery_reports ADD COLUMN quality_flags TEXT DEFAULT '[]'")
        self._conn.commit()

    def _migrate_brainstorm_review(self) -> None:
        """Add review_result column to brainstorm_sessions if missing (migration)."""
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(brainstorm_sessions)").fetchall()}
        if "review_result" not in cols:
            self._conn.execute("ALTER TABLE brainstorm_sessions ADD COLUMN review_result TEXT DEFAULT ''")
            self._conn.commit()

    def _migrate_research_plan_history(self) -> None:
        """Add review_history column to research_plans if missing."""
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(research_plans)").fetchall()}
        if "review_history" not in cols:
            self._conn.execute("ALTER TABLE research_plans ADD COLUMN review_history TEXT DEFAULT '[]'")
            self._conn.commit()

    def _migrate_chat_tables(self) -> None:
        """Create chat_sessions and chat_messages tables if missing."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id TEXT PRIMARY KEY,
                topic_id TEXT NOT NULL,
                title TEXT DEFAULT '',
                status TEXT DEFAULT 'active',
                created_at TEXT,
                updated_at TEXT,
                message_count INTEGER DEFAULT 0,
                FOREIGN KEY (topic_id) REFERENCES topics(id)
            );
            CREATE TABLE IF NOT EXISTS chat_messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                topic_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                cited_papers TEXT DEFAULT '[]',
                status TEXT DEFAULT 'completed',
                created_at TEXT,
                FOREIGN KEY (session_id) REFERENCES chat_sessions(id)
            );
        """)
        self._conn.commit()

    def _migrate_topic_sources(self) -> None:
        """Add OpenAlex/OpenReview columns to topics if missing."""
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(topics)").fetchall()}
        new_cols = [
            ("openalex_enabled", "INTEGER DEFAULT 0"),
            ("openalex_keywords", "TEXT DEFAULT '[]'"),
            ("openalex_lookback_days", "INTEGER DEFAULT 7"),
            ("openalex_venues", "TEXT DEFAULT '[]'"),
            ("openalex_max_results", "INTEGER DEFAULT 200"),
            ("openreview_enabled", "INTEGER DEFAULT 0"),
            ("openreview_venues", "TEXT DEFAULT '[]'"),
            ("openreview_keywords", "TEXT DEFAULT '[]'"),
            ("openreview_max_results", "INTEGER DEFAULT 100"),
            ("search_date_from", "TEXT DEFAULT ''"),
            ("search_date_to", "TEXT DEFAULT ''"),
        ]
        for col_name, col_def in new_cols:
            if col_name not in cols:
                self._conn.execute(f"ALTER TABLE topics ADD COLUMN {col_name} {col_def}")
        self._conn.commit()

    # ---- Topics ----

    def list_topics(self) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM topics ORDER BY created_at DESC").fetchall()
        return [self._topic_row(r) for r in rows]

    def get_topic(self, topic_id: str) -> dict | None:
        row = self._conn.execute("SELECT * FROM topics WHERE id = ?", (topic_id,)).fetchone()
        return self._topic_row(row) if row else None

    def create_topic(self, topic: dict) -> dict:
        with self._lock:
            self._conn.execute(
                """INSERT INTO topics
                   (id, name, description, arxiv_keywords, arxiv_categories,
                    arxiv_lookback_days, github_keywords, github_lookback_days,
                    schedule_cron, enabled,
                    openalex_enabled, openalex_keywords, openalex_lookback_days, openalex_venues, openalex_max_results,
                    openreview_enabled, openreview_venues, openreview_keywords, openreview_max_results,
                    search_date_from, search_date_to)
                   VALUES (:id, :name, :description, :arxiv_keywords, :arxiv_categories,
                           :arxiv_lookback_days, :github_keywords, :github_lookback_days,
                           :schedule_cron, :enabled,
                           :openalex_enabled, :openalex_keywords, :openalex_lookback_days, :openalex_venues, :openalex_max_results,
                           :openreview_enabled, :openreview_venues, :openreview_keywords, :openreview_max_results,
                           :search_date_from, :search_date_to)""",
                {
                    "id": topic["id"],
                    "name": topic["name"],
                    "description": topic.get("description", ""),
                    "arxiv_keywords": json.dumps(topic.get("arxiv_keywords", [])),
                    "arxiv_categories": json.dumps(topic.get("arxiv_categories", [])),
                    "arxiv_lookback_days": topic.get("arxiv_lookback_days", 2),
                    "github_keywords": json.dumps(topic.get("github_keywords", [])),
                    "github_lookback_days": topic.get("github_lookback_days", 7),
                    "schedule_cron": topic.get("schedule_cron", ""),
                    "enabled": 1 if topic.get("enabled", True) else 0,
                    "openalex_enabled": 1 if topic.get("openalex_enabled", False) else 0,
                    "openalex_keywords": json.dumps(topic.get("openalex_keywords", [])),
                    "openalex_lookback_days": topic.get("openalex_lookback_days", 7),
                    "openalex_venues": json.dumps(topic.get("openalex_venues", [])),
                    "openalex_max_results": topic.get("openalex_max_results", 200),
                    "openreview_enabled": 1 if topic.get("openreview_enabled", False) else 0,
                    "openreview_venues": json.dumps(topic.get("openreview_venues", [])),
                    "openreview_keywords": json.dumps(topic.get("openreview_keywords", [])),
                    "openreview_max_results": topic.get("openreview_max_results", 100),
                    "search_date_from": topic.get("search_date_from", ""),
                    "search_date_to": topic.get("search_date_to", ""),
                },
            )
            self._conn.commit()
        return self.get_topic(topic["id"])

    def update_topic(self, topic_id: str, updates: dict) -> dict | None:
        fields = []
        params = []
        field_map = {
            "name": "name",
            "description": "description",
            "arxiv_lookback_days": "arxiv_lookback_days",
            "github_lookback_days": "github_lookback_days",
            "schedule_cron": "schedule_cron",
            "enabled": "enabled",
            "openalex_lookback_days": "openalex_lookback_days",
            "openalex_max_results": "openalex_max_results",
            "openreview_max_results": "openreview_max_results",
            "search_date_from": "search_date_from",
            "search_date_to": "search_date_to",
        }
        bool_fields = {"openalex_enabled", "openreview_enabled"}
        list_fields = {
            "arxiv_keywords", "arxiv_categories", "github_keywords",
            "openalex_keywords", "openalex_venues",
            "openreview_venues", "openreview_keywords",
        }

        for key, val in updates.items():
            if key in field_map:
                fields.append(f"{field_map[key]} = ?")
                params.append(val)
            elif key in bool_fields:
                fields.append(f"{key} = ?")
                params.append(1 if val else 0)
            elif key in list_fields:
                fields.append(f"{key} = ?")
                params.append(json.dumps(val))

        if not fields:
            return self.get_topic(topic_id)

        params.append(topic_id)
        with self._lock:
            self._conn.execute(
                f"UPDATE topics SET {', '.join(fields)} WHERE id = ?", params
            )
            self._conn.commit()
        return self.get_topic(topic_id)

    def delete_topic(self, topic_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM chat_messages WHERE topic_id = ?", (topic_id,))
            self._conn.execute("DELETE FROM chat_sessions WHERE topic_id = ?", (topic_id,))
            self._conn.execute("DELETE FROM research_plans WHERE topic_id = ?", (topic_id,))
            self._conn.execute("DELETE FROM brainstorm_sessions WHERE topic_id = ?", (topic_id,))
            self._conn.execute("DELETE FROM sessions WHERE topic_id = ?", (topic_id,))
            self._conn.execute("DELETE FROM topics WHERE id = ?", (topic_id,))
            self._conn.commit()

    # ---- Sessions ----

    def create_session(self, topic_id: str) -> dict:
        """Create a new session with auto-generated ID: YYYY-MM-DD_NNN.

        The count-then-insert is wrapped in a mutex so concurrent calls for the
        same topic on the same day each get a unique, incrementing session ID.
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            count = self._conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE id LIKE ?",
                (f"{today}_%",),
            ).fetchone()[0]
            session_id = f"{today}_{count + 1:03d}"
            self._conn.execute(
                """INSERT INTO sessions (id, topic_id, started_at, status)
                   VALUES (?, ?, ?, 'running')""",
                (session_id, topic_id, now),
            )
            self._conn.commit()
        return self.get_session(topic_id, session_id)

    def get_session(self, topic_id: str, session_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE id = ? AND topic_id = ?",
            (session_id, topic_id),
        ).fetchone()
        return dict(row) if row else None

    def list_sessions(self, topic_id: str, limit: int = 50, offset: int = 0) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM sessions WHERE topic_id = ? ORDER BY started_at DESC LIMIT ? OFFSET ?",
            (topic_id, limit, offset),
        ).fetchall()
        return [dict(r) for r in rows]

    def update_session(self, topic_id: str, session_id: str, updates: dict) -> None:
        allowed = {"status", "finished_at", "paper_count", "repo_count", "report_path", "insights_path"}
        fields = []
        params = []
        for key, val in updates.items():
            if key in allowed:
                fields.append(f"{key} = ?")
                params.append(val)
        if not fields:
            return
        params.extend([session_id, topic_id])
        with self._lock:
            self._conn.execute(
                f"UPDATE sessions SET {', '.join(fields)} WHERE id = ? AND topic_id = ?",
                params,
            )
            self._conn.commit()

    def get_latest_session(self, topic_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE topic_id = ? ORDER BY started_at DESC LIMIT 1",
            (topic_id,),
        ).fetchone()
        return dict(row) if row else None

    # ---- Brainstorm Sessions ----

    def create_brainstorm_session(
        self,
        topic_id: str,
        mode: str = "auto",
        user_idea: str = "",
        run_code_verification: bool = False,
    ) -> dict:
        import uuid
        session_id = f"bs-{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                """INSERT INTO brainstorm_sessions
                   (id, topic_id, mode, user_idea, status, started_at, run_code_verification)
                   VALUES (?, ?, ?, ?, 'running', ?, ?)""",
                (session_id, topic_id, mode, user_idea, now, 1 if run_code_verification else 0),
            )
            self._conn.commit()
        return self.get_brainstorm_session(topic_id, session_id)

    def get_brainstorm_session(self, topic_id: str, session_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM brainstorm_sessions WHERE id = ? AND topic_id = ?",
            (session_id, topic_id),
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        # Parse JSON fields
        try:
            d["ideas_json"] = json.loads(d.get("ideas_json", "[]") or "[]")
        except (json.JSONDecodeError, TypeError):
            d["ideas_json"] = []
        d["run_code_verification"] = bool(d.get("run_code_verification", 0))
        return d

    def list_brainstorm_sessions(self, topic_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM brainstorm_sessions WHERE topic_id = ? ORDER BY started_at DESC",
            (topic_id,),
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            try:
                d["ideas_json"] = json.loads(d.get("ideas_json", "[]") or "[]")
            except (json.JSONDecodeError, TypeError):
                d["ideas_json"] = []
            d["run_code_verification"] = bool(d.get("run_code_verification", 0))
            result.append(d)
        return result

    def update_brainstorm_session(self, topic_id: str, session_id: str, updates: dict) -> None:
        allowed = {"status", "finished_at", "ideas_json", "literature_result", "logic_result", "code_result", "review_result"}
        fields = []
        params = []
        for key, val in updates.items():
            if key in allowed:
                fields.append(f"{key} = ?")
                if key == "ideas_json" and not isinstance(val, str):
                    params.append(json.dumps(val))
                else:
                    params.append(val)
        if not fields:
            return
        params.extend([session_id, topic_id])
        with self._lock:
            self._conn.execute(
                f"UPDATE brainstorm_sessions SET {', '.join(fields)} WHERE id = ? AND topic_id = ?",
                params,
            )
            self._conn.commit()

    # ---- Research Plans ----

    def create_research_plan(
        self,
        topic_id: str,
        idea: dict,
        brainstorm_session_id: str = "",
    ) -> dict:
        import uuid
        plan_id = f"rp-{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat()
        idea_title = idea.get("title", "")
        with self._lock:
            self._conn.execute(
                """INSERT INTO research_plans
                   (id, topic_id, brainstorm_session_id, idea_title, idea_json, status, started_at)
                   VALUES (?, ?, ?, ?, ?, 'running', ?)""",
                (plan_id, topic_id, brainstorm_session_id, idea_title, json.dumps(idea), now),
            )
            self._conn.commit()
        return self.get_research_plan(topic_id, plan_id)

    def get_research_plan(self, topic_id: str, plan_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM research_plans WHERE id = ? AND topic_id = ?",
            (plan_id, topic_id),
        ).fetchone()
        if not row:
            return None
        return self._plan_row(row)

    def list_research_plans(self, topic_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM research_plans WHERE topic_id = ? ORDER BY started_at DESC",
            (topic_id,),
        ).fetchall()
        return [self._plan_row(r) for r in rows]

    @staticmethod
    def _plan_row(row: sqlite3.Row) -> dict:
        d = dict(row)
        try:
            d["idea_json"] = json.loads(d.get("idea_json", "{}") or "{}")
        except (json.JSONDecodeError, TypeError):
            d["idea_json"] = {}
        try:
            d["review_history"] = json.loads(d.get("review_history", "[]") or "[]")
        except (json.JSONDecodeError, TypeError):
            d["review_history"] = []
        return d

    def update_research_plan(self, topic_id: str, plan_id: str, updates: dict) -> None:
        allowed = {
            "status", "finished_at", "introduction", "related_work",
            "methodology", "experimental_design", "expected_results",
            "timeline", "review", "full_markdown", "review_history",
        }
        fields = []
        params = []
        for key, val in updates.items():
            if key in allowed:
                fields.append(f"{key} = ?")
                params.append(val)
        if not fields:
            return
        params.extend([plan_id, topic_id])
        with self._lock:
            self._conn.execute(
                f"UPDATE research_plans SET {', '.join(fields)} WHERE id = ? AND topic_id = ?",
                params,
            )
            self._conn.commit()

    # ---- Discovery Reports ----

    def create_discovery_report(self, report_type: str) -> dict:
        import uuid
        report_id = f"dr-{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                """INSERT INTO discovery_reports
                   (id, type, status, started_at)
                   VALUES (?, ?, 'running', ?)""",
                (report_id, report_type, now),
            )
            self._conn.commit()
        return self.get_discovery_report(report_id)

    def get_discovery_report(self, report_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM discovery_reports WHERE id = ?", (report_id,)
        ).fetchone()
        if not row:
            return None
        return self._discovery_row(row)

    def list_discovery_reports(self, report_type: str | None = None, limit: int = 20) -> list[dict]:
        if report_type:
            rows = self._conn.execute(
                "SELECT * FROM discovery_reports WHERE type = ? ORDER BY started_at DESC LIMIT ?",
                (report_type, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM discovery_reports ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._discovery_row(r) for r in rows]

    def update_discovery_report(self, report_id: str, updates: dict) -> None:
        allowed = {"status", "finished_at", "content", "papers_json", "paper_count", "source_stats", "quality_score", "quality_flags"}
        fields = []
        params = []
        for key, val in updates.items():
            if key in allowed:
                fields.append(f"{key} = ?")
                if key in ("papers_json", "source_stats", "quality_flags") and not isinstance(val, str):
                    params.append(json.dumps(val))
                else:
                    params.append(val)
        if not fields:
            return
        params.append(report_id)
        with self._lock:
            self._conn.execute(
                f"UPDATE discovery_reports SET {', '.join(fields)} WHERE id = ?",
                params,
            )
            self._conn.commit()

    def get_latest_discovery_report(self, report_type: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM discovery_reports WHERE type = ? AND status = 'completed' ORDER BY finished_at DESC LIMIT 1",
            (report_type,),
        ).fetchone()
        if not row:
            return None
        return self._discovery_row(row)

    @staticmethod
    def _discovery_row(row: sqlite3.Row) -> dict:
        d = dict(row)
        for field in ("papers_json", "source_stats", "quality_flags"):
            if isinstance(d.get(field), str):
                try:
                    d[field] = json.loads(d[field])
                except (json.JSONDecodeError, TypeError):
                    d[field] = [] if field in ("papers_json", "quality_flags") else {}
        return d

    # ---- Chat Sessions ----

    def create_chat_session(self, topic_id: str, title: str = "") -> dict:
        import uuid
        session_id = f"ch-{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                """INSERT INTO chat_sessions (id, topic_id, title, status, created_at, updated_at)
                   VALUES (?, ?, ?, 'active', ?, ?)""",
                (session_id, topic_id, title, now, now),
            )
            self._conn.commit()
        return self.get_chat_session(topic_id, session_id)

    def get_chat_session(self, topic_id: str, session_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM chat_sessions WHERE id = ? AND topic_id = ?",
            (session_id, topic_id),
        ).fetchone()
        return dict(row) if row else None

    def list_chat_sessions(self, topic_id: str, limit: int = 50) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM chat_sessions WHERE topic_id = ? ORDER BY updated_at DESC LIMIT ?",
            (topic_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_chat_session(self, topic_id: str, session_id: str) -> bool:
        with self._lock:
            self._conn.execute(
                "DELETE FROM chat_messages WHERE session_id = ? AND topic_id = ?",
                (session_id, topic_id),
            )
            cur = self._conn.execute(
                "DELETE FROM chat_sessions WHERE id = ? AND topic_id = ?",
                (session_id, topic_id),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def add_chat_message(
        self,
        topic_id: str,
        session_id: str,
        role: str,
        content: str,
        cited_papers: list | None = None,
        status: str = "completed",
    ) -> dict:
        import uuid
        msg_id = f"cm-{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat()
        cited_json = json.dumps(cited_papers or [])
        with self._lock:
            self._conn.execute(
                """INSERT INTO chat_messages (id, session_id, topic_id, role, content, cited_papers, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (msg_id, session_id, topic_id, role, content, cited_json, status, now),
            )
            self._conn.execute(
                "UPDATE chat_sessions SET message_count = message_count + 1, updated_at = ? WHERE id = ?",
                (now, session_id),
            )
            self._conn.commit()
        return self.get_chat_message(msg_id)

    def get_chat_message(self, msg_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM chat_messages WHERE id = ?", (msg_id,)
        ).fetchone()
        return self._chat_msg_row(row) if row else None

    def update_chat_message(self, msg_id: str, updates: dict) -> None:
        allowed = {"content", "cited_papers", "status"}
        fields = []
        params = []
        for key, val in updates.items():
            if key in allowed:
                fields.append(f"{key} = ?")
                if key == "cited_papers" and not isinstance(val, str):
                    params.append(json.dumps(val))
                else:
                    params.append(val)
        if not fields:
            return
        params.append(msg_id)
        with self._lock:
            self._conn.execute(
                f"UPDATE chat_messages SET {', '.join(fields)} WHERE id = ?",
                params,
            )
            self._conn.commit()

    def list_chat_messages(self, topic_id: str, session_id: str, limit: int = 100) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM chat_messages WHERE session_id = ? AND topic_id = ? ORDER BY created_at ASC LIMIT ?",
            (session_id, topic_id, limit),
        ).fetchall()
        return [self._chat_msg_row(r) for r in rows]

    @staticmethod
    def _chat_msg_row(row: sqlite3.Row) -> dict:
        d = dict(row)
        try:
            d["cited_papers"] = json.loads(d.get("cited_papers", "[]") or "[]")
        except (json.JSONDecodeError, TypeError):
            d["cited_papers"] = []
        return d

    # ---- Translations ----

    def get_translation(
        self, source_type: str, source_id: str, field: str, language: str = "zh"
    ) -> str | None:
        row = self._conn.execute(
            "SELECT content FROM translations WHERE source_type = ? AND source_id = ? AND field = ? AND language = ?",
            (source_type, source_id, field, language),
        ).fetchone()
        return row["content"] if row else None

    def save_translation(
        self, source_type: str, source_id: str, field: str, language: str, content: str
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                """INSERT INTO translations (source_type, source_id, field, language, content, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(source_type, source_id, field, language)
                   DO UPDATE SET content = excluded.content, created_at = excluded.created_at""",
                (source_type, source_id, field, language, content, now),
            )
            self._conn.commit()

    # ---- Startup recovery ----

    def recover_stale_tasks(self) -> dict[str, int]:
        """Mark all 'running'/'pending'/'generating' records as 'failed'.

        Should be called once at server startup. Any task that was in a
        transient state when the previous process died cannot be resumed,
        so we fail them cleanly.

        Returns a dict of {table: count_recovered}.
        """
        now = datetime.now(timezone.utc).isoformat()
        counts: dict[str, int] = {}
        with self._lock:
            # sessions
            cur = self._conn.execute(
                "UPDATE sessions SET status = 'failed', finished_at = ? WHERE status = 'running'",
                (now,),
            )
            counts["sessions"] = cur.rowcount

            # brainstorm_sessions
            cur = self._conn.execute(
                "UPDATE brainstorm_sessions SET status = 'failed', finished_at = ? WHERE status = 'running'",
                (now,),
            )
            counts["brainstorm_sessions"] = cur.rowcount

            # research_plans
            cur = self._conn.execute(
                "UPDATE research_plans SET status = 'failed', finished_at = ? WHERE status = 'running'",
                (now,),
            )
            counts["research_plans"] = cur.rowcount

            # discovery_reports
            cur = self._conn.execute(
                "UPDATE discovery_reports SET status = 'failed', finished_at = ? WHERE status = 'running'",
                (now,),
            )
            counts["discovery_reports"] = cur.rowcount

            # chat_messages (pending/generating → failed)
            cur = self._conn.execute(
                "UPDATE chat_messages SET status = 'failed' WHERE status IN ('pending', 'generating')",
            )
            counts["chat_messages"] = cur.rowcount

            self._conn.commit()

        recovered = {k: v for k, v in counts.items() if v > 0}
        return recovered

    # ---- Helpers ----

    @staticmethod
    def _topic_row(row: sqlite3.Row) -> dict:
        d = dict(row)
        for field in (
            "arxiv_keywords", "arxiv_categories", "github_keywords",
            "openalex_keywords", "openalex_venues",
            "openreview_venues", "openreview_keywords",
        ):
            if isinstance(d.get(field), str):
                try:
                    d[field] = json.loads(d[field])
                except (json.JSONDecodeError, TypeError):
                    d[field] = []
        d["enabled"] = bool(d.get("enabled", 1))
        d["openalex_enabled"] = bool(d.get("openalex_enabled", 0))
        d["openreview_enabled"] = bool(d.get("openreview_enabled", 0))
        return d

    def close(self) -> None:
        self._conn.close()
