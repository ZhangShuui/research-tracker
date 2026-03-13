"""SQLite persistence: deduplication + notification tracking."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS seen_arxiv (
    arxiv_id   TEXT PRIMARY KEY,
    title      TEXT,
    authors    TEXT,
    abstract   TEXT,
    url        TEXT,
    published  TEXT,
    summary    TEXT,
    key_insight TEXT DEFAULT '',
    method     TEXT DEFAULT '',
    contribution TEXT DEFAULT '',
    math_concepts TEXT DEFAULT '[]',
    venue      TEXT DEFAULT '',
    cited_works TEXT DEFAULT '[]',
    added_at   TEXT DEFAULT (datetime('now')),
    notified   INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS seen_github (
    repo_full_name TEXT PRIMARY KEY,
    description    TEXT,
    url            TEXT,
    stars          INTEGER,
    pushed_at      TEXT,
    summary        TEXT,
    added_at       TEXT DEFAULT (datetime('now')),
    notified       INTEGER DEFAULT 0
);
"""

_NEW_ARXIV_COLUMNS = [
    ("key_insight", "TEXT DEFAULT ''"),
    ("method", "TEXT DEFAULT ''"),
    ("contribution", "TEXT DEFAULT ''"),
    ("math_concepts", "TEXT DEFAULT '[]'"),
    ("venue", "TEXT DEFAULT ''"),
    ("cited_works", "TEXT DEFAULT '[]'"),
    ("quality_score", "INTEGER DEFAULT 0"),
    ("paper_id", "TEXT DEFAULT ''"),
    ("source", "TEXT DEFAULT 'arxiv'"),
    ("citation_count", "INTEGER DEFAULT 0"),
]


class Storage:
    def __init__(self, data_dir: str, topic_id: str | None = None):
        if topic_id:
            path = Path(data_dir) / topic_id
        else:
            path = Path(data_dir)
        path.mkdir(parents=True, exist_ok=True)
        self.db_path = path / "tracker.db"
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """Add missing columns to seen_arxiv for older databases."""
        existing = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(seen_arxiv)").fetchall()
        }
        added_paper_id = False
        for col_name, col_def in _NEW_ARXIV_COLUMNS:
            if col_name not in existing:
                self._conn.execute(
                    f"ALTER TABLE seen_arxiv ADD COLUMN {col_name} {col_def}"
                )
                if col_name == "paper_id":
                    added_paper_id = True
        if added_paper_id:
            self._conn.execute(
                "UPDATE seen_arxiv SET paper_id = arxiv_id WHERE paper_id = ''"
            )
        self._conn.commit()

    # ---- arXiv ----

    def is_arxiv_seen(self, arxiv_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM seen_arxiv WHERE arxiv_id = ?", (arxiv_id,)
        ).fetchone()
        return row is not None

    def is_paper_seen(self, paper_id: str) -> bool:
        """Check both paper_id and arxiv_id columns for dedup across sources."""
        row = self._conn.execute(
            "SELECT 1 FROM seen_arxiv WHERE paper_id = ? OR arxiv_id = ?",
            (paper_id, paper_id),
        ).fetchone()
        return row is not None

    def insert_arxiv(self, paper: dict) -> None:
        self._conn.execute(
            """INSERT OR IGNORE INTO seen_arxiv
               (arxiv_id, title, authors, abstract, url, published, summary,
                key_insight, method, contribution, math_concepts, venue, cited_works,
                quality_score, paper_id, source, citation_count)
               VALUES (:arxiv_id, :title, :authors, :abstract, :url, :published, :summary,
                       :key_insight, :method, :contribution, :math_concepts, :venue, :cited_works,
                       :quality_score, :paper_id, :source, :citation_count)""",
            {
                **paper,
                "key_insight": paper.get("key_insight", ""),
                "method": paper.get("method", ""),
                "contribution": paper.get("contribution", ""),
                "math_concepts": json.dumps(paper.get("math_concepts", [])),
                "venue": paper.get("venue", ""),
                "cited_works": json.dumps(paper.get("cited_works", [])),
                "quality_score": paper.get("quality_score", 0),
                "paper_id": paper.get("paper_id", paper.get("arxiv_id", "")),
                "source": paper.get("source", "arxiv"),
                "citation_count": paper.get("citation_count", 0),
            },
        )
        self._conn.commit()

    def get_unnotified_arxiv(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM seen_arxiv WHERE notified = 0"
        ).fetchall()
        return [self._arxiv_row(r) for r in rows]

    def mark_arxiv_notified(self, arxiv_id: str) -> None:
        self._conn.execute(
            "UPDATE seen_arxiv SET notified = 1 WHERE arxiv_id = ?", (arxiv_id,)
        )
        self._conn.commit()

    def get_arxiv(self, arxiv_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM seen_arxiv WHERE arxiv_id = ?", (arxiv_id,)
        ).fetchone()
        return self._arxiv_row(row) if row else None

    def get_all_arxiv(
        self,
        search: str = "",
        venue: str = "",
        source: str = "",
        date_from: str = "",
        date_to: str = "",
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict], int]:
        where_clauses: list[str] = []
        params: list[str | int] = []

        if search:
            where_clauses.append("(title LIKE ? OR abstract LIKE ? OR authors LIKE ?)")
            like = f"%{search}%"
            params.extend([like, like, like])
        if venue:
            where_clauses.append("venue LIKE ?")
            params.append(f"%{venue}%")
        if source:
            where_clauses.append("source = ?")
            params.append(source)
        if date_from:
            where_clauses.append("published >= ?")
            params.append(date_from)
        if date_to:
            where_clauses.append("published <= ?")
            params.append(date_to + "T23:59:59")

        where = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        total = self._conn.execute(
            f"SELECT COUNT(*) FROM seen_arxiv {where}", params
        ).fetchone()[0]

        rows = self._conn.execute(
            f"SELECT * FROM seen_arxiv {where} ORDER BY added_at DESC LIMIT ? OFFSET ?",
            [*params, limit, offset],
        ).fetchall()

        return [self._arxiv_row(r) for r in rows], total

    def get_all_github(
        self, limit: int = 50, offset: int = 0
    ) -> tuple[list[dict], int]:
        total = self._conn.execute(
            "SELECT COUNT(*) FROM seen_github"
        ).fetchone()[0]
        rows = self._conn.execute(
            "SELECT * FROM seen_github ORDER BY added_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [dict(r) for r in rows], total

    def delete_arxiv(self, arxiv_id: str) -> bool:
        """Delete a single paper. Returns True if a row was deleted."""
        cur = self._conn.execute(
            "DELETE FROM seen_arxiv WHERE arxiv_id = ?", (arxiv_id,)
        )
        self._conn.commit()
        return cur.rowcount > 0

    def update_arxiv_quality(self, arxiv_id: str, quality_score: int) -> bool:
        """Update quality_score for a single paper. Returns True if updated."""
        cur = self._conn.execute(
            "UPDATE seen_arxiv SET quality_score = ? WHERE arxiv_id = ?",
            (quality_score, arxiv_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def delete_arxiv_below_quality(self, min_quality: int) -> int:
        """Delete all papers with quality_score < min_quality. Returns count deleted."""
        cur = self._conn.execute(
            "DELETE FROM seen_arxiv WHERE quality_score < ? AND quality_score > 0",
            (min_quality,),
        )
        self._conn.commit()
        return cur.rowcount

    # ---- GitHub ----

    def is_github_seen(self, repo_full_name: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM seen_github WHERE repo_full_name = ?", (repo_full_name,)
        ).fetchone()
        return row is not None

    def insert_github(self, repo: dict) -> None:
        self._conn.execute(
            """INSERT OR IGNORE INTO seen_github
               (repo_full_name, description, url, stars, pushed_at, summary)
               VALUES (:repo_full_name, :description, :url, :stars, :pushed_at, :summary)""",
            repo,
        )
        self._conn.commit()

    def get_unnotified_github(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM seen_github WHERE notified = 0"
        ).fetchall()
        return [dict(r) for r in rows]

    def mark_github_notified(self, repo_full_name: str) -> None:
        self._conn.execute(
            "UPDATE seen_github SET notified = 1 WHERE repo_full_name = ?",
            (repo_full_name,),
        )
        self._conn.commit()

    # ---- Helpers ----

    @staticmethod
    def _arxiv_row(row: sqlite3.Row) -> dict:
        d = dict(row)
        for field in ("math_concepts", "cited_works"):
            val = d.get(field, "[]")
            if isinstance(val, str):
                try:
                    d[field] = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    d[field] = []
        return d

    def close(self) -> None:
        self._conn.close()
