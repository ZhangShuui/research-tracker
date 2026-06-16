# Deep-Dig (Agentic) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** From one seed paper, an autonomous Claude agent explores its citation chain (Semantic Scholar), semantic neighbors (rag), and math neighbors, then produces — in a two-phase user-gated flow with agentic Q&A — an insights memo and structured new ideas, with the dig-corpus and citation graph reconstructed from the agent's tool trace.

**Architecture:** A new httpx source (`semantic_scholar.py`) supplies references/citations. A new agent module (`deep_dig_agent.py`, mirroring `brainstorm_agent.py`) runs Claude Agent SDK tool-use loops whose tools record every paper + edge into a shared collector. An orchestration module (`deep_dig.py`) resolves the seed and runs each phase, falling back to a deterministic path if the SDK/agent fails. Two new registry tables persist sessions + Q&A messages. Server endpoints (reusing `_brainstorm_executor`) drive the two-phase lifecycle. A new Next.js page mirrors the deep-read page and adds a lightweight SVG graph.

**Tech Stack:** Python 3.14, FastAPI, SQLite (`registry.py`), `claude-agent-sdk`, `httpx`, pytest; Next.js 18 + TanStack Query + Tailwind + lucide-react.

**Status lifecycle:** `exploring` → `insights_ready` (user gate) → `generating_ideas` → `completed`; `failed` on phase-1 error. Phase-2 (ideas) failure reverts to `insights_ready` + `error_message` so the user can retry. Q&A is available once status is `insights_ready`, `generating_ideas`, or `completed`.

---

## Conventions used across tasks

- **Session id:** `dd-<8hex>`. **Message id:** `ddm-<8hex>`.
- **Paper id key:** every paper dict uses `arxiv_id` as the primary id (fallback `paper_id`), matching the rest of the repo.
- **Collector node `[Pn]` index:** assigned in first-seen order (`P1`, `P2`, …); the agent cites these in the memo; `dig_corpus_json` preserves them.
- **Run tests with:** `uv run pytest <path> -v` from the repo root.
- **Commit after every task.** Branch first if on `main` (see Task 0).

---

## Task 0: Branch

**Files:** none

- [ ] **Step 1: Create a feature branch**

Run:
```bash
git checkout -b feat/deep-dig
```
Expected: `Switched to a new branch 'feat/deep-dig'`

---

## Task 1: Semantic Scholar source

**Files:**
- Create: `src/paper_tracker/sources/semantic_scholar.py`
- Test: `tests/test_semantic_scholar.py`

The repo's `httpx_get_with_retry` (in `sources/__init__.py`) retries 429/503 and raises `httpx.HTTPStatusError` on other 4xx — so we wrap calls in try/except and return `[]` on any failure (a missing citation path must degrade, never break the dig).

- [ ] **Step 1: Write the failing test**

Create `tests/test_semantic_scholar.py`:
```python
"""Tests for the Semantic Scholar citation source. HTTP is mocked."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import httpx

from paper_tracker.sources import semantic_scholar as s2


def _resp(payload: dict) -> MagicMock:
    r = MagicMock()
    r.json.return_value = payload
    r.raise_for_status = MagicMock()
    return r


_REF_PAYLOAD = {
    "data": [
        {
            "isInfluential": True,
            "intents": ["methodology"],
            "citedPaper": {
                "paperId": "abc123",
                "externalIds": {"ArXiv": "1706.03762", "DOI": "10.1/x"},
                "title": "Attention Is All You Need",
                "abstract": "The Transformer.",
                "year": 2017,
                "venue": "NeurIPS",
                "authors": [{"name": "Vaswani"}, {"name": "Shazeer"}],
                "citationCount": 100000,
                "influentialCitationCount": 9000,
            },
        },
        {
            "isInfluential": False,
            "intents": [],
            "citedPaper": {
                "paperId": "noTitle",
                "externalIds": {},
                "title": None,  # must be dropped
            },
        },
    ]
}


class TestGetReferences:
    def test_parses_and_maps(self):
        with patch("paper_tracker.sources.semantic_scholar.httpx_get_with_retry",
                   return_value=_resp(_REF_PAYLOAD)):
            out = s2.get_references("2401.00001", {})
        assert len(out) == 1  # the null-title row is dropped
        p = out[0]
        assert p["arxiv_id"] == "1706.03762"
        assert p["source"] == "semantic_scholar"
        assert p["title"] == "Attention Is All You Need"
        assert p["authors"] == "Vaswani, Shazeer"
        assert p["url"] == "https://arxiv.org/abs/1706.03762"
        assert p["venue"] == "NeurIPS"
        assert p["doi"] == "10.1/x"
        assert p["citation_count"] == 100000
        assert p["influential_citation_count"] == 9000
        assert p["_edge_influential"] is True
        assert p["_edge_intents"] == ["methodology"]

    def test_s2_only_paper_gets_s2_id(self):
        payload = {"data": [{"isInfluential": False, "citedPaper": {
            "paperId": "xyz", "externalIds": {}, "title": "No arXiv", "authors": []}}]}
        with patch("paper_tracker.sources.semantic_scholar.httpx_get_with_retry",
                   return_value=_resp(payload)):
            out = s2.get_references("2401.00001", {})
        assert out[0]["arxiv_id"] == "s2:xyz"
        assert out[0]["url"] == "https://www.semanticscholar.org/paper/xyz"

    def test_returns_empty_on_http_error(self):
        err = httpx.HTTPStatusError("404", request=MagicMock(), response=MagicMock(status_code=404))
        with patch("paper_tracker.sources.semantic_scholar.httpx_get_with_retry", side_effect=err):
            assert s2.get_references("2401.00001", {}) == []


class TestGetCitations:
    def test_citing_paper_key_and_influential_first(self):
        payload = {"data": [
            {"isInfluential": False, "citingPaper": {
                "paperId": "p2", "externalIds": {"ArXiv": "2402.00002"}, "title": "Later non-infl",
                "authors": [], "influentialCitationCount": 0}},
            {"isInfluential": True, "citingPaper": {
                "paperId": "p1", "externalIds": {"ArXiv": "2402.00001"}, "title": "Later influential",
                "authors": [], "influentialCitationCount": 5}},
        ]}
        with patch("paper_tracker.sources.semantic_scholar.httpx_get_with_retry",
                   return_value=_resp(payload)):
            out = s2.get_citations("2401.00001", {})
        assert [p["arxiv_id"] for p in out] == ["2402.00001", "2402.00002"]  # influential first


class TestS2Id:
    def test_normalizes_forms(self):
        assert s2._s2_id("2401.00001") == "ARXIV:2401.00001"
        assert s2._s2_id("https://arxiv.org/abs/2401.00001") == "ARXIV:2401.00001"
        assert s2._s2_id("s2:abc123") == "abc123"
        assert s2._s2_id("CorpusId:42") == "CorpusId:42"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_semantic_scholar.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'paper_tracker.sources.semantic_scholar'`

- [ ] **Step 3: Write the implementation**

Create `src/paper_tracker/sources/semantic_scholar.py`:
```python
"""Semantic Scholar Graph API — citation chain source for deep-dig.

Fetches a paper's references (what it cites) and citations (what cites it),
mapped to the repo's standard paper dict + per-edge metadata. Every public
function returns [] on ANY failure (network / 404 / rate-limit exhausted) so a
missing citation path degrades gracefully instead of breaking the dig.

Auth is optional: an API key (cfg["deep_dig"]["s2_api_key"] or env S2_API_KEY)
lifts the shared rate limit but is not required.
"""

from __future__ import annotations

import logging
import os
import re

import httpx

from paper_tracker.sources import httpx_get_with_retry

log = logging.getLogger(__name__)

_S2_BASE = "https://api.semanticscholar.org/graph/v1"
_FIELDS = ("paperId,externalIds,title,abstract,year,venue,authors,"
           "citationCount,influentialCitationCount")


def _s2_id(paper_id: str) -> str:
    """Normalize a paper id to an S2 path segment.

    arXiv id / arXiv URL -> 'ARXIV:<id>'; 's2:<hash>' -> '<hash>';
    'CorpusId:<n>' / 'DOI:<doi>' passed through.
    """
    pid = (paper_id or "").strip()
    if pid.startswith("s2:"):
        return pid[3:]
    if pid.startswith(("CorpusId:", "DOI:", "ARXIV:")):
        return pid
    m = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", pid)
    if m:
        return f"ARXIV:{m.group(1)}"
    return f"ARXIV:{pid}"


def _headers(cfg: dict) -> dict:
    key = (cfg.get("deep_dig", {}).get("s2_api_key") or os.environ.get("S2_API_KEY") or "").strip()
    return {"x-api-key": key} if key else {}


def _map_paper(raw: dict, *, influential: bool, intents: list | None) -> dict | None:
    if not raw or not raw.get("title"):
        return None
    ext = raw.get("externalIds") or {}
    arxiv_id = ext.get("ArXiv") or f"s2:{raw.get('paperId', '')}"
    doi = ext.get("DOI", "") or ""
    authors = ", ".join(a.get("name", "") for a in (raw.get("authors") or []) if a.get("name"))
    if ext.get("ArXiv"):
        url = f"https://arxiv.org/abs/{ext['ArXiv']}"
    else:
        url = f"https://www.semanticscholar.org/paper/{raw.get('paperId', '')}"
    return {
        "arxiv_id": arxiv_id,
        "paper_id": arxiv_id,
        "source": "semantic_scholar",
        "title": raw.get("title", ""),
        "authors": authors,
        "abstract": raw.get("abstract") or "",
        "url": url,
        "published": str(raw.get("year") or ""),
        "venue": raw.get("venue") or "",
        "summary": "", "key_insight": "", "method": "", "contribution": "",
        "math_concepts": [], "cited_works": [],
        "citation_count": raw.get("citationCount") or 0,
        "influential_citation_count": raw.get("influentialCitationCount") or 0,
        "doi": doi,
        "_edge_influential": bool(influential),
        "_edge_intents": intents or [],
    }


def _fetch(paper_id: str, kind: str, cfg: dict, limit: int) -> list[dict]:
    """kind is 'references' or 'citations'. Returns mapped papers, [] on failure."""
    url = f"{_S2_BASE}/paper/{_s2_id(paper_id)}/{kind}"
    params = {"fields": _FIELDS, "limit": limit}
    inner_key = "citedPaper" if kind == "references" else "citingPaper"
    try:
        resp = httpx_get_with_retry(url, params=params, headers=_headers(cfg), timeout=20)
        rows = resp.json().get("data", [])
    except httpx.HTTPError as e:
        log.debug("S2 %s failed for %s: %s", kind, paper_id, e)
        return []
    except Exception as e:  # JSON / shape errors
        log.debug("S2 %s parse failed for %s: %s", kind, paper_id, e)
        return []
    out = []
    for row in rows:
        p = _map_paper(row.get(inner_key) or {},
                       influential=row.get("isInfluential", False),
                       intents=row.get("intents"))
        if p:
            out.append(p)
    # influential first, then by citation count
    out.sort(key=lambda p: (p["_edge_influential"], p["influential_citation_count"]), reverse=True)
    return out


def get_references(paper_id: str, cfg: dict, *, limit: int = 50) -> list[dict]:
    """Papers that ``paper_id`` cites. [] on any failure."""
    return _fetch(paper_id, "references", cfg, limit)


def get_citations(paper_id: str, cfg: dict, *, limit: int = 50) -> list[dict]:
    """Papers that cite ``paper_id`` (influential first). [] on any failure."""
    return _fetch(paper_id, "citations", cfg, limit)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_semantic_scholar.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add src/paper_tracker/sources/semantic_scholar.py tests/test_semantic_scholar.py
git commit -m "feat: add Semantic Scholar citation source for deep-dig"
```

---

## Task 2: Registry — deep_dig tables + CRUD

**Files:**
- Modify: `src/paper_tracker/registry.py` (add migration call in `__init__`, new migration method, CRUD methods)
- Test: `tests/test_deep_dig_registry.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_deep_dig_registry.py`:
```python
"""Unit tests for registry.py — deep_dig_sessions / deep_dig_messages CRUD."""

from __future__ import annotations

import json

import pytest

from paper_tracker.registry import Registry


@pytest.fixture()
def reg(tmp_path):
    r = Registry(str(tmp_path))
    r.create_topic({
        "id": "tp-test", "name": "Test Topic",
        "arxiv_keywords": ["test"], "arxiv_categories": [], "github_keywords": [],
    })
    yield r
    r.close()


class TestCreateGet:
    def test_create_defaults(self, reg):
        s = reg.create_deep_dig_session("tp-test", "2501.00001", "My Paper")
        assert s["id"].startswith("dd-")
        assert s["topic_id"] == "tp-test"
        assert s["seed_paper_id"] == "2501.00001"
        assert s["seed_title"] == "My Paper"
        assert s["status"] == "exploring"
        assert s["language"] == "en"
        assert s["insights_md"] == ""
        assert s["dig_corpus_json"] == []
        assert s["graph_json"] == {}
        assert s["ideas_json"] == []
        assert s["started_at"] is not None

    def test_get_wrong_topic_none(self, reg):
        s = reg.create_deep_dig_session("tp-test", "2501.00001")
        assert reg.get_deep_dig_session("other", s["id"]) is None

    def test_unique_ids(self, reg):
        a = reg.create_deep_dig_session("tp-test", "2501.00001")
        b = reg.create_deep_dig_session("tp-test", "2501.00001")
        assert a["id"] != b["id"]


class TestUpdate:
    def test_update_json_fields_serialized(self, reg):
        s = reg.create_deep_dig_session("tp-test", "2501.00001")
        corpus = [{"index": "P1", "id": "a1", "title": "A"}]
        graph = {"nodes": [{"id": "a1"}], "edges": []}
        ideas = [{"title": "Idea", "novelty_score": 8}]
        reg.update_deep_dig_session("tp-test", s["id"], {
            "status": "insights_ready",
            "insights_md": "# memo",
            "dig_corpus_json": corpus,
            "graph_json": graph,
            "ideas_json": ideas,
            "steering_notes": "focus on theory",
        })
        got = reg.get_deep_dig_session("tp-test", s["id"])
        assert got["status"] == "insights_ready"
        assert got["insights_md"] == "# memo"
        assert got["dig_corpus_json"] == corpus
        assert got["graph_json"] == graph
        assert got["ideas_json"] == ideas
        assert got["steering_notes"] == "focus on theory"

    def test_update_accepts_prejson_strings(self, reg):
        s = reg.create_deep_dig_session("tp-test", "2501.00001")
        reg.update_deep_dig_session("tp-test", s["id"], {"ideas_json": json.dumps([{"title": "X"}])})
        got = reg.get_deep_dig_session("tp-test", s["id"])
        assert got["ideas_json"] == [{"title": "X"}]

    def test_update_ignores_unknown(self, reg):
        s = reg.create_deep_dig_session("tp-test", "2501.00001")
        reg.update_deep_dig_session("tp-test", s["id"], {"bogus": 1})
        got = reg.get_deep_dig_session("tp-test", s["id"])
        assert "bogus" not in got


class TestListDelete:
    def test_list_desc(self, reg):
        a = reg.create_deep_dig_session("tp-test", "2501.00001")
        b = reg.create_deep_dig_session("tp-test", "2501.00002")
        ids = [s["id"] for s in reg.list_deep_dig_sessions("tp-test")]
        assert ids == [b["id"], a["id"]]

    def test_delete_cascades_messages(self, reg):
        s = reg.create_deep_dig_session("tp-test", "2501.00001")
        reg.add_deep_dig_message("tp-test", s["id"], "user", "Q")
        reg.add_deep_dig_message("tp-test", s["id"], "assistant", "A")
        assert len(reg.list_deep_dig_messages("tp-test", s["id"])) == 2
        assert reg.delete_deep_dig_session("tp-test", s["id"]) is True
        assert reg.get_deep_dig_session("tp-test", s["id"]) is None
        assert reg.list_deep_dig_messages("tp-test", s["id"]) == []


class TestMessages:
    def test_add_and_order_asc(self, reg):
        s = reg.create_deep_dig_session("tp-test", "2501.00001")
        m1 = reg.add_deep_dig_message("tp-test", s["id"], "user", "Q1")
        m2 = reg.add_deep_dig_message("tp-test", s["id"], "assistant", "", status="pending")
        assert m1["id"].startswith("ddm-")
        msgs = reg.list_deep_dig_messages("tp-test", s["id"])
        assert [m["id"] for m in msgs] == [m1["id"], m2["id"]]

    def test_update_message(self, reg):
        s = reg.create_deep_dig_session("tp-test", "2501.00001")
        m = reg.add_deep_dig_message("tp-test", s["id"], "assistant", "", status="pending")
        reg.update_deep_dig_message(m["id"], {"content": "done", "status": "completed"})
        got = reg.get_deep_dig_message(m["id"])
        assert got["content"] == "done" and got["status"] == "completed"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_deep_dig_registry.py -v`
Expected: FAIL — `AttributeError: 'Registry' object has no attribute 'create_deep_dig_session'`

- [ ] **Step 3: Register the migration call**

In `src/paper_tracker/registry.py`, in `Registry.__init__`, add the migration call next to the other `self._migrate_*` calls (right after `self._migrate_deep_read_tables()`):
```python
        self._migrate_deep_read_tables()
        self._migrate_deep_dig_tables()
```

- [ ] **Step 4: Add the migration method + CRUD**

In `src/paper_tracker/registry.py`, add these methods to the `Registry` class (place them right after `get_deep_read_message`, near line 923):
```python
    # ----- deep_dig -------------------------------------------------------

    def _migrate_deep_dig_tables(self) -> None:
        """Create deep_dig_sessions and deep_dig_messages tables if missing."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS deep_dig_sessions (
                id TEXT PRIMARY KEY,
                topic_id TEXT NOT NULL,
                seed_paper_id TEXT NOT NULL,
                seed_title TEXT DEFAULT '',
                status TEXT DEFAULT 'exploring',
                language TEXT DEFAULT 'en',
                started_at TEXT,
                finished_at TEXT,
                insights_md TEXT DEFAULT '',
                dig_corpus_json TEXT DEFAULT '[]',
                graph_json TEXT DEFAULT '{}',
                steering_notes TEXT DEFAULT '',
                ideas_json TEXT DEFAULT '[]',
                error_message TEXT DEFAULT '',
                FOREIGN KEY (topic_id) REFERENCES topics(id)
            );
            CREATE TABLE IF NOT EXISTS deep_dig_messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                topic_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                status TEXT DEFAULT 'completed',
                created_at TEXT,
                FOREIGN KEY (session_id) REFERENCES deep_dig_sessions(id)
            );
        """)
        self._conn.commit()

    def _parse_deep_dig_row(self, row) -> dict:
        d = dict(row)
        try:
            d["dig_corpus_json"] = json.loads(d.get("dig_corpus_json", "[]") or "[]")
        except (json.JSONDecodeError, TypeError):
            d["dig_corpus_json"] = []
        try:
            d["ideas_json"] = json.loads(d.get("ideas_json", "[]") or "[]")
        except (json.JSONDecodeError, TypeError):
            d["ideas_json"] = []
        try:
            d["graph_json"] = json.loads(d.get("graph_json", "{}") or "{}")
        except (json.JSONDecodeError, TypeError):
            d["graph_json"] = {}
        return d

    def create_deep_dig_session(self, topic_id: str, seed_paper_id: str,
                                seed_title: str = "", language: str = "en") -> dict:
        import uuid
        session_id = f"dd-{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                """INSERT INTO deep_dig_sessions
                   (id, topic_id, seed_paper_id, seed_title, status, started_at, language)
                   VALUES (?, ?, ?, ?, 'exploring', ?, ?)""",
                (session_id, topic_id, seed_paper_id, seed_title, now, language),
            )
            self._conn.commit()
        return self.get_deep_dig_session(topic_id, session_id)

    def get_deep_dig_session(self, topic_id: str, session_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM deep_dig_sessions WHERE id = ? AND topic_id = ?",
            (session_id, topic_id),
        ).fetchone()
        return self._parse_deep_dig_row(row) if row else None

    def list_deep_dig_sessions(self, topic_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM deep_dig_sessions WHERE topic_id = ? ORDER BY started_at DESC",
            (topic_id,),
        ).fetchall()
        return [self._parse_deep_dig_row(r) for r in rows]

    def update_deep_dig_session(self, topic_id: str, session_id: str, updates: dict) -> None:
        allowed = {
            "status", "finished_at", "insights_md", "dig_corpus_json",
            "graph_json", "steering_notes", "ideas_json", "error_message",
        }
        json_fields = {"dig_corpus_json", "graph_json", "ideas_json"}
        fields, params = [], []
        for key, val in updates.items():
            if key in allowed:
                fields.append(f"{key} = ?")
                if key in json_fields and not isinstance(val, str):
                    params.append(json.dumps(val))
                else:
                    params.append(val)
        if not fields:
            return
        params.extend([session_id, topic_id])
        with self._lock:
            self._conn.execute(
                f"UPDATE deep_dig_sessions SET {', '.join(fields)} WHERE id = ? AND topic_id = ?",
                params,
            )
            self._conn.commit()

    def delete_deep_dig_session(self, topic_id: str, session_id: str) -> bool:
        with self._lock:
            self._conn.execute(
                "DELETE FROM deep_dig_messages WHERE session_id = ? AND topic_id = ?",
                (session_id, topic_id),
            )
            cur = self._conn.execute(
                "DELETE FROM deep_dig_sessions WHERE id = ? AND topic_id = ?",
                (session_id, topic_id),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def add_deep_dig_message(self, topic_id: str, session_id: str, role: str,
                             content: str, status: str = "completed") -> dict:
        import uuid
        msg_id = f"ddm-{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                """INSERT INTO deep_dig_messages (id, session_id, topic_id, role, content, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (msg_id, session_id, topic_id, role, content, status, now),
            )
            self._conn.commit()
        return self.get_deep_dig_message(msg_id)

    def list_deep_dig_messages(self, topic_id: str, session_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM deep_dig_messages WHERE session_id = ? AND topic_id = ? ORDER BY created_at ASC",
            (session_id, topic_id),
        ).fetchall()
        return [dict(r) for r in rows]

    def update_deep_dig_message(self, msg_id: str, updates: dict) -> None:
        allowed = {"content", "status"}
        fields, params = [], []
        for key, val in updates.items():
            if key in allowed:
                fields.append(f"{key} = ?")
                params.append(val)
        if not fields:
            return
        params.append(msg_id)
        with self._lock:
            self._conn.execute(
                f"UPDATE deep_dig_messages SET {', '.join(fields)} WHERE id = ?", params,
            )
            self._conn.commit()

    def get_deep_dig_message(self, msg_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM deep_dig_messages WHERE id = ?", (msg_id,)
        ).fetchone()
        return dict(row) if row else None
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_deep_dig_registry.py -v`
Expected: PASS (all tests)

- [ ] **Step 6: Commit**

```bash
git add src/paper_tracker/registry.py tests/test_deep_dig_registry.py
git commit -m "feat: add deep_dig_sessions/messages registry tables + CRUD"
```

---

## Task 3: Registry — recover_stale_tasks coverage

**Files:**
- Modify: `src/paper_tracker/registry.py` (`recover_stale_tasks`)
- Test: `tests/test_deep_dig_registry.py` (append a class)

deep_dig "running-like" statuses are `exploring` and `generating_ideas`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_deep_dig_registry.py`:
```python
class TestRecoverStale:
    def test_recovers_exploring_and_generating(self, reg):
        s1 = reg.create_deep_dig_session("tp-test", "2501.00001")  # exploring
        s2 = reg.create_deep_dig_session("tp-test", "2501.00002")
        reg.update_deep_dig_session("tp-test", s2["id"], {"status": "generating_ideas"})
        s3 = reg.create_deep_dig_session("tp-test", "2501.00003")
        reg.update_deep_dig_session("tp-test", s3["id"], {"status": "insights_ready"})
        m = reg.add_deep_dig_message("tp-test", s1["id"], "assistant", "", status="pending")

        counts = reg.recover_stale_tasks()
        assert counts.get("deep_dig_sessions", 0) == 2
        assert counts.get("deep_dig_messages", 0) == 1
        assert reg.get_deep_dig_session("tp-test", s1["id"])["status"] == "failed"
        assert reg.get_deep_dig_session("tp-test", s2["id"])["status"] == "failed"
        assert reg.get_deep_dig_session("tp-test", s3["id"])["status"] == "insights_ready"
        assert reg.get_deep_dig_message(m["id"])["status"] == "failed"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_deep_dig_registry.py::TestRecoverStale -v`
Expected: FAIL — `assert 0 == 2` (deep_dig not yet handled).

- [ ] **Step 3: Extend `recover_stale_tasks`**

In `src/paper_tracker/registry.py`, inside `recover_stale_tasks`, add these two blocks right before `self._conn.commit()` (after the deep_read_messages block):
```python
        # deep_dig_sessions (exploring/generating_ideas → failed)
        cur = self._conn.execute(
            """UPDATE deep_dig_sessions SET status = 'failed', finished_at = ?,
               error_message = 'Interrupted: server process restarted while running.'
               WHERE status IN ('exploring', 'generating_ideas')""",
            (now,),
        )
        counts["deep_dig_sessions"] = cur.rowcount

        # deep_dig_messages (pending/generating → failed)
        cur = self._conn.execute(
            "UPDATE deep_dig_messages SET status = 'failed' WHERE status IN ('pending', 'generating')",
        )
        counts["deep_dig_messages"] = cur.rowcount
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_deep_dig_registry.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/paper_tracker/registry.py tests/test_deep_dig_registry.py
git commit -m "feat: recover stale deep_dig sessions/messages on startup"
```

---

## Task 4: Config — `[deep_dig]` section

**Files:**
- Modify: `src/paper_tracker/config.py` (`_default_deep_dig`, wire into `from_topic`)
- Modify: `config.toml` (add `[deep_dig]`)
- Test: `tests/test_config_deep_dig.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_config_deep_dig.py`:
```python
from paper_tracker import config


def test_default_deep_dig_keys():
    d = config._default_deep_dig()
    assert d["max_turns"] == 18
    assert d["wall_clock"] == 900
    assert d["max_papers"] == 50
    assert d["per_tool_topk"] == 8
    assert d["s2_api_key"] == ""


def test_from_topic_includes_deep_dig():
    topic = {"name": "T", "arxiv_keywords": [], "arxiv_categories": [], "github_keywords": []}
    cfg = config.from_topic(topic, {"deep_dig": {"max_turns": 5}})
    assert cfg["deep_dig"]["max_turns"] == 5          # override honored
    assert cfg["deep_dig"]["max_papers"] == 50        # default filled
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config_deep_dig.py -v`
Expected: FAIL — `AttributeError: module 'paper_tracker.config' has no attribute '_default_deep_dig'`

- [ ] **Step 3: Implement in `config.py`**

Add the default function near `_default_summarizer` in `src/paper_tracker/config.py`:
```python
def _default_deep_dig() -> dict:
    return {
        "s2_api_key": "",
        "claude_model": "opus",
        "max_turns": 18,
        "wall_clock": 900,
        "max_papers": 50,
        "per_tool_topk": 8,
    }
```

In `from_topic()`, add a `deep_dig` key to the returned dict, right after the `"summarizer"` line (line 73). `base_cfg` is guaranteed non-None at that point, so mirror the summarizer style exactly:
```python
        "summarizer": {**_default_summarizer(), **base_cfg.get("summarizer", {})},
        "deep_dig": {**_default_deep_dig(), **base_cfg.get("deep_dig", {})},
```

- [ ] **Step 4: Add to `config.toml`**

Append to `config.toml`:
```toml
[deep_dig]
s2_api_key = ""      # optional; else env S2_API_KEY; works empty (lower rate limit)
claude_model = "opus"
max_turns = 18
wall_clock = 900     # seconds, agent loop budget
max_papers = 50      # total papers added to the collector per run
per_tool_topk = 8    # papers returned to the model per tool call
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_config_deep_dig.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/paper_tracker/config.py config.toml tests/test_config_deep_dig.py
git commit -m "feat: add [deep_dig] config section + defaults"
```

---

## Task 5: deep_dig_agent — collector + node/edge helpers

**Files:**
- Create: `src/paper_tracker/deep_dig_agent.py` (collector helpers only in this task)
- Test: `tests/test_deep_dig_agent.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_deep_dig_agent.py`:
```python
"""Tests for deep_dig_agent: collector, tool backends, entry points.
The Agent SDK loop is mocked — no real claude calls."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

from paper_tracker import deep_dig_agent as dda


def _paper(pid, title="T", **kw):
    base = {"arxiv_id": pid, "title": title, "abstract": "abs", "url": f"u/{pid}",
            "source": "semantic_scholar", "authors": "A", "venue": "", "published": "2020",
            "citation_count": 0, "influential_citation_count": 0, "math_concepts": []}
    base.update(kw)
    return base


class TestCollector:
    def test_add_node_assigns_sequential_index(self):
        c = dda._new_collector()
        n1 = dda._add_node(c, _paper("a"), "seed", [], 50)
        n2 = dda._add_node(c, _paper("b"), "references", ["a"], 50)
        assert n1["index"] == "P1" and n2["index"] == "P2"
        assert n2["parents"] == ["a"]

    def test_add_node_dedups_and_merges_parents(self):
        c = dda._new_collector()
        dda._add_node(c, _paper("a"), "seed", [], 50)
        first = dda._add_node(c, _paper("b"), "references", ["a"], 50)
        again = dda._add_node(c, _paper("b"), "citations", ["x"], 50)
        assert again["index"] == first["index"]          # same node
        assert "x" in again["parents"]                    # parent merged
        assert len(c["order"]) == 2

    def test_add_node_respects_max_papers(self):
        c = dda._new_collector()
        dda._add_node(c, _paper("a"), "seed", [], 2)
        dda._add_node(c, _paper("b"), "references", [], 2)
        assert dda._add_node(c, _paper("c"), "references", [], 2) is None
        assert len(c["nodes"]) == 2

    def test_resolve_index_or_id(self):
        c = dda._new_collector()
        dda._add_node(c, _paper("arxiv1"), "seed", [], 50)
        assert dda._resolve(c, "P1") == "arxiv1"
        assert dda._resolve(c, "[P1]") == "arxiv1"
        assert dda._resolve(c, "arxiv1") == "arxiv1"
        assert dda._resolve(c, "unknown99") == "unknown99"  # passthrough for traversal

    def test_strip_ungrounded_citations(self):
        c = dda._new_collector()
        dda._add_node(c, _paper("a"), "seed", [], 50)  # P1
        md = "Great point [P1] but also [P9] is fake."
        out = dda._strip_ungrounded(md, c)
        assert "[P1]" in out and "[P9]" not in out

    def test_build_dig_corpus_and_graph(self):
        c = dda._new_collector()
        dda._add_node(c, _paper("a"), "seed", [], 50)
        dda._add_node(c, _paper("b"), "references", ["a"], 50)
        dda._add_edge(c, "a", "b", "cites", True, ["methodology"])
        corpus = dda._build_dig_corpus(c)
        graph = dda._build_graph(c, 50)
        assert [p["index"] for p in corpus] == ["P1", "P2"]
        assert corpus[0]["id"] == "a"
        assert {n["id"] for n in graph["nodes"]} == {"a", "b"}
        assert graph["edges"][0]["kind"] == "cites" and graph["edges"][0]["influential"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_deep_dig_agent.py::TestCollector -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'paper_tracker.deep_dig_agent'`

- [ ] **Step 3: Implement the module header + collector**

Create `src/paper_tracker/deep_dig_agent.py`:
```python
"""Agentic deep-dig: explore a seed paper's neighborhood (citations + semantic +
math), synthesize insights, generate ideas, and answer follow-up questions.

Mirrors brainstorm_agent.py: a Claude Agent SDK tool-use loop driving the local
``claude`` CLI. Every tool records the papers and citation edges it touches into a
shared collector, so the dig-corpus, [Pn] provenance, and citation graph are
reconstructed deterministically from the tool trace — even though traversal is
model-driven.

Each public entry point returns a sentinel (None / "") on ANY failure so the
orchestrator (deep_dig.py) can fall back to a deterministic path.
"""

from __future__ import annotations

import asyncio
import logging
import re

from paper_tracker.storage import Storage
from paper_tracker.sources import arxiv, semantic_scholar
from paper_tracker import rag, crossdomain
from paper_tracker.brainstorm_agent import _format_card, _fetch_fulltext

log = logging.getLogger(__name__)

_MODEL = "opus"
_DEFAULTS = {"max_turns": 18, "wall_clock": 900, "max_papers": 50, "per_tool_topk": 8}
_TOOL_NAMES = ("get_references", "get_citations", "search_related",
               "search_math", "read_card", "read_full_text")


def _budget(cfg: dict) -> dict:
    dd = cfg.get("deep_dig", {})
    return {k: dd.get(k, v) for k, v in _DEFAULTS.items()}


# --------------------------------------------------------------------------
# Collector
# --------------------------------------------------------------------------

def _new_collector() -> dict:
    return {"nodes": {}, "edges": [], "order": []}


def _add_node(coll: dict, paper: dict, found_via: str, parents: list, max_papers: int):
    pid = paper.get("arxiv_id") or paper.get("paper_id") or ""
    if not pid:
        return None
    if pid in coll["nodes"]:
        node = coll["nodes"][pid]
        for par in parents or []:
            if par and par not in node["parents"]:
                node["parents"].append(par)
        return node
    if len(coll["nodes"]) >= max_papers:
        return None
    idx = f"P{len(coll['order']) + 1}"
    node = {
        "index": idx, "id": pid, "arxiv_id": pid, "source": paper.get("source", ""),
        "title": paper.get("title", ""), "authors": paper.get("authors", ""),
        "abstract": paper.get("abstract", ""), "url": paper.get("url", ""),
        "venue": paper.get("venue", ""), "year": paper.get("published", ""),
        "citation_count": paper.get("citation_count", 0),
        "influential_citation_count": paper.get("influential_citation_count", 0),
        "found_via": found_via, "parents": list(parents or []),
        "math_concepts": paper.get("math_concepts", []),
        "summary": paper.get("summary", ""), "key_insight": paper.get("key_insight", ""),
    }
    coll["nodes"][pid] = node
    coll["order"].append(pid)
    return node


def _add_edge(coll, frm, to, kind, influential=False, intents=None):
    coll["edges"].append({"from": frm, "to": to, "kind": kind,
                          "influential": bool(influential), "intents": intents or []})


def _resolve(coll: dict, ref: str) -> str:
    ref = (ref or "").strip().strip("[]")
    if ref in coll["nodes"]:
        return ref
    for pid, node in coll["nodes"].items():
        if node["index"] == ref:
            return pid
    return ref  # assume raw id for traversal


def _strip_ungrounded(md: str, coll: dict) -> str:
    valid = {n["index"] for n in coll["nodes"].values()}
    return re.sub(r"\[(P\d+)\]", lambda m: m.group(0) if m.group(1) in valid else "", md or "")


def _build_dig_corpus(coll: dict) -> list[dict]:
    out = []
    for pid in coll["order"]:
        n = coll["nodes"][pid]
        out.append({
            "index": n["index"], "id": n["id"], "source": n["source"], "title": n["title"],
            "authors": n["authors"], "venue": n["venue"], "url": n["url"],
            "abstract": n["abstract"], "summary": n["summary"], "key_insight": n["key_insight"],
            "math_concepts": n["math_concepts"], "found_via": n["found_via"],
            "influential_citation_count": n["influential_citation_count"],
        })
    return out


def _build_graph(coll: dict, max_papers: int) -> dict:
    kept = set(list(coll["nodes"].keys())[:max_papers])
    nodes = [{
        "id": n["id"], "index": n["index"], "title": n["title"], "found_via": n["found_via"],
        "citation_count": n["citation_count"],
        "influential_citation_count": n["influential_citation_count"],
    } for pid, n in coll["nodes"].items() if pid in kept]
    edges = [e for e in coll["edges"] if e["from"] in kept and e["to"] in kept]
    return {"nodes": nodes, "edges": edges}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_deep_dig_agent.py::TestCollector -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/paper_tracker/deep_dig_agent.py tests/test_deep_dig_agent.py
git commit -m "feat: deep_dig_agent collector + graph/corpus builders"
```

---

## Task 6: deep_dig_agent — tool backends

**Files:**
- Modify: `src/paper_tracker/deep_dig_agent.py`
- Test: `tests/test_deep_dig_agent.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_deep_dig_agent.py`:
```python
class TestToolBackends:
    def test_references_adds_nodes_and_edges(self):
        c = dda._new_collector()
        dda._add_node(c, _paper("seed"), "seed", [], 50)
        refs = [_paper("r1", _edge_influential=True, _edge_intents=["methodology"]),
                _paper("r2", _edge_influential=False, _edge_intents=[])]
        with patch("paper_tracker.deep_dig_agent.semantic_scholar.get_references", return_value=refs):
            out = dda._do_references(c, "P1", {}, topk=8, max_papers=50)
        assert "r1" in out and "r2" in out
        assert {e["kind"] for e in c["edges"]} == {"cites"}
        assert c["nodes"]["r1"]["found_via"] == "references"

    def test_citations_uses_cited_by_edge(self):
        c = dda._new_collector()
        dda._add_node(c, _paper("seed"), "seed", [], 50)
        with patch("paper_tracker.deep_dig_agent.semantic_scholar.get_citations",
                   return_value=[_paper("c1")]):
            dda._do_citations(c, "P1", {}, topk=8, max_papers=50)
        assert c["edges"][0]["kind"] == "cited_by"
        assert c["nodes"]["c1"]["found_via"] == "citations"

    def test_references_handles_failure(self):
        c = dda._new_collector()
        dda._add_node(c, _paper("seed"), "seed", [], 50)
        with patch("paper_tracker.deep_dig_agent.semantic_scholar.get_references", return_value=[]):
            out = dda._do_references(c, "P1", {}, topk=8, max_papers=50)
        assert "No" in out or "results" in out.lower()

    def test_search_related_uses_rag_then_keyword(self):
        c = dda._new_collector()
        store = MagicMock()
        store.get_all_arxiv.return_value = ([], 0)
        with patch("paper_tracker.deep_dig_agent.Storage", return_value=store), \
             patch("paper_tracker.deep_dig_agent.rag.search_papers",
                   return_value=[(_paper("s1"), 0.9)]):
            out = dda._do_search_related(c, "transport", "/d", "t1", topk=8, max_papers=50)
        assert "s1" in out
        assert c["nodes"]["s1"]["found_via"] == "search_related"

    def test_search_math_filters_corpus_to_math(self):
        c = dda._new_collector()
        store = MagicMock()
        store.get_all_arxiv.return_value = ([], 0)
        with patch("paper_tracker.deep_dig_agent.arxiv.search_by_query",
                   return_value=[_paper("m1", venue="math.OC")]), \
             patch("paper_tracker.deep_dig_agent.Storage", return_value=store), \
             patch("paper_tracker.deep_dig_agent.rag.search_papers",
                   return_value=[(_paper("c_math", venue="math.PR"), 0.8),
                                 (_paper("c_cs", venue="cs.LG"), 0.7)]):
            dda._do_search_math(c, "optimal transport", "/d", topk=8, max_papers=50)
        assert "m1" in c["nodes"]            # arxiv hit kept
        assert "c_math" in c["nodes"]        # math corpus hit kept
        assert "c_cs" not in c["nodes"]      # non-math corpus hit dropped

    def test_read_full_text_resolves_index(self):
        c = dda._new_collector()
        dda._add_node(c, _paper("2401.00001", source="arxiv"), "seed", [], 50)
        with patch("paper_tracker.deep_dig_agent._fetch_fulltext", return_value="BODY"):
            out = dda._do_read_full_text(c, "P1")
        assert out == "BODY"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_deep_dig_agent.py::TestToolBackends -v`
Expected: FAIL — `AttributeError: module 'paper_tracker.deep_dig_agent' has no attribute '_do_references'`

- [ ] **Step 3: Implement the tool backends**

Append to `src/paper_tracker/deep_dig_agent.py`:
```python
# --------------------------------------------------------------------------
# Tool backends (sync; run off the event loop via asyncio.to_thread)
# --------------------------------------------------------------------------

def _fmt_results(nodes: list[dict]) -> str:
    if not nodes:
        return "No new results."
    out = []
    for n in nodes:
        out.append(f"[{n['index']}] {n['title']} (id={n['id']}, via={n['found_via']}, "
                   f"cites={n['citation_count']}, infl={n['influential_citation_count']})\n"
                   f"  {(n.get('abstract') or '')[:240]}")
    return "\n\n".join(out)


def _do_references(coll, ref, cfg, *, topk, max_papers) -> str:
    pid = _resolve(coll, ref)
    refs = semantic_scholar.get_references(pid, cfg, limit=max(topk, 20))[:topk]
    added = []
    for p in refs:
        n = _add_node(coll, p, "references", [pid], max_papers)
        if n:
            _add_edge(coll, pid, n["id"], "cites",
                      p.get("_edge_influential"), p.get("_edge_intents"))
            added.append(n)
    return _fmt_results(added)


def _do_citations(coll, ref, cfg, *, topk, max_papers) -> str:
    pid = _resolve(coll, ref)
    cits = semantic_scholar.get_citations(pid, cfg, limit=max(topk, 20))[:topk]
    added = []
    for p in cits:
        n = _add_node(coll, p, "citations", [pid], max_papers)
        if n:
            _add_edge(coll, pid, n["id"], "cited_by",
                      p.get("_edge_influential"), p.get("_edge_intents"))
            added.append(n)
    return _fmt_results(added)


def _do_search_related(coll, query, data_dir, topic_id, *, topk, max_papers) -> str:
    papers: list[dict] = []
    for tid in (topic_id, crossdomain.CROSSDOMAIN_ID):
        try:
            store = Storage(data_dir, tid)
            try:
                hits = rag.search_papers(store, query, max_results=topk)
                if hits:
                    papers += [p for p, _ in hits]
                else:  # no embeddings → keyword fallback
                    kw, _ = store.get_all_arxiv(search=query, limit=topk)
                    papers += kw
            finally:
                store.close()
        except Exception as e:
            log.debug("search_related %s failed: %s", tid, e)
    added = []
    for p in papers[:topk]:
        n = _add_node(coll, p, "search_related", [], max_papers)
        if n:
            added.append(n)
    return _fmt_results(added)


def _do_search_math(coll, query, data_dir, *, topk, max_papers) -> str:
    papers: list[dict] = []
    try:
        papers += arxiv.search_by_query(query, max_results=topk)
    except Exception as e:
        log.debug("search_math arxiv failed: %s", e)
    try:
        store = Storage(data_dir, crossdomain.CROSSDOMAIN_ID)
        try:
            hits = rag.search_papers(store, query, max_results=topk)
            cand = [p for p, _ in hits] if hits else store.get_all_arxiv(search=query, limit=topk)[0]
        finally:
            store.close()
        papers += [p for p in cand if (p.get("venue", "") or "").startswith(("math.", "stat."))]
    except Exception as e:
        log.debug("search_math corpus failed: %s", e)
    added = []
    for p in papers[:topk]:
        n = _add_node(coll, p, "search_math", [], max_papers)
        if n:
            added.append(n)
    return _fmt_results(added)


def _do_read_card(coll, ref) -> str:
    pid = _resolve(coll, ref)
    n = coll["nodes"].get(pid)
    return _format_card(n) if n else f"No paper {ref!r} in the dig so far."


def _do_read_full_text(coll, ref) -> str:
    pid = _resolve(coll, ref)
    n = coll["nodes"].get(pid)
    if not n:
        return f"No paper {ref!r} in the dig so far."
    return _fetch_fulltext(n)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_deep_dig_agent.py::TestToolBackends -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/paper_tracker/deep_dig_agent.py tests/test_deep_dig_agent.py
git commit -m "feat: deep_dig_agent tool backends (citations/semantic/math/read)"
```

---

## Task 7: deep_dig_agent — agent loop + entry points

**Files:**
- Modify: `src/paper_tracker/deep_dig_agent.py`
- Test: `tests/test_deep_dig_agent.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_deep_dig_agent.py`:
```python
class TestEntryPoints:
    def test_explore_returns_memo_corpus_graph(self):
        async def fake_run(prompt, tools, model, max_turns):
            return "Insight grounded in [P1]. Also [P9] hallucinated."
        with patch.object(dda, "_build_tools", return_value=[]), \
             patch.object(dda, "_run_agent", fake_run):
            res = dda.explore_and_synthesize(_paper("seed"), data_dir="/d", topic_id="t1", cfg={})
        assert res is not None
        assert "[P1]" in res["insights_md"] and "[P9]" not in res["insights_md"]  # stripped
        assert res["dig_corpus"][0]["index"] == "P1"
        assert res["graph"]["nodes"][0]["id"] == "seed"

    def test_explore_returns_none_on_failure(self):
        async def boom(*a, **k):
            raise RuntimeError("sdk down")
        with patch.object(dda, "_build_tools", return_value=[]), \
             patch.object(dda, "_run_agent", boom):
            assert dda.explore_and_synthesize(_paper("seed"), data_dir="/d", topic_id="t1", cfg={}) is None

    def test_generate_ideas_returns_raw_or_empty(self):
        corpus = [{"index": "P1", "id": "a", "title": "A", "abstract": "x"}]
        async def fake_run(prompt, tools, model, max_turns):
            return '[{"title":"Idea"}]'
        with patch.object(dda, "_build_tools", return_value=[]), \
             patch.object(dda, "_run_agent", fake_run):
            out = dda.generate_ideas(insights_md="m", dig_corpus=corpus, steering_notes="",
                                     focus_ids=[], data_dir="/d", topic_id="t1", cfg={})
        assert out == '[{"title":"Idea"}]'

    def test_answer_question_returns_text(self):
        corpus = [{"index": "P1", "id": "a", "title": "A", "abstract": "x"}]
        async def fake_run(prompt, tools, model, max_turns):
            return "Here is the answer."
        with patch.object(dda, "_build_tools", return_value=[]), \
             patch.object(dda, "_run_agent", fake_run):
            out = dda.answer_question(question="why?", insights_md="m", ideas=[], dig_corpus=corpus,
                                      prior_messages=[], data_dir="/d", topic_id="t1", cfg={})
        assert out == "Here is the answer."
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_deep_dig_agent.py::TestEntryPoints -v`
Expected: FAIL — `AttributeError: ... has no attribute 'explore_and_synthesize'`

- [ ] **Step 3: Implement tools wiring, agent loop, preambles, entry points**

Append to `src/paper_tracker/deep_dig_agent.py`:
```python
# --------------------------------------------------------------------------
# Agent loop
# --------------------------------------------------------------------------

def _build_tools(ctx: dict):
    from claude_agent_sdk import tool

    coll = ctx["collector"]
    cfg = ctx["cfg"]
    b = ctx["budget"]
    on_event = ctx.get("on_event")

    def _emit(msg: str):
        if on_event:
            try:
                on_event(msg)
            except Exception:
                pass

    @tool("get_references", "Fetch the papers a paper cites (its references), by [Pn] index or arXiv id.", {"paper": str})
    async def get_references(args):
        text = await asyncio.to_thread(_do_references, coll, str(args.get("paper", "")), cfg,
                                       topk=b["per_tool_topk"], max_papers=b["max_papers"])
        _emit(f"get_references({args.get('paper')}) → {len(coll['nodes'])} papers total")
        return {"content": [{"type": "text", "text": text}]}

    @tool("get_citations", "Fetch the papers that cite a paper (influential first), by [Pn] index or arXiv id.", {"paper": str})
    async def get_citations(args):
        text = await asyncio.to_thread(_do_citations, coll, str(args.get("paper", "")), cfg,
                                       topk=b["per_tool_topk"], max_papers=b["max_papers"])
        _emit(f"get_citations({args.get('paper')}) → {len(coll['nodes'])} papers total")
        return {"content": [{"type": "text", "text": text}]}

    @tool("search_related", "Semantic search of the topic library + knowledge base for related papers.", {"query": str})
    async def search_related(args):
        text = await asyncio.to_thread(_do_search_related, coll, str(args.get("query", "")),
                                       ctx["data_dir"], ctx["topic_id"],
                                       topk=b["per_tool_topk"], max_papers=b["max_papers"])
        _emit(f"search_related({args.get('query')}) → {len(coll['nodes'])} papers total")
        return {"content": [{"type": "text", "text": text}]}

    @tool("search_math", "Search arXiv math + the math knowledge base for borrowable math tools/theory.", {"query": str})
    async def search_math(args):
        text = await asyncio.to_thread(_do_search_math, coll, str(args.get("query", "")),
                                       ctx["data_dir"], topk=b["per_tool_topk"], max_papers=b["max_papers"])
        _emit(f"search_math({args.get('query')}) → {len(coll['nodes'])} papers total")
        return {"content": [{"type": "text", "text": text}]}

    @tool("read_card", "Read the concept-card/abstract of a dig paper by [Pn] index or id.", {"paper": str})
    async def read_card(args):
        text = await asyncio.to_thread(_do_read_card, coll, str(args.get("paper", "")))
        return {"content": [{"type": "text", "text": text}]}

    @tool("read_full_text", "Fetch the original full text (first pages) of a dig paper by [Pn] index or id.", {"paper": str})
    async def read_full_text(args):
        text = await asyncio.to_thread(_do_read_full_text, coll, str(args.get("paper", "")))
        return {"content": [{"type": "text", "text": text}]}

    return [get_references, get_citations, search_related, search_math, read_card, read_full_text]


async def _run_agent(prompt: str, tools, model: str, max_turns: int) -> str:
    from claude_agent_sdk import query, create_sdk_mcp_server, ClaudeAgentOptions

    server = create_sdk_mcp_server(name="dig", version="1.0.0", tools=tools)
    options = ClaudeAgentOptions(
        mcp_servers={"dig": server},
        allowed_tools=[f"mcp__dig__{n}" for n in _TOOL_NAMES],
        model=model,
        max_turns=max_turns,
    )
    result_text, last_text = "", ""
    async for msg in query(prompt=prompt, options=options):
        if type(msg).__name__ == "ResultMessage":
            r = getattr(msg, "result", None)
            if r:
                result_text = str(r)
        content = getattr(msg, "content", None)
        if isinstance(content, list):
            chunk = "".join(getattr(bl, "text", "") or "" for bl in content)
            if chunk.strip():
                last_text = chunk
    return result_text or last_text


def _index_list(corpus_or_collector) -> str:
    if isinstance(corpus_or_collector, dict):  # collector
        items = [corpus_or_collector["nodes"][p] for p in corpus_or_collector["order"]]
    else:
        items = corpus_or_collector
    return "\n".join(f"- {p['index']}: {p.get('title', '')}" for p in items) or "(none yet)"


def _seed_from_corpus(corpus: list[dict]) -> dict:
    coll = _new_collector()
    for p in sorted(corpus, key=lambda x: int(str(x.get("index", "P0"))[1:] or 0)):
        pid = p.get("id") or p.get("arxiv_id") or ""
        if not pid:
            continue
        coll["nodes"][pid] = {
            "index": p.get("index"), "id": pid, "source": p.get("source", ""),
            "title": p.get("title", ""), "authors": p.get("authors", ""),
            "abstract": p.get("abstract", ""), "url": p.get("url", ""),
            "venue": p.get("venue", ""), "year": p.get("year", p.get("published", "")),
            "citation_count": p.get("citation_count", 0),
            "influential_citation_count": p.get("influential_citation_count", 0),
            "found_via": p.get("found_via", ""), "parents": [],
            "math_concepts": p.get("math_concepts", []),
            "summary": p.get("summary", ""), "key_insight": p.get("key_insight", ""),
        }
        coll["order"].append(pid)
    return coll


_EXPLORE_PREAMBLE = """You are a research analyst doing a DEEP DIG from one seed paper. You have TOOLS \
to explore its intellectual neighborhood and ground every claim in REAL papers.

Seed paper (already indexed as P1):
{index_list}

Tools:
- get_references(paper): papers the given paper cites (lineage / what it builds on).
- get_citations(paper): papers that cite it (influential first; what built on it).
- search_related(query): semantic search of the local library + knowledge base.
- search_math(query): arXiv math + math knowledge base for borrowable math tools/theory.
- read_card(paper) / read_full_text(paper): read a dig paper by its [Pn] index or id.

Budget: explore roughly {max_papers} papers. Prefer influential citations and high-relevance \
matches. Traverse a SECOND hop (get_references/get_citations on a non-seed [Pn]) only from the \
most pivotal papers. Use search_math to find theory the seed could borrow.

How to work:
1. From the seed, get_references and get_citations to map lineage.
2. Branch with search_related and search_math for cross-pollination.
3. read_full_text the few most pivotal papers.
4. THEN write an insights memo.

Your FINAL message must be ONLY the insights memo in markdown, citing papers by their [Pn] index \
(e.g. [P3]). Cover: key trends, emerging methods, cross-paper connections, research gaps & \
opportunities. No tool calls, no commentary outside the memo.
"""

_IDEAS_PREAMBLE = """You are generating NOVEL research ideas from a completed deep dig. You have TOOLS \
to re-read the dug papers and check prior art before proposing ideas.

Insights memo:
{insights}

User steering (may be empty): {steering}

Dug papers (read_card / read_full_text by [Pn]):
{index_list}

Focus especially on these papers: {focus}

Tools: read_card(paper), read_full_text(paper), search_related(query), search_math(query), \
get_references(paper), get_citations(paper).

Generate 3-5 concrete, novel ideas. For EACH: "title", "problem", "motivation", "method", \
"experiment_plan", "novelty_score" (1-10), "feasibility_score" (1-10). Each idea must tackle a \
DIFFERENT core problem. Ground motivation in the dug papers by [Pn].

Your FINAL message must be ONLY a JSON array of idea objects — no markdown fences, no commentary.
"""

_QA_PREAMBLE = """You are answering a follow-up question about a completed deep dig. You have TOOLS \
to dig further if needed.

Insights memo:
{insights}

Generated ideas (JSON, may be empty): {ideas}

Dug papers (read_card / read_full_text by [Pn]):
{index_list}

Prior conversation:
{history}

Tools: read_card, read_full_text, search_related, search_math, get_references, get_citations.

Answer the user's question grounded in the dig. Dig further with tools when useful.

================= QUESTION =================
{question}

Your FINAL message must be ONLY the answer in markdown.
"""


def _run(ctx: dict, prompt: str) -> str:
    tools = _build_tools(ctx)
    b = ctx["budget"]
    return asyncio.run(asyncio.wait_for(
        _run_agent(prompt, tools, ctx["cfg"].get("deep_dig", {}).get("claude_model", _MODEL),
                   b["max_turns"]),
        timeout=b["wall_clock"]))


def explore_and_synthesize(seed: dict, *, data_dir, topic_id, cfg, on_event=None) -> dict | None:
    try:
        b = _budget(cfg)
        coll = _new_collector()
        _add_node(coll, seed, "seed", [], b["max_papers"])
        ctx = {"collector": coll, "data_dir": data_dir, "topic_id": topic_id,
               "cfg": cfg, "budget": b, "on_event": on_event}
        prompt = _EXPLORE_PREAMBLE.format(index_list=_index_list(coll), max_papers=b["max_papers"])
        raw = _run(ctx, prompt)
        memo = _strip_ungrounded((raw or "").strip(), coll)
        if not memo:
            return None
        return {"insights_md": memo, "dig_corpus": _build_dig_corpus(coll),
                "graph": _build_graph(coll, b["max_papers"])}
    except Exception as e:
        log.warning("Agentic explore failed (%s) — caller will fall back", e)
        return None


def generate_ideas(*, insights_md, dig_corpus, steering_notes, focus_ids,
                   data_dir, topic_id, cfg, on_event=None) -> str:
    try:
        b = _budget(cfg)
        coll = _seed_from_corpus(dig_corpus)
        ctx = {"collector": coll, "data_dir": data_dir, "topic_id": topic_id,
               "cfg": cfg, "budget": b, "on_event": on_event}
        focus = ", ".join(focus_ids) if focus_ids else "(none specified)"
        prompt = _IDEAS_PREAMBLE.format(insights=insights_md[:8000], steering=steering_notes or "(none)",
                                        index_list=_index_list(dig_corpus), focus=focus)
        return (_run(ctx, prompt) or "").strip()
    except Exception as e:
        log.warning("Agentic idea gen failed (%s) — caller will fall back", e)
        return ""


def answer_question(*, question, insights_md, ideas, dig_corpus, prior_messages,
                    data_dir, topic_id, cfg, on_event=None) -> str:
    try:
        import json as _json
        b = _budget(cfg)
        coll = _seed_from_corpus(dig_corpus)
        ctx = {"collector": coll, "data_dir": data_dir, "topic_id": topic_id,
               "cfg": cfg, "budget": b, "on_event": on_event}
        history = "\n".join(f"{m.get('role')}: {m.get('content')}" for m in (prior_messages or []))
        prompt = _QA_PREAMBLE.format(insights=insights_md[:6000], ideas=_json.dumps(ideas)[:3000],
                                     index_list=_index_list(dig_corpus), history=history[:4000],
                                     question=question)
        return (_run(ctx, prompt) or "").strip()
    except Exception as e:
        log.warning("Agentic QA failed (%s) — caller will fall back", e)
        return ""
```

- [ ] **Step 4: Run the full agent test file**

Run: `uv run pytest tests/test_deep_dig_agent.py -v`
Expected: PASS (all classes)

- [ ] **Step 5: Commit**

```bash
git add src/paper_tracker/deep_dig_agent.py tests/test_deep_dig_agent.py
git commit -m "feat: deep_dig_agent loop + explore/ideas/qa entry points"
```

---

## Task 8: deep_dig orchestration — resolve_seed

**Files:**
- Create: `src/paper_tracker/deep_dig.py`
- Test: `tests/test_deep_dig.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_deep_dig.py`:
```python
"""Tests for deep_dig orchestration (agent mocked)."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

from paper_tracker import deep_dig


def _paper(pid, **kw):
    base = {"arxiv_id": pid, "title": "T", "abstract": "a", "url": f"u/{pid}", "source": "arxiv"}
    base.update(kw)
    return base


class TestResolveSeed:
    def test_library_first(self):
        store = MagicMock()
        store.get_arxiv.return_value = _paper("2501.00001")
        with patch("paper_tracker.deep_dig.Storage", return_value=store):
            p = deep_dig.resolve_seed("/d", "t1", {}, paper_id="2501.00001")
        assert p["arxiv_id"] == "2501.00001"

    def test_falls_back_to_arxiv_fetch(self):
        store = MagicMock()
        store.get_arxiv.return_value = None
        with patch("paper_tracker.deep_dig.Storage", return_value=store), \
             patch("paper_tracker.deep_dig.arxiv.extract_arxiv_id", return_value="2401.00009"), \
             patch("paper_tracker.deep_dig.arxiv.fetch_by_id", return_value=_paper("2401.00009")):
            p = deep_dig.resolve_seed("/d", "t1", {}, seed_query="https://arxiv.org/abs/2401.00009")
        assert p["arxiv_id"] == "2401.00009"

    def test_falls_back_to_pdf(self):
        with patch("paper_tracker.deep_dig.arxiv.extract_arxiv_id", return_value=""), \
             patch("paper_tracker.deep_dig.pdf.is_pdf_url", return_value=True), \
             patch("paper_tracker.deep_dig.pdf.fetch_pdf_paper", return_value=_paper("pdf:abc")):
            p = deep_dig.resolve_seed("/d", "t1", {}, seed_query="http://x/y.pdf")
        assert p["arxiv_id"] == "pdf:abc"

    def test_none_when_unresolvable(self):
        with patch("paper_tracker.deep_dig.arxiv.extract_arxiv_id", return_value=""), \
             patch("paper_tracker.deep_dig.pdf.is_pdf_url", return_value=False):
            assert deep_dig.resolve_seed("/d", "t1", {}, seed_query="not a paper") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_deep_dig.py::TestResolveSeed -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'paper_tracker.deep_dig'`

- [ ] **Step 3: Implement the module header + resolve_seed**

Create `src/paper_tracker/deep_dig.py`:
```python
"""Deep-dig orchestration: resolve a seed paper and run each phase (explore →
ideas → Q&A) via the agentic path, falling back to a deterministic path when the
Claude Agent SDK is unavailable or the agent fails. Pure compute — persistence
and progress live in server.py (mirroring deep_read.py)."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from paper_tracker import deep_dig_agent, rag, crossdomain, summarizer, insights
from paper_tracker.llm import call_cli
from paper_tracker.brainstorm import _parse_ideas
from paper_tracker.sources import arxiv, pdf, semantic_scholar
from paper_tracker.storage import Storage

log = logging.getLogger(__name__)


def resolve_seed(data_dir: str, topic_id: str, cfg: dict, *,
                 paper_id: str | None = None, seed_query: str | None = None) -> dict | None:
    """Resolve a seed paper: library → arxiv.fetch_by_id → pdf.fetch_pdf_paper."""
    if paper_id:
        store = Storage(data_dir, topic_id)
        try:
            p = store.get_arxiv(paper_id)
        finally:
            store.close()
        if p:
            return p
    q = (seed_query or paper_id or "").strip()
    if not q:
        return None
    aid = arxiv.extract_arxiv_id(q)
    if aid:
        try:
            p = arxiv.fetch_by_id(aid)
            if p:
                return p
        except Exception as e:
            log.debug("arxiv.fetch_by_id failed for %s: %s", aid, e)
    if pdf.is_pdf_url(q):
        try:
            p = pdf.fetch_pdf_paper(q, cfg)
            if p:
                return p
        except Exception as e:
            log.debug("pdf.fetch_pdf_paper failed for %s: %s", q, e)
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_deep_dig.py::TestResolveSeed -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/paper_tracker/deep_dig.py tests/test_deep_dig.py
git commit -m "feat: deep_dig.resolve_seed (library/arxiv/pdf)"
```

---

## Task 9: deep_dig orchestration — phase runners + deterministic fallbacks

**Files:**
- Modify: `src/paper_tracker/deep_dig.py`
- Test: `tests/test_deep_dig.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_deep_dig.py`:
```python
class TestRunExplore:
    def test_uses_agent_result_when_present(self):
        agent_res = {"insights_md": "memo", "dig_corpus": [{"index": "P1"}], "graph": {"nodes": []}}
        with patch("paper_tracker.deep_dig.deep_dig_agent.explore_and_synthesize", return_value=agent_res):
            out = deep_dig.run_explore(_paper("seed"), "Topic", "/d", "t1", {})
        assert out["insights_md"] == "memo"

    def test_falls_back_when_agent_none(self):
        with patch("paper_tracker.deep_dig.deep_dig_agent.explore_and_synthesize", return_value=None), \
             patch("paper_tracker.deep_dig._explore_deterministic",
                   return_value={"insights_md": "fallback", "dig_corpus": [], "graph": {}}) as fb:
            out = deep_dig.run_explore(_paper("seed"), "Topic", "/d", "t1", {})
        assert out["insights_md"] == "fallback"
        fb.assert_called_once()


class TestRunIdeas:
    def test_uses_agent_then_parses(self):
        with patch("paper_tracker.deep_dig.deep_dig_agent.generate_ideas", return_value='[{"title":"X"}]'), \
             patch("paper_tracker.deep_dig._parse_ideas", return_value=[{"title": "X"}]) as pi:
            out = deep_dig.run_ideas("memo", [{"index": "P1"}], "", [], "Topic", "/d", "t1", {})
        assert out == [{"title": "X"}]
        pi.assert_called_once()

    def test_falls_back_to_oneshot_when_agent_empty(self):
        with patch("paper_tracker.deep_dig.deep_dig_agent.generate_ideas", return_value=""), \
             patch("paper_tracker.deep_dig.call_cli", return_value='[{"title":"Y"}]'), \
             patch("paper_tracker.deep_dig._parse_ideas", return_value=[{"title": "Y"}]):
            out = deep_dig.run_ideas("memo", [{"index": "P1"}], "", [], "Topic", "/d", "t1", {})
        assert out == [{"title": "Y"}]


class TestRunQa:
    def test_uses_agent(self):
        with patch("paper_tracker.deep_dig.deep_dig_agent.answer_question", return_value="answer"):
            out = deep_dig.run_qa("q?", "memo", [], [{"index": "P1"}], [], "/d", "t1", {})
        assert out["content"] == "answer"

    def test_falls_back_to_call_cli(self):
        with patch("paper_tracker.deep_dig.deep_dig_agent.answer_question", return_value=""), \
             patch("paper_tracker.deep_dig.call_cli", return_value="oneshot answer"):
            out = deep_dig.run_qa("q?", "memo", [], [{"index": "P1"}], [], "/d", "t1", {})
        assert out["content"] == "oneshot answer"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_deep_dig.py::TestRunExplore -v`
Expected: FAIL — `AttributeError: module 'paper_tracker.deep_dig' has no attribute 'run_explore'`

- [ ] **Step 3: Implement phase runners + deterministic fallbacks**

Append to `src/paper_tracker/deep_dig.py`:
```python
_IDEAS_PROMPT = """\
You are a senior researcher generating novel ideas from a deep-dig of one paper's neighborhood.

## Insights memo
{insights}

## User steering (may be empty)
{steering}

## Dug papers
{papers}

Generate 3-5 concrete, novel research ideas. For EACH provide: "title", "problem", "motivation", \
"method", "experiment_plan", "novelty_score" (1-10), "feasibility_score" (1-10). Each idea must \
tackle a DIFFERENT core problem. Reply ONLY with a JSON array of idea objects."""

_QA_PROMPT = """\
Answer the question about a deep-dig, grounded in the memo + papers below.

## Insights memo
{insights}

## Dug papers
{papers}

## Conversation so far
{history}

## Question
{question}

Reply in markdown."""


def _corpus_brief(dig_corpus: list[dict], limit: int = 40) -> str:
    lines = []
    for p in dig_corpus[:limit]:
        lines.append(f"[{p.get('index')}] {p.get('title')} ({p.get('venue') or p.get('found_via')}) "
                     f"— {(p.get('summary') or p.get('abstract') or '')[:200]}")
    return "\n".join(lines)


def run_explore(seed: dict, topic_name: str, data_dir: str, topic_id: str, cfg: dict,
                on_event=None) -> dict:
    res = deep_dig_agent.explore_and_synthesize(
        seed, data_dir=data_dir, topic_id=topic_id, cfg=cfg, on_event=on_event)
    if res and res.get("insights_md"):
        return res
    log.info("deep_dig: explore falling back to deterministic path")
    return _explore_deterministic(seed, topic_name, data_dir, topic_id, cfg)


def _explore_deterministic(seed: dict, topic_name: str, data_dir: str, topic_id: str,
                           cfg: dict) -> dict:
    b = deep_dig_agent._budget(cfg)
    coll = deep_dig_agent._new_collector()
    deep_dig_agent._add_node(coll, seed, "seed", [], b["max_papers"])
    seed_id = seed.get("arxiv_id") or seed.get("paper_id") or ""
    topk = b["per_tool_topk"]
    # citation chain
    for p in semantic_scholar.get_references(seed_id, cfg, limit=topk):
        n = deep_dig_agent._add_node(coll, p, "references", [seed_id], b["max_papers"])
        if n:
            deep_dig_agent._add_edge(coll, seed_id, n["id"], "cites", p.get("_edge_influential"))
    for p in semantic_scholar.get_citations(seed_id, cfg, limit=topk):
        n = deep_dig_agent._add_node(coll, p, "citations", [seed_id], b["max_papers"])
        if n:
            deep_dig_agent._add_edge(coll, seed_id, n["id"], "cited_by", p.get("_edge_influential"))
    # semantic + math (reuse the agent tool backends against the same collector)
    seed_text = f"{seed.get('title', '')} {seed.get('abstract', '')}"
    deep_dig_agent._do_search_related(coll, seed_text, data_dir, topic_id,
                                      topk=topk, max_papers=b["max_papers"])
    math_q = ", ".join(seed.get("math_concepts") or []) or seed.get("title", "")
    deep_dig_agent._do_search_math(coll, math_q, data_dir, topk=topk, max_papers=b["max_papers"])
    # summarize the dug papers (skip the seed if already summarized)
    papers = [coll["nodes"][pid] for pid in coll["order"]]
    try:
        summarizer.summarize_papers([p for p in papers if not p.get("summary")], cfg)
    except Exception as e:
        log.debug("deterministic summarize failed: %s", e)
    # synthesize insights via the existing agentic insights writer
    memo = ""
    try:
        tmp = Path(tempfile.mkdtemp(prefix="deep_dig_"))
        path = insights.generate_agentic(papers, topic_name, tmp, cfg)
        if path and Path(path).exists():
            memo = Path(path).read_text(encoding="utf-8")
    except Exception as e:
        log.warning("deterministic insights failed: %s", e)
    if not memo:
        memo = "## Deep-Dig Insights\n\n(LLM unavailable — papers gathered below.)\n\n" + \
               _corpus_brief(deep_dig_agent._build_dig_corpus(coll))
    return {"insights_md": memo, "dig_corpus": deep_dig_agent._build_dig_corpus(coll),
            "graph": deep_dig_agent._build_graph(coll, b["max_papers"])}


def run_ideas(insights_md: str, dig_corpus: list[dict], steering_notes: str, focus_ids: list[str],
              topic_name: str, data_dir: str, topic_id: str, cfg: dict, on_event=None) -> list[dict]:
    raw = deep_dig_agent.generate_ideas(
        insights_md=insights_md, dig_corpus=dig_corpus, steering_notes=steering_notes,
        focus_ids=focus_ids, data_dir=data_dir, topic_id=topic_id, cfg=cfg, on_event=on_event)
    if not raw:
        log.info("deep_dig: ideas falling back to one-shot")
        prompt = _IDEAS_PROMPT.format(insights=insights_md[:8000],
                                      steering=steering_notes or "(none)",
                                      papers=_corpus_brief(dig_corpus))
        raw = call_cli(prompt, cfg, model="opus", timeout=600) or ""
    return _parse_ideas(raw, cfg=cfg)


def run_qa(question: str, insights_md: str, ideas: list[dict], dig_corpus: list[dict],
           prior_messages: list[dict], data_dir: str, topic_id: str, cfg: dict) -> dict:
    ans = deep_dig_agent.answer_question(
        question=question, insights_md=insights_md, ideas=ideas, dig_corpus=dig_corpus,
        prior_messages=prior_messages, data_dir=data_dir, topic_id=topic_id, cfg=cfg)
    if not ans:
        history = "\n".join(f"{m.get('role')}: {m.get('content')}" for m in (prior_messages or []))
        prompt = _QA_PROMPT.format(insights=insights_md[:6000], papers=_corpus_brief(dig_corpus),
                                   history=history[:4000], question=question)
        ans = call_cli(prompt, cfg, model="opus", timeout=600) or \
            "An error occurred while generating the response."
    return {"content": ans}
```

- [ ] **Step 4: Run the full deep_dig test file**

Run: `uv run pytest tests/test_deep_dig.py -v`
Expected: PASS (all classes)

- [ ] **Step 5: Commit**

```bash
git add src/paper_tracker/deep_dig.py tests/test_deep_dig.py
git commit -m "feat: deep_dig phase runners + deterministic fallbacks"
```

---

## Task 10: Server endpoints

**Files:**
- Modify: `src/paper_tracker/server.py`
- Test: `tests/test_deep_dig_server.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_deep_dig_server.py`:
```python
"""Unit tests for server.py — Deep Dig endpoints (registry/pipeline mocked)."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from paper_tracker.server import app, _deep_dig_progress, _deep_dig_qa_progress

_FAKE_TOPIC = {"id": "test-topic", "name": "Test Topic", "arxiv_keywords": [],
               "arxiv_categories": [], "github_keywords": [], "enabled": True}
_SEED = {"arxiv_id": "2501.00001", "title": "Seed", "abstract": "a", "url": "u", "source": "arxiv"}
_SESSION = {"id": "dd-abc12345", "topic_id": "test-topic", "seed_paper_id": "2501.00001",
            "seed_title": "Seed", "status": "exploring", "insights_md": "", "dig_corpus_json": [],
            "graph_json": {}, "ideas_json": [], "steering_notes": "", "language": "en"}
_SESSION_READY = {**_SESSION, "status": "insights_ready", "insights_md": "memo",
                  "dig_corpus_json": [{"index": "P1", "id": "x"}]}


@pytest.fixture(autouse=True)
def _globals():
    import paper_tracker.server as srv
    mock_reg = MagicMock()
    mock_reg.get_topic.return_value = _FAKE_TOPIC
    srv._registry = mock_reg
    srv._data_dir = "/tmp/test-pt"
    srv._base_cfg = {"paths": {"data_dir": "/tmp/test-pt"}, "summarizer": {}}
    srv._scheduler = MagicMock()
    _deep_dig_progress.clear()
    _deep_dig_qa_progress.clear()
    yield
    srv._registry = None
    srv._scheduler = None


@pytest.fixture()
def client():
    return TestClient(app, raise_server_exceptions=False)


class TestStart:
    @patch("paper_tracker.server._brainstorm_executor")
    @patch("paper_tracker.server.deep_dig.resolve_seed", return_value=_SEED)
    def test_start_202(self, _resolve, mock_exec, client):
        import paper_tracker.server as srv
        srv._registry.create_deep_dig_session.return_value = _SESSION
        resp = client.post("/api/topics/test-topic/deep-dig", json={"paper_id": "2501.00001"})
        assert resp.status_code == 202
        assert resp.json()["session_id"] == "dd-abc12345"
        mock_exec.submit.assert_called_once()

    @patch("paper_tracker.server.deep_dig.resolve_seed", return_value=None)
    def test_start_400_on_bad_seed(self, _resolve, client):
        resp = client.post("/api/topics/test-topic/deep-dig", json={"seed_query": "junk"})
        assert resp.status_code == 400


class TestGetAndList:
    def test_get_detail_includes_messages(self, client):
        import paper_tracker.server as srv
        srv._registry.get_deep_dig_session.return_value = _SESSION_READY
        srv._registry.list_deep_dig_messages.return_value = [{"id": "ddm-1", "role": "user"}]
        resp = client.get("/api/topics/test-topic/deep-dig/dd-abc12345")
        assert resp.status_code == 200
        assert resp.json()["messages"][0]["id"] == "ddm-1"

    def test_list(self, client):
        import paper_tracker.server as srv
        srv._registry.list_deep_dig_sessions.return_value = [_SESSION]
        resp = client.get("/api/topics/test-topic/deep-dig")
        assert resp.status_code == 200
        assert len(resp.json()["sessions"]) == 1


class TestGenerateIdeas:
    @patch("paper_tracker.server._brainstorm_executor")
    def test_requires_insights_ready(self, _exec, client):
        import paper_tracker.server as srv
        srv._registry.get_deep_dig_session.return_value = _SESSION  # exploring
        resp = client.post("/api/topics/test-topic/deep-dig/dd-abc12345/generate-ideas", json={})
        assert resp.status_code == 409

    @patch("paper_tracker.server._brainstorm_executor")
    def test_accepts_when_ready(self, mock_exec, client):
        import paper_tracker.server as srv
        srv._registry.get_deep_dig_session.return_value = _SESSION_READY
        resp = client.post("/api/topics/test-topic/deep-dig/dd-abc12345/generate-ideas",
                           json={"steering_notes": "theory", "focus_paper_ids": ["x"]})
        assert resp.status_code == 202
        mock_exec.submit.assert_called_once()


class TestMessages:
    @patch("paper_tracker.server._brainstorm_executor")
    def test_send_message_202(self, mock_exec, client):
        import paper_tracker.server as srv
        srv._registry.get_deep_dig_session.return_value = _SESSION_READY
        srv._registry.add_deep_dig_message.side_effect = [
            {"id": "ddm-user"}, {"id": "ddm-asst"}]
        resp = client.post("/api/topics/test-topic/deep-dig/dd-abc12345/messages",
                           json={"content": "why?"})
        assert resp.status_code == 202
        assert resp.json()["assistant_msg_id"] == "ddm-asst"
        mock_exec.submit.assert_called_once()

    def test_send_message_rejected_before_ready(self, client):
        import paper_tracker.server as srv
        srv._registry.get_deep_dig_session.return_value = _SESSION  # exploring
        resp = client.post("/api/topics/test-topic/deep-dig/dd-abc12345/messages",
                           json={"content": "why?"})
        assert resp.status_code == 409


class TestDelete:
    def test_delete_204(self, client):
        import paper_tracker.server as srv
        srv._registry.delete_deep_dig_session.return_value = True
        resp = client.delete("/api/topics/test-topic/deep-dig/dd-abc12345")
        assert resp.status_code == 204
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_deep_dig_server.py -v`
Expected: FAIL — `ImportError: cannot import name '_deep_dig_progress'`

- [ ] **Step 3: Add the import, models, progress dicts**

In `src/paper_tracker/server.py`, add to the paper_tracker imports near the top:
```python
from paper_tracker import deep_dig
```

Near the deep-read progress dicts (around line 2365), add:
```python
# In-memory deep-dig progress: session_id → {status, stage, log: [...]}
_deep_dig_progress: dict[str, dict] = {}
# In-memory deep-dig QA progress: assistant_msg_id → {status}
_deep_dig_qa_progress: dict[str, dict] = {}
```

Near the deep-read Pydantic models (around line 2347), add:
```python
class DeepDigCreate(BaseModel):
    paper_id: str | None = None
    seed_query: str | None = None
    language: str = "en"


class DeepDigIdeasCreate(BaseModel):
    steering_notes: str = ""
    focus_paper_ids: list[str] = []


class DeepDigMessageCreate(BaseModel):
    content: str

    @field_validator("content")
    @classmethod
    def _not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content must not be empty")
        return v
```

- [ ] **Step 4: Add the endpoints**

In `src/paper_tracker/server.py`, after the deep-read endpoints (after `get_deep_read_message_progress`, ~line 2619), add:
```python
# ---------------------------------------------------------------------------
# Deep Dig
# ---------------------------------------------------------------------------

@app.post("/api/topics/{topic_id}/deep-dig", status_code=202)
async def start_deep_dig(topic_id: str, body: DeepDigCreate) -> dict:
    reg = _get_registry()
    topic = reg.get_topic(topic_id)
    if not topic:
        raise HTTPException(404, detail="Topic not found")
    topic_cfg = cfg_module.from_topic(topic, _base_cfg)

    seed = deep_dig.resolve_seed(_data_dir, topic_id, topic_cfg,
                                 paper_id=body.paper_id, seed_query=body.seed_query)
    if not seed:
        raise HTTPException(400, detail="Could not resolve a seed paper from the input")

    seed_id = seed.get("arxiv_id") or seed.get("paper_id") or ""
    session = reg.create_deep_dig_session(topic_id, seed_id, seed.get("title", ""), body.language)
    session_id = session["id"]
    _deep_dig_progress[session_id] = {"status": "exploring", "stage": "exploring", "log": []}

    def _run():
        try:
            def _on_event(msg: str):
                p = _deep_dig_progress.get(session_id) or {"status": "exploring", "log": []}
                p["log"] = (p.get("log", []) + [msg])[-12:]
                _deep_dig_progress[session_id] = p

            res = deep_dig.run_explore(seed, topic["name"], _data_dir, topic_id, topic_cfg,
                                       on_event=_on_event)
            from datetime import datetime, timezone
            reg.update_deep_dig_session(topic_id, session_id, {
                "status": "insights_ready",
                "insights_md": res.get("insights_md", ""),
                "dig_corpus_json": res.get("dig_corpus", []),
                "graph_json": res.get("graph", {}),
            })
            _deep_dig_progress[session_id] = {"status": "insights_ready", "stage": "insights_ready",
                                              "log": _deep_dig_progress.get(session_id, {}).get("log", [])}
        except Exception as e:
            log.exception("Deep dig explore failed: %s", e)
            from datetime import datetime, timezone
            reg.update_deep_dig_session(topic_id, session_id, {
                "status": "failed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error_message": str(e)[:2000],
            })
            _deep_dig_progress[session_id] = {"status": "failed", "stage": "failed", "log": []}

    _brainstorm_executor.submit(_run)
    return {"status": "started", "session_id": session_id}


@app.get("/api/topics/{topic_id}/deep-dig")
async def list_deep_dig_sessions(topic_id: str) -> dict:
    reg = _get_registry()
    if not reg.get_topic(topic_id):
        raise HTTPException(404, detail="Topic not found")
    return {"sessions": reg.list_deep_dig_sessions(topic_id)}


@app.get("/api/topics/{topic_id}/deep-dig/{session_id}")
async def get_deep_dig_session(topic_id: str, session_id: str) -> dict:
    reg = _get_registry()
    session = reg.get_deep_dig_session(topic_id, session_id)
    if not session:
        raise HTTPException(404, detail="Deep dig session not found")
    messages = reg.list_deep_dig_messages(topic_id, session_id)
    return {**session, "messages": messages}


@app.get("/api/topics/{topic_id}/deep-dig/{session_id}/progress")
async def get_deep_dig_progress(topic_id: str, session_id: str) -> dict:
    progress = _deep_dig_progress.get(session_id)
    if progress:
        if progress["status"] in ("completed", "failed", "insights_ready"):
            # keep insights_ready visible once; drop terminal states
            if progress["status"] in ("completed", "failed"):
                _deep_dig_progress.pop(session_id, None)
        return {"session_id": session_id, **progress}
    reg = _get_registry()
    session = reg.get_deep_dig_session(topic_id, session_id)
    if not session:
        raise HTTPException(404, detail="Deep dig session not found")
    return {"session_id": session_id, "status": session["status"], "stage": session["status"], "log": []}


@app.post("/api/topics/{topic_id}/deep-dig/{session_id}/generate-ideas", status_code=202)
async def generate_deep_dig_ideas(topic_id: str, session_id: str, body: DeepDigIdeasCreate) -> dict:
    reg = _get_registry()
    topic = reg.get_topic(topic_id)
    if not topic:
        raise HTTPException(404, detail="Topic not found")
    session = reg.get_deep_dig_session(topic_id, session_id)
    if not session:
        raise HTTPException(404, detail="Deep dig session not found")
    if session["status"] not in ("insights_ready", "completed"):
        raise HTTPException(409, detail="Ideas can only be generated after insights are ready")
    topic_cfg = cfg_module.from_topic(topic, _base_cfg)

    reg.update_deep_dig_session(topic_id, session_id, {
        "status": "generating_ideas", "steering_notes": body.steering_notes, "error_message": "",
    })
    _deep_dig_progress[session_id] = {"status": "generating_ideas", "stage": "generating_ideas", "log": []}

    def _run():
        try:
            ideas = deep_dig.run_ideas(
                session.get("insights_md", ""), session.get("dig_corpus_json", []),
                body.steering_notes, body.focus_paper_ids, topic["name"],
                _data_dir, topic_id, topic_cfg)
            from datetime import datetime, timezone
            reg.update_deep_dig_session(topic_id, session_id, {
                "status": "completed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "ideas_json": ideas,
            })
            _deep_dig_progress[session_id] = {"status": "completed", "stage": "completed", "log": []}
        except Exception as e:
            log.exception("Deep dig ideas failed: %s", e)
            reg.update_deep_dig_session(topic_id, session_id, {
                "status": "insights_ready", "error_message": str(e)[:2000],
            })
            _deep_dig_progress[session_id] = {"status": "insights_ready", "stage": "insights_ready", "log": []}

    _brainstorm_executor.submit(_run)
    return {"status": "started", "session_id": session_id}


@app.post("/api/topics/{topic_id}/deep-dig/{session_id}/messages", status_code=202)
async def send_deep_dig_message(topic_id: str, session_id: str, body: DeepDigMessageCreate) -> dict:
    reg = _get_registry()
    topic = reg.get_topic(topic_id)
    if not topic:
        raise HTTPException(404, detail="Topic not found")
    session = reg.get_deep_dig_session(topic_id, session_id)
    if not session:
        raise HTTPException(404, detail="Deep dig session not found")
    if session["status"] not in ("insights_ready", "generating_ideas", "completed"):
        raise HTTPException(409, detail="Can only ask questions once insights are ready")
    topic_cfg = cfg_module.from_topic(topic, _base_cfg)

    user_msg = reg.add_deep_dig_message(topic_id, session_id, "user", body.content)
    assistant_msg = reg.add_deep_dig_message(topic_id, session_id, "assistant", "", status="pending")
    assistant_msg_id = assistant_msg["id"]
    _deep_dig_qa_progress[assistant_msg_id] = {"status": "pending"}

    def _run():
        try:
            reg.update_deep_dig_message(assistant_msg_id, {"status": "generating"})
            _deep_dig_qa_progress[assistant_msg_id] = {"status": "generating"}
            prior = [m for m in reg.list_deep_dig_messages(topic_id, session_id)
                     if m["id"] != assistant_msg_id]
            result = deep_dig.run_qa(
                body.content, session.get("insights_md", ""), session.get("ideas_json", []),
                session.get("dig_corpus_json", []), prior, _data_dir, topic_id, topic_cfg)
            reg.update_deep_dig_message(assistant_msg_id, {"content": result["content"], "status": "completed"})
            _deep_dig_qa_progress[assistant_msg_id] = {"status": "completed"}
        except Exception as e:
            log.exception("Deep dig QA failed: %s", e)
            reg.update_deep_dig_message(assistant_msg_id, {
                "content": "An error occurred while generating the response.", "status": "failed"})
            _deep_dig_qa_progress[assistant_msg_id] = {"status": "failed"}

    _brainstorm_executor.submit(_run)
    return {"user_msg_id": user_msg["id"], "assistant_msg_id": assistant_msg_id}


@app.get("/api/topics/{topic_id}/deep-dig/{session_id}/messages/{msg_id}/progress")
async def get_deep_dig_message_progress(topic_id: str, session_id: str, msg_id: str) -> dict:
    progress = _deep_dig_qa_progress.get(msg_id)
    if progress:
        if progress["status"] in ("completed", "failed"):
            _deep_dig_qa_progress.pop(msg_id, None)
        return {"msg_id": msg_id, **progress}
    reg = _get_registry()
    msg = reg.get_deep_dig_message(msg_id)
    if not msg:
        raise HTTPException(404, detail="Message not found")
    return {"msg_id": msg_id, "status": msg["status"]}


@app.delete("/api/topics/{topic_id}/deep-dig/{session_id}", status_code=204)
async def delete_deep_dig_session(topic_id: str, session_id: str) -> None:
    reg = _get_registry()
    if not reg.delete_deep_dig_session(topic_id, session_id):
        raise HTTPException(404, detail="Deep dig session not found")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_deep_dig_server.py -v`
Expected: PASS (all classes)

- [ ] **Step 6: Run the whole backend suite (regression)**

Run: `uv run pytest -q`
Expected: PASS (no regressions in existing tests)

- [ ] **Step 7: Commit**

```bash
git add src/paper_tracker/server.py tests/test_deep_dig_server.py
git commit -m "feat: deep-dig server endpoints (start/gate/ideas/messages/progress)"
```

---

## Task 11: Frontend — api.ts functions + types

**Files:**
- Modify: `frontend/src/lib/api.ts`

No frontend test harness exists; verify with the TypeScript compiler.

- [ ] **Step 1: Add the TypeScript interfaces**

In `frontend/src/lib/api.ts`, near the deep-read interfaces, add:
```typescript
export interface DeepDigCorpusPaper {
  index: string;
  id: string;
  source: string;
  title: string;
  authors: string;
  venue: string;
  url: string;
  abstract: string;
  summary: string;
  key_insight: string;
  math_concepts: string[];
  found_via: string;
  influential_citation_count: number;
}

export interface DeepDigGraphNode {
  id: string;
  index: string;
  title: string;
  found_via: string;
  citation_count: number;
  influential_citation_count: number;
}

export interface DeepDigGraphEdge {
  from: string;
  to: string;
  kind: string;
  influential: boolean;
  intents: string[];
}

export interface DeepDigGraph {
  nodes: DeepDigGraphNode[];
  edges: DeepDigGraphEdge[];
}

export interface DeepDigIdea {
  title: string;
  problem: string;
  motivation: string;
  method: string;
  experiment_plan: string;
  novelty_score: number;
  feasibility_score: number;
}

export interface DeepDigMessage {
  id: string;
  session_id: string;
  topic_id: string;
  role: "user" | "assistant";
  content: string;
  status: "pending" | "generating" | "completed" | "failed";
  created_at: string;
}

export interface DeepDigSession {
  id: string;
  topic_id: string;
  seed_paper_id: string;
  seed_title: string;
  status: "exploring" | "insights_ready" | "generating_ideas" | "completed" | "failed";
  language: string;
  started_at: string;
  finished_at: string | null;
  insights_md: string;
  dig_corpus_json: DeepDigCorpusPaper[];
  graph_json: DeepDigGraph;
  steering_notes: string;
  ideas_json: DeepDigIdea[];
  error_message: string;
}

export interface DeepDigSessionDetail extends DeepDigSession {
  messages: DeepDigMessage[];
}
```

- [ ] **Step 2: Add the API functions**

In the `api` object in `frontend/src/lib/api.ts`, add (after the deep-read functions):
```typescript
  startDeepDig: (
    topicId: string,
    body: { paper_id?: string; seed_query?: string; language?: string },
  ) =>
    req<{ status: string; session_id: string }>(`/api/topics/${topicId}/deep-dig`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  listDeepDigSessions: (topicId: string) =>
    req<{ sessions: DeepDigSession[] }>(`/api/topics/${topicId}/deep-dig`),

  getDeepDigSession: (topicId: string, sessionId: string) =>
    req<DeepDigSessionDetail>(`/api/topics/${topicId}/deep-dig/${sessionId}`),

  getDeepDigProgress: (topicId: string, sessionId: string) =>
    req<{ session_id: string; status: string; stage: string; log: string[] }>(
      `/api/topics/${topicId}/deep-dig/${sessionId}/progress`,
    ),

  generateDeepDigIdeas: (
    topicId: string,
    sessionId: string,
    body: { steering_notes?: string; focus_paper_ids?: string[] },
  ) =>
    req<{ status: string; session_id: string }>(
      `/api/topics/${topicId}/deep-dig/${sessionId}/generate-ideas`,
      { method: "POST", body: JSON.stringify(body) },
    ),

  sendDeepDigMessage: (topicId: string, sessionId: string, content: string) =>
    req<{ user_msg_id: string; assistant_msg_id: string }>(
      `/api/topics/${topicId}/deep-dig/${sessionId}/messages`,
      { method: "POST", body: JSON.stringify({ content }) },
    ),

  getDeepDigMessageProgress: (topicId: string, sessionId: string, msgId: string) =>
    req<{ msg_id: string; status: string }>(
      `/api/topics/${topicId}/deep-dig/${sessionId}/messages/${msgId}/progress`,
    ),

  deleteDeepDigSession: (topicId: string, sessionId: string) =>
    req<void>(`/api/topics/${topicId}/deep-dig/${sessionId}`, { method: "DELETE" }),
```

- [ ] **Step 3: Type-check**

Run: `cd frontend && npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/lib/api.ts
git commit -m "feat: deep-dig frontend API client + types"
```

---

## Task 12: Frontend — DeepDigGraph component (lightweight SVG)

**Files:**
- Create: `frontend/src/components/DeepDigGraph.tsx`

- [ ] **Step 1: Create the component**

Create `frontend/src/components/DeepDigGraph.tsx`:
```tsx
"use client";

import { DeepDigGraph as Graph, DeepDigGraphNode } from "@/lib/api";

const GROUPS: { key: string; label: string; color: string }[] = [
  { key: "seed", label: "Seed", color: "#2563eb" },
  { key: "references", label: "References", color: "#059669" },
  { key: "citations", label: "Citations", color: "#d97706" },
  { key: "search_related", label: "Related", color: "#7c3aed" },
  { key: "search_math", label: "Math", color: "#db2777" },
];

const W = 720;
const COL_GAP = W / (GROUPS.length + 1);
const ROW_GAP = 34;
const TOP = 40;

export function DeepDigGraph({ graph }: { graph: Graph }) {
  const nodes = graph?.nodes ?? [];
  const edges = graph?.edges ?? [];
  if (!nodes.length) {
    return <p className="text-xs text-gray-400">No graph available.</p>;
  }

  // place nodes in columns by found_via group
  const pos = new Map<string, { x: number; y: number; node: DeepDigGraphNode }>();
  GROUPS.forEach((g, gi) => {
    const groupNodes = nodes.filter((n) =>
      g.key === "seed" ? n.found_via === "seed" : n.found_via === g.key,
    );
    groupNodes.forEach((n, ri) => {
      pos.set(n.id, { x: COL_GAP * (gi + 1), y: TOP + ri * ROW_GAP, node: n });
    });
  });
  // any node whose group wasn't matched (fallback) → last column
  nodes.forEach((n, i) => {
    if (!pos.has(n.id)) pos.set(n.id, { x: COL_GAP * GROUPS.length, y: TOP + i * ROW_GAP, node: n });
  });

  const maxY = Math.max(TOP, ...Array.from(pos.values()).map((p) => p.y)) + ROW_GAP;
  const colorOf = (via: string) =>
    GROUPS.find((g) => g.key === via)?.color ?? "#6b7280";

  return (
    <div className="overflow-x-auto">
      {/* legend */}
      <div className="flex flex-wrap gap-3 mb-2">
        {GROUPS.map((g) => (
          <span key={g.key} className="flex items-center gap-1 text-[11px] text-gray-600">
            <span className="inline-block w-2.5 h-2.5 rounded-full" style={{ background: g.color }} />
            {g.label}
          </span>
        ))}
      </div>
      <svg width={W} height={maxY} className="min-w-[720px]">
        {edges.map((e, i) => {
          const a = pos.get(e.from);
          const b = pos.get(e.to);
          if (!a || !b) return null;
          return (
            <line
              key={i}
              x1={a.x}
              y1={a.y}
              x2={b.x}
              y2={b.y}
              stroke={e.influential ? "#f59e0b" : "#d1d5db"}
              strokeWidth={e.influential ? 1.8 : 0.8}
            />
          );
        })}
        {Array.from(pos.values()).map(({ x, y, node }) => {
          const r = Math.min(9, 4 + Math.log10((node.citation_count || 0) + 1));
          return (
            <g key={node.id}>
              <title>
                {node.index}: {node.title} ({node.citation_count} citations)
              </title>
              <circle cx={x} cy={y} r={r} fill={colorOf(node.found_via)} />
              <text x={x + r + 3} y={y + 3} fontSize="9" fill="#374151">
                {node.index}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
```

- [ ] **Step 2: Type-check**

Run: `cd frontend && npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/DeepDigGraph.tsx
git commit -m "feat: lightweight SVG DeepDigGraph component"
```

---

## Task 13: Frontend — Deep Dig page

**Files:**
- Create: `frontend/src/app/topics/[id]/deep-dig/page.tsx`

This page mirrors the deep-read page (seed picker, session list, progress, content, Q&A) and adds the gate (steering + focus + Generate ideas) and the graph.

- [ ] **Step 1: Create the page**

Create `frontend/src/app/topics/[id]/deep-dig/page.tsx`:
```tsx
"use client";

import { useState, useEffect, useRef } from "react";
import { useParams } from "next/navigation";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Search, Send, Trash2, Loader2, Sparkles, Compass } from "lucide-react";
import { api, DeepDigMessage, DeepDigCorpusPaper, DeepDigIdea } from "@/lib/api";
import { useDebounce } from "@/lib/hooks";
import { MathMarkdown } from "@/components/MathMarkdown";
import { DeepDigGraph } from "@/components/DeepDigGraph";

export default function DeepDigPage() {
  const { id: topicId } = useParams<{ id: string }>();
  const queryClient = useQueryClient();

  // seed selection
  const [paperSearch, setPaperSearch] = useState("");
  const [showDropdown, setShowDropdown] = useState(false);
  const [selectedPaperId, setSelectedPaperId] = useState<string | null>(null);
  const [selectedPaperTitle, setSelectedPaperTitle] = useState("");
  const [seedQuery, setSeedQuery] = useState("");
  const dropdownRef = useRef<HTMLDivElement>(null);
  const debouncedSearch = useDebounce(paperSearch, 300);

  // session + gate state
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [steering, setSteering] = useState("");
  const [focusIds, setFocusIds] = useState<string[]>([]);
  const [qaInput, setQaInput] = useState("");
  const [pendingMsgId, setPendingMsgId] = useState<string | null>(null);

  // paper search dropdown
  const { data: paperResults } = useQuery({
    queryKey: ["papers-search", topicId, debouncedSearch],
    queryFn: () => api.getPapers(topicId, { search: debouncedSearch, limit: 20 }),
    enabled: showDropdown && debouncedSearch.length > 0,
  });

  useEffect(() => {
    function onClick(e: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setShowDropdown(false);
      }
    }
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, []);

  // sessions list
  const { data: sessionsData } = useQuery({
    queryKey: ["deep-dig-sessions", topicId],
    queryFn: () => api.listDeepDigSessions(topicId),
    refetchInterval: 5000,
  });
  const sessions = sessionsData?.sessions ?? [];

  // active session
  const { data: session } = useQuery({
    queryKey: ["deep-dig-session", topicId, activeSessionId],
    queryFn: () => api.getDeepDigSession(topicId, activeSessionId!),
    enabled: !!activeSessionId,
    refetchInterval: (query) => {
      const s = query.state.data?.status;
      return s === "exploring" || s === "generating_ideas" ? 3000 : false;
    },
  });

  // progress (during exploring / generating)
  const { data: progress } = useQuery({
    queryKey: ["deep-dig-progress", topicId, activeSessionId],
    queryFn: () => api.getDeepDigProgress(topicId, activeSessionId!),
    enabled:
      !!activeSessionId &&
      (session?.status === "exploring" || session?.status === "generating_ideas"),
    refetchInterval: 2000,
  });

  // QA pending poll
  useQuery({
    queryKey: ["deep-dig-qa", topicId, activeSessionId, pendingMsgId],
    queryFn: async () => {
      const res = await api.getDeepDigMessageProgress(topicId, activeSessionId!, pendingMsgId!);
      if (res.status === "completed" || res.status === "failed") {
        setPendingMsgId(null);
        queryClient.invalidateQueries({ queryKey: ["deep-dig-session", topicId, activeSessionId] });
      }
      return res;
    },
    enabled: !!pendingMsgId && !!activeSessionId,
    refetchInterval: 2000,
  });

  useEffect(() => {
    if (session) setSteering(session.steering_notes || "");
  }, [session?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const startMut = useMutation({
    mutationFn: () =>
      api.startDeepDig(topicId, {
        paper_id: selectedPaperId || undefined,
        seed_query: !selectedPaperId && seedQuery ? seedQuery : undefined,
      }),
    onSuccess: (res) => {
      setActiveSessionId(res.session_id);
      queryClient.invalidateQueries({ queryKey: ["deep-dig-sessions", topicId] });
    },
  });

  const ideasMut = useMutation({
    mutationFn: () =>
      api.generateDeepDigIdeas(topicId, activeSessionId!, {
        steering_notes: steering,
        focus_paper_ids: focusIds,
      }),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["deep-dig-session", topicId, activeSessionId] }),
  });

  const deleteMut = useMutation({
    mutationFn: (sid: string) => api.deleteDeepDigSession(topicId, sid),
    onSuccess: () => {
      setActiveSessionId(null);
      queryClient.invalidateQueries({ queryKey: ["deep-dig-sessions", topicId] });
    },
  });

  async function handleSendQA() {
    if (!qaInput.trim() || !activeSessionId || pendingMsgId) return;
    const content = qaInput;
    setQaInput("");
    const res = await api.sendDeepDigMessage(topicId, activeSessionId, content);
    setPendingMsgId(res.assistant_msg_id);
    queryClient.invalidateQueries({ queryKey: ["deep-dig-session", topicId, activeSessionId] });
  }

  function toggleFocus(id: string) {
    setFocusIds((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));
  }

  const canStart = !!selectedPaperId || !!seedQuery.trim();
  const corpus: DeepDigCorpusPaper[] = session?.dig_corpus_json ?? [];
  const ideas: DeepDigIdea[] = session?.ideas_json ?? [];

  return (
    <div className="flex gap-4 min-h-[600px]">
      {/* Sidebar */}
      <div className="w-64 flex-shrink-0 space-y-3">
        <div className="relative" ref={dropdownRef}>
          <Search size={14} className="absolute left-2.5 top-2.5 text-gray-400" />
          <input
            value={paperSearch}
            onChange={(e) => {
              setPaperSearch(e.target.value);
              setShowDropdown(true);
            }}
            onFocus={() => paperSearch && setShowDropdown(true)}
            placeholder="Search library papers..."
            className="w-full pl-8 pr-3 py-2 text-sm border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          {showDropdown && paperResults?.papers?.length ? (
            <div className="absolute z-20 mt-1 w-full bg-white border border-gray-200 rounded-lg shadow-lg max-h-64 overflow-y-auto">
              {paperResults.papers.map((p) => (
                <button
                  key={p.arxiv_id}
                  onClick={() => {
                    setSelectedPaperId(p.arxiv_id);
                    setSelectedPaperTitle(p.title);
                    setSeedQuery("");
                    setShowDropdown(false);
                    setPaperSearch("");
                  }}
                  className="w-full text-left px-3 py-2 text-xs hover:bg-blue-50 border-b border-gray-100 last:border-0"
                >
                  <div className="font-medium text-gray-900 line-clamp-2">{p.title}</div>
                  <div className="text-gray-400 mt-0.5">{p.arxiv_id}</div>
                </button>
              ))}
            </div>
          ) : null}
        </div>

        {selectedPaperTitle && (
          <div className="bg-blue-50 rounded-lg px-3 py-2 text-xs text-blue-800">
            <span className="font-medium">Seed:</span>{" "}
            <span className="line-clamp-2">{selectedPaperTitle}</span>
          </div>
        )}

        <div className="text-[11px] text-gray-400 text-center">or</div>
        <input
          value={seedQuery}
          onChange={(e) => {
            setSeedQuery(e.target.value);
            setSelectedPaperId(null);
            setSelectedPaperTitle("");
          }}
          placeholder="Paste arXiv id / URL / PDF URL"
          className="w-full px-3 py-2 text-sm border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
        />

        <button
          onClick={() => startMut.mutate()}
          disabled={!canStart || startMut.isPending}
          className="w-full flex items-center justify-center gap-2 px-3 py-2 text-sm font-medium bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
        >
          {startMut.isPending ? <Loader2 size={14} className="animate-spin" /> : <Compass size={14} />}
          Start Deep Dig
        </button>

        <div className="space-y-1">
          <h3 className="text-xs font-semibold text-gray-500 uppercase px-1">Sessions</h3>
          {sessions.length === 0 ? (
            <p className="text-xs text-gray-400 px-1">No sessions yet</p>
          ) : (
            sessions.map((s) => (
              <div
                key={s.id}
                onClick={() => setActiveSessionId(s.id)}
                className={`group flex items-start justify-between rounded-lg px-3 py-2 cursor-pointer ${
                  activeSessionId === s.id ? "bg-blue-50 border border-blue-200" : "hover:bg-gray-50 border border-transparent"
                }`}
              >
                <div className="min-w-0 flex-1">
                  <div className="text-xs font-medium text-gray-900 line-clamp-1">
                    {s.seed_title || s.seed_paper_id}
                  </div>
                  <div className="flex items-center gap-1.5 mt-0.5">
                    <span className="text-[10px] text-gray-400 font-mono">{s.id}</span>
                    <span className="text-[10px] text-gray-500">{s.status}</span>
                  </div>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteMut.mutate(s.id);
                  }}
                  className="p-0.5 text-gray-300 hover:text-red-500 opacity-0 group-hover:opacity-100"
                >
                  <Trash2 size={12} />
                </button>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Main */}
      <div className="flex-1 min-w-0">
        {!activeSessionId || !session ? (
          <div className="flex items-center justify-center h-96 text-gray-400">
            Select or start a deep dig
          </div>
        ) : (
          <div className="space-y-4">
            {/* progress */}
            {(session.status === "exploring" || session.status === "generating_ideas") && (
              <div className="bg-white rounded-xl border border-gray-200 p-4">
                <div className="flex items-center gap-2 text-sm text-amber-700">
                  <Loader2 size={14} className="animate-spin" />
                  {session.status === "exploring" ? "Exploring the citation graph..." : "Generating ideas..."}
                </div>
                {progress?.log?.length ? (
                  <div className="mt-2 space-y-0.5 max-h-40 overflow-y-auto">
                    {progress.log.map((l, i) => (
                      <div key={i} className="text-[11px] text-gray-500 font-mono">{l}</div>
                    ))}
                  </div>
                ) : null}
              </div>
            )}

            {session.status === "failed" && (
              <div className="bg-red-50 rounded-xl border border-red-200 p-4 text-sm text-red-700">
                Deep dig failed. {session.error_message}
              </div>
            )}

            {/* insights */}
            {session.insights_md && (
              <div className="bg-white rounded-xl border border-gray-200 p-5">
                <h3 className="text-sm font-semibold text-gray-900 mb-2">Insights</h3>
                <MathMarkdown className="prose prose-sm max-w-none">{session.insights_md}</MathMarkdown>
              </div>
            )}

            {/* graph */}
            {session.graph_json?.nodes?.length ? (
              <div className="bg-white rounded-xl border border-gray-200 p-5">
                <h3 className="text-sm font-semibold text-gray-900 mb-2">Citation / relation graph</h3>
                <DeepDigGraph graph={session.graph_json} />
              </div>
            ) : null}

            {/* gate */}
            {(session.status === "insights_ready" || session.status === "completed") && (
              <div className="bg-white rounded-xl border border-gray-200 p-5 space-y-3">
                <h3 className="text-sm font-semibold text-gray-900">
                  {session.status === "completed" ? "Regenerate ideas" : "Generate ideas"}
                </h3>
                <textarea
                  value={steering}
                  onChange={(e) => setSteering(e.target.value)}
                  placeholder="Optional: steer the ideas (directions to emphasize, constraints)…"
                  className="w-full h-20 text-sm border border-gray-200 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 resize-y"
                />
                {corpus.length > 0 && (
                  <div>
                    <div className="text-xs text-gray-500 mb-1">Focus papers (optional)</div>
                    <div className="flex flex-wrap gap-1.5 max-h-32 overflow-y-auto">
                      {corpus.map((p) => (
                        <button
                          key={p.id}
                          onClick={() => toggleFocus(p.id)}
                          title={p.title}
                          className={`px-2 py-0.5 text-[11px] rounded-full border ${
                            focusIds.includes(p.id)
                              ? "bg-blue-600 text-white border-blue-600"
                              : "bg-white text-gray-600 border-gray-200 hover:border-blue-300"
                          }`}
                        >
                          {p.index}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
                <button
                  onClick={() => ideasMut.mutate()}
                  disabled={ideasMut.isPending}
                  className="flex items-center gap-2 px-3 py-2 text-sm font-medium bg-violet-600 text-white rounded-lg hover:bg-violet-700 disabled:opacity-50"
                >
                  {ideasMut.isPending ? <Loader2 size={14} className="animate-spin" /> : <Sparkles size={14} />}
                  {session.status === "completed" ? "Regenerate ideas" : "Generate ideas"}
                </button>
              </div>
            )}

            {/* ideas */}
            {ideas.length > 0 && (
              <div className="space-y-3">
                <h3 className="text-sm font-semibold text-gray-900">New ideas</h3>
                {ideas.map((idea, i) => (
                  <div key={i} className="bg-white rounded-xl border border-gray-200 p-5 space-y-2">
                    <div className="flex items-start justify-between gap-2">
                      <h4 className="text-sm font-semibold text-gray-900">{idea.title}</h4>
                      <div className="flex gap-1 flex-shrink-0">
                        <span className="text-[10px] px-1.5 py-0.5 bg-emerald-100 text-emerald-700 rounded">N {idea.novelty_score}</span>
                        <span className="text-[10px] px-1.5 py-0.5 bg-blue-100 text-blue-700 rounded">F {idea.feasibility_score}</span>
                      </div>
                    </div>
                    <p className="text-xs text-gray-600"><span className="font-medium">Problem:</span> {idea.problem}</p>
                    <p className="text-xs text-gray-600"><span className="font-medium">Motivation:</span> {idea.motivation}</p>
                    <p className="text-xs text-gray-600"><span className="font-medium">Method:</span> {idea.method}</p>
                    <p className="text-xs text-gray-600"><span className="font-medium">Experiment:</span> {idea.experiment_plan}</p>
                  </div>
                ))}
              </div>
            )}

            {/* Q&A */}
            {(session.status === "insights_ready" ||
              session.status === "generating_ideas" ||
              session.status === "completed") && (
              <div className="bg-white rounded-xl border border-gray-200 p-5 space-y-3">
                <h3 className="text-sm font-semibold text-gray-900">Ask questions</h3>
                {session.messages?.length > 0 && (
                  <div className="space-y-3 max-h-96 overflow-y-auto">
                    {session.messages.map((msg: DeepDigMessage) => (
                      <div
                        key={msg.id}
                        className={`rounded-lg px-4 py-3 text-sm ${
                          msg.role === "user" ? "bg-blue-50 text-blue-900 ml-8" : "bg-gray-50 text-gray-800 mr-8"
                        }`}
                      >
                        {msg.status === "pending" || msg.status === "generating" ? (
                          <span className="flex items-center gap-2 text-gray-400">
                            <Loader2 size={14} className="animate-spin" />
                            {msg.status === "generating" ? "Digging..." : "Queued..."}
                          </span>
                        ) : msg.content ? (
                          <MathMarkdown className="prose prose-sm max-w-none">{msg.content}</MathMarkdown>
                        ) : (
                          <span className="text-gray-400 italic">
                            {msg.status === "failed" ? "Failed to generate response" : ""}
                          </span>
                        )}
                      </div>
                    ))}
                  </div>
                )}
                <div className="flex gap-2">
                  <input
                    value={qaInput}
                    onChange={(e) => setQaInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault();
                        handleSendQA();
                      }
                    }}
                    placeholder="Ask about the dig, papers, or ideas..."
                    disabled={!!pendingMsgId}
                    className="flex-1 px-3 py-2 text-sm border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
                  />
                  <button
                    onClick={handleSendQA}
                    disabled={!qaInput.trim() || !!pendingMsgId}
                    className="px-3 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
                  >
                    {pendingMsgId ? <Loader2 size={14} className="animate-spin" /> : <Send size={14} />}
                  </button>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Type-check**

Run: `cd frontend && npx tsc --noEmit`
Expected: no errors. (If `api.getPapers` signature differs, match the call used in `deep-read/page.tsx`.)

- [ ] **Step 3: Commit**

```bash
git add frontend/src/app/topics/[id]/deep-dig/page.tsx
git commit -m "feat: deep-dig page (seed picker, insights, graph, gate, ideas, Q&A)"
```

---

## Task 14: Frontend — nav tab

**Files:**
- Modify: `frontend/src/app/topics/[id]/layout.tsx`

- [ ] **Step 1: Add the tab**

In `frontend/src/app/topics/[id]/layout.tsx`, update the `TABS` array to add Deep Dig after Deep Read:
```typescript
const TABS = [
  { label: "Overview", href: "" },
  { label: "Papers", href: "/papers" },
  { label: "Deep Read", href: "/deep-read" },
  { label: "Deep Dig", href: "/deep-dig" },
  { label: "Insights", href: "/insights" },
  { label: "Brainstorm", href: "/brainstorm" },
  { label: "Research Plan", href: "/research-plan" },
  { label: "Chat", href: "/chat" },
] as const;
```

- [ ] **Step 2: Build the frontend**

Run: `cd frontend && npm run build`
Expected: build succeeds (compiles the new route).

- [ ] **Step 3: Commit**

```bash
git add frontend/src/app/topics/[id]/layout.tsx
git commit -m "feat: add Deep Dig nav tab"
```

---

## Task 15: Manual end-to-end verification

**Files:** none (verification + notes)

- [ ] **Step 1: Start the backend**

Run: `uv run uvicorn paper_tracker.server:app --reload --port 8000`
Expected: starts; logs show `recover_stale_tasks` ran (no crash on the new tables).

- [ ] **Step 2: Start the frontend**

Run (new terminal): `cd frontend && npm run dev`
Open: `http://localhost:3000`, pick a topic with papers, open the **Deep Dig** tab.

- [ ] **Step 3: Run a dig**

Pick a library paper (or paste an arXiv id), click **Start Deep Dig**. Verify:
- progress log streams tool events;
- status moves to `insights_ready`; insights memo + citation graph render;
- the gate appears (steering + focus chips + Generate ideas).

- [ ] **Step 4: Generate ideas + Q&A**

Click **Generate ideas** → ideas render with novelty/feasibility. Ask a question → assistant message goes pending → completed with an answer.

- [ ] **Step 5: Verify S2 fallback resilience**

Temporarily set an invalid `S2_API_KEY` (or disconnect network) and start a dig: it should still complete via semantic + math paths (degraded), not error.

- [ ] **Step 6: Final regression + commit any fixes**

Run: `uv run pytest -q`
Expected: all pass. Commit any fixes found during manual testing.

---

## Notes for the implementer

- **Reused helpers:** `deep_dig_agent` imports `_format_card` and `_fetch_fulltext` from `brainstorm_agent` — do not duplicate them. It defines its OWN `_run_agent` + `_TOOL_NAMES` (the brainstorm one hardcodes brainstorm's tool names in `allowed_tools`, so it cannot be reused directly).
- **`_parse_ideas` import:** comes from `brainstorm` (`from paper_tracker.brainstorm import _parse_ideas`); it accepts `(raw, single=False, cfg=None)`.
- **Idea dict keys** must stay exactly: `title`, `problem`, `motivation`, `method`, `experiment_plan`, `novelty_score`, `feasibility_score` (the `DeepDigIdea` TS interface and the page rendering depend on these).
- **Status strings** must stay exactly: `exploring`, `insights_ready`, `generating_ideas`, `completed`, `failed` (server guards, recover_stale, and the page switch on these).
- **Do not write dug papers into the topic library** — they live only in `dig_corpus_json`.
