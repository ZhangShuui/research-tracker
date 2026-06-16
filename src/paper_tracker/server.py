"""FastAPI REST API for the paper-tracker web service."""

from __future__ import annotations

import hashlib
import logging
import re
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator

from paper_tracker import config as cfg_module
from paper_tracker import crossdomain
from paper_tracker import insights
from paper_tracker import rag
from paper_tracker.brainstorm import run_brainstorm, check_prior_art
from paper_tracker.chat import generate_chat_response
from paper_tracker.discovery import run_trending, run_math_insights, run_community_ideas, review_discovery_report, gather_cross_domain_papers
from paper_tracker.research_plan import generate_research_plan, refine_research_plan
from paper_tracker.registry import Registry
from paper_tracker.scheduler import Scheduler
from paper_tracker.sources import arxiv, pdf as pdf_source
from paper_tracker.storage import Storage
from paper_tracker.summarizer import refilter_papers, summarize_papers

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Startup / shared state
# ---------------------------------------------------------------------------

_registry: Registry | None = None
_scheduler: Scheduler | None = None
_data_dir: str = ""
_base_cfg: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _registry, _scheduler, _data_dir, _base_cfg

    try:
        base_cfg = cfg_module.load()
    except FileNotFoundError:
        base_cfg = cfg_module._default_base_cfg()

    _data_dir = base_cfg["paths"]["data_dir"]
    _base_cfg = base_cfg
    _registry = Registry(_data_dir)
    _scheduler = Scheduler(_registry, _data_dir, base_cfg)
    _scheduler.start()

    _maybe_import_legacy(base_cfg)

    # Recover tasks that were stuck in 'running' from a previous crash/restart
    recovered = _registry.recover_stale_tasks()
    if recovered:
        log.warning("Recovered stale tasks on startup: %s", recovered)

    log.info("Paper Tracker API started. data_dir=%s", _data_dir)

    yield  # ← app runs here

    if _scheduler:
        _scheduler.stop()
    if _registry:
        _registry.close()
    log.info("Paper Tracker API stopped.")


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Paper Tracker API", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _maybe_import_legacy(base_cfg: dict) -> None:
    """Import config.toml as a default topic if no topics exist yet."""
    if not _registry or _registry.list_topics():
        return
    search = base_cfg.get("search", {})
    if not search:
        return
    topic_id = "interactive-video-world-models"
    _registry.create_topic({
        "id": topic_id,
        "name": "Interactive Video Generation / World Models",
        "description": "Auto-imported from config.toml",
        "arxiv_keywords": search.get("arxiv_keywords", []),
        "arxiv_categories": search.get("arxiv_categories", []),
        "arxiv_lookback_days": search.get("arxiv_lookback_days", 2),
        "github_keywords": search.get("github_keywords", []),
        "github_lookback_days": search.get("github_lookback_days", 7),
        "schedule_cron": "",
        "enabled": True,
    })
    log.info("Auto-imported legacy topic '%s'", topic_id)


def _get_registry() -> Registry:
    if _registry is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return _registry


def _get_scheduler() -> Scheduler:
    if _scheduler is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return _scheduler


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class TopicCreate(BaseModel):
    name: str
    description: str = ""
    arxiv_keywords: list[str] = []
    arxiv_categories: list[str] = []
    arxiv_lookback_days: int = 2
    github_keywords: list[str] = []
    github_lookback_days: int = 7
    schedule_cron: str = ""
    enabled: bool = True
    # OpenAlex
    openalex_enabled: bool = False
    openalex_keywords: list[str] = []
    openalex_lookback_days: int = 7
    openalex_venues: list[str] = []
    openalex_max_results: int = 200
    # OpenReview
    openreview_enabled: bool = False
    openreview_venues: list[str] = []
    openreview_keywords: list[str] = []
    openreview_max_results: int = 100
    # Date range override
    search_date_from: str = ""
    search_date_to: str = ""
    # Early relevance prefilter
    prefilter_enabled: bool = True
    prefilter_criteria: str = ""

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("name must not be empty")
        return v.strip()

    @field_validator("schedule_cron")
    @classmethod
    def valid_cron_or_empty(cls, v: str) -> str:
        if v and not re.match(r"^(\S+ ){4}\S+$", v.strip()):
            raise ValueError("schedule_cron must be a valid 5-field cron expression or empty")
        return v.strip()


class TopicUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    arxiv_keywords: list[str] | None = None
    arxiv_categories: list[str] | None = None
    arxiv_lookback_days: int | None = None
    github_keywords: list[str] | None = None
    github_lookback_days: int | None = None
    schedule_cron: str | None = None
    enabled: bool | None = None
    # OpenAlex
    openalex_enabled: bool | None = None
    openalex_keywords: list[str] | None = None
    openalex_lookback_days: int | None = None
    openalex_venues: list[str] | None = None
    openalex_max_results: int | None = None
    # OpenReview
    openreview_enabled: bool | None = None
    openreview_venues: list[str] | None = None
    openreview_keywords: list[str] | None = None
    openreview_max_results: int | None = None
    # Date range override
    search_date_from: str | None = None
    search_date_to: str | None = None
    # Early relevance prefilter
    prefilter_enabled: bool | None = None
    prefilter_criteria: str | None = None


# ---------------------------------------------------------------------------
# Topic endpoints
# ---------------------------------------------------------------------------

@app.get("/api/topics")
async def list_topics() -> list[dict]:
    reg = _get_registry()
    sched = _get_scheduler()
    topics = reg.list_topics()
    for t in topics:
        t["is_running"] = sched.is_running(t["id"])
        latest = reg.get_latest_session(t["id"])
        t["latest_session"] = latest
    return topics


class QuickTopicCreate(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("name must not be empty")
        return v.strip()


def _generate_topic_config(name: str) -> dict:
    """Use LLM to generate keywords, description, categories from a topic name."""
    from paper_tracker.llm import call_cli
    import json as _json

    prompt = f"""Given the research topic name: "{name}"

Generate a JSON object with:
- "description": A concise 1-sentence description of this research area
- "arxiv_keywords": A list of 6-8 search keyword phrases for finding relevant papers on arXiv. IMPORTANT: these are used as exact phrase matches (all:"keyword") on arXiv API, so:
  - At least 3 keywords MUST be short (1-2 words), e.g. "LLM agent", "diffusion"
  - Remaining keywords can be 2-3 words max, e.g. "multi-step reasoning"
  - NEVER use 4+ word phrases — they match almost nothing on arXiv
  - Mix broad terms (high recall) with specific terms (high precision)
- "arxiv_categories": A list of relevant arXiv categories (e.g. "cs.CV", "cs.AI", "cs.LG", "cs.CL", "cs.RO", "stat.ML")
- "github_keywords": A list of 2-4 keyword phrases for finding related GitHub repos

Output ONLY the JSON object, no markdown fences, no explanation."""

    cfg = {"summarizer": {"claude_path": "claude", "claude_model": "sonnet", "claude_timeout": 300}}
    raw = call_cli(prompt, cfg, model="sonnet", timeout=300)
    if raw:
        # Strip markdown fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```\w*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
        try:
            result = _json.loads(raw)
            # Post-process: split overly long keywords and ensure short ones exist
            if "arxiv_keywords" in result:
                result["arxiv_keywords"] = _fix_keywords(result["arxiv_keywords"])
            return result
        except _json.JSONDecodeError:
            log.warning("Failed to parse LLM config output: %s", raw[:300])
    return {}


def _fix_keywords(keywords: list[str]) -> list[str]:
    """Ensure keywords are short enough for arXiv exact-phrase matching.

    - Split any 4+ word phrase into shorter sub-phrases
    - Ensure at least 3 keywords are 1-2 words
    """
    fixed: list[str] = []
    for kw in keywords:
        words = kw.strip().split()
        if len(words) <= 2:
            fixed.append(kw.strip())
        elif len(words) == 3:
            # Keep the 3-word phrase AND add a 2-word sub-phrase
            fixed.append(kw.strip())
            fixed.append(" ".join(words[:2]))
        else:
            # Split "agentic AI long-horizon tasks" → "agentic AI", "long-horizon tasks"
            mid = len(words) // 2
            fixed.append(" ".join(words[:mid]))
            fixed.append(" ".join(words[mid:]))

    # Deduplicate preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for kw in fixed:
        low = kw.lower()
        if low not in seen:
            seen.add(low)
            deduped.append(kw)

    # Ensure at least 3 short (1-2 word) keywords
    short_count = sum(1 for kw in deduped if len(kw.split()) <= 2)
    if short_count < 3:
        # Extract individual meaningful words from longer keywords as extra short keywords
        all_words = set()
        for kw in deduped:
            for w in kw.split():
                if len(w) >= 3 and w.lower() not in {"the", "and", "for", "with"}:
                    all_words.add(w)
        existing_lower = {kw.lower() for kw in deduped}
        for w in sorted(all_words, key=len, reverse=True):
            if w.lower() not in existing_lower:
                deduped.append(w)
                existing_lower.add(w.lower())
                short_count += 1
            if short_count >= 3:
                break

    return deduped


@app.post("/api/topics/quick", status_code=201)
async def quick_create_topic(body: QuickTopicCreate) -> dict:
    """Create a topic from just a name. Auto-generates keywords/config via LLM."""
    from datetime import datetime, timedelta, timezone

    reg = _get_registry()
    topic_id = re.sub(r"[^a-z0-9-]", "-", body.name.lower())[:64].strip("-")
    if reg.get_topic(topic_id):
        topic_id = f"{topic_id}-{uuid.uuid4().hex[:6]}"

    # Generate config via LLM
    generated = _generate_topic_config(body.name)

    # Default: search from 1 year ago to today
    date_from = (datetime.now(timezone.utc) - timedelta(days=365)).strftime("%Y-%m-%d")
    date_to = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    arxiv_kws = generated.get("arxiv_keywords", [body.name])

    topic = reg.create_topic({
        "id": topic_id,
        "name": body.name,
        "description": generated.get("description", ""),
        "arxiv_keywords": arxiv_kws,
        "arxiv_categories": generated.get("arxiv_categories", ["cs.CV", "cs.AI", "cs.LG"]),
        "arxiv_lookback_days": 365,
        "github_keywords": generated.get("github_keywords", [body.name]),
        "github_lookback_days": 365,
        "schedule_cron": "",
        "enabled": True,
        "search_date_from": date_from,
        "search_date_to": date_to,
        # Enable OpenAlex & OpenReview by default, reuse arXiv keywords
        "openalex_enabled": True,
        "openalex_keywords": arxiv_kws[:4],
        "openalex_lookback_days": 365,
        "openreview_enabled": True,
        "openreview_keywords": arxiv_kws[:4],
    })
    return topic


# ---------------------------------------------------------------------------
# Guided topic creation (streaming)
# ---------------------------------------------------------------------------

class GuidedMessage(BaseModel):
    role: str
    content: str

class GuidedRequest(BaseModel):
    messages: list[GuidedMessage]


@app.post("/api/topics/guided")
async def guided_create_topic(body: GuidedRequest):
    """SSE streaming endpoint for guided topic creation conversation."""
    from paper_tracker.guided import stream_guided_response

    messages = [{"role": m.role, "content": m.content} for m in body.messages]
    return StreamingResponse(
        stream_guided_response(messages),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/topics", status_code=201)
async def create_topic(body: TopicCreate) -> dict:
    reg = _get_registry()
    topic_id = re.sub(r"[^a-z0-9-]", "-", body.name.lower())[:64].strip("-")
    # Ensure uniqueness
    if reg.get_topic(topic_id):
        topic_id = f"{topic_id}-{uuid.uuid4().hex[:6]}"
    topic = reg.create_topic({
        "id": topic_id,
        **body.model_dump(),
    })
    # Register cron if provided
    if body.schedule_cron:
        _get_scheduler().update_schedule(topic_id, body.schedule_cron)
    return topic


@app.get("/api/topics/{topic_id}")
async def get_topic(topic_id: str) -> dict:
    reg = _get_registry()
    topic = reg.get_topic(topic_id)
    if not topic:
        raise HTTPException(404, detail="Topic not found")
    topic["is_running"] = _get_scheduler().is_running(topic_id)
    topic["latest_session"] = reg.get_latest_session(topic_id)
    return topic


@app.put("/api/topics/{topic_id}")
async def update_topic(topic_id: str, body: TopicUpdate) -> dict:
    reg = _get_registry()
    if not reg.get_topic(topic_id):
        raise HTTPException(404, detail="Topic not found")
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    topic = reg.update_topic(topic_id, updates)
    if "schedule_cron" in updates:
        _get_scheduler().update_schedule(topic_id, updates["schedule_cron"])
    return topic


@app.delete("/api/topics/{topic_id}", status_code=204)
async def delete_topic(topic_id: str) -> None:
    reg = _get_registry()
    if not reg.get_topic(topic_id):
        raise HTTPException(404, detail="Topic not found")
    reg.delete_topic(topic_id)


# ---------------------------------------------------------------------------
# Run / stop endpoints
# ---------------------------------------------------------------------------

@app.post("/api/topics/{topic_id}/run", status_code=202)
async def run_topic(topic_id: str) -> dict:
    reg = _get_registry()
    if not reg.get_topic(topic_id):
        raise HTTPException(404, detail="Topic not found")
    sched = _get_scheduler()
    started = sched.trigger_now(topic_id)
    if not started:
        raise HTTPException(409, detail="Topic is already running")
    return {"status": "triggered", "topic_id": topic_id}


@app.post("/api/topics/{topic_id}/stop")
async def stop_topic(topic_id: str) -> dict:
    sched = _get_scheduler()
    cancelled = sched.cancel(topic_id)
    return {"cancelled": cancelled, "topic_id": topic_id}


@app.get("/api/topics/{topic_id}/progress")
async def get_topic_progress(topic_id: str) -> dict:
    """Get live pipeline progress for a running topic."""
    sched = _get_scheduler()
    if not sched.is_running(topic_id):
        return {"running": False, "topic_id": topic_id}
    progress = sched.progress.get(topic_id, {})
    return {"running": True, "topic_id": topic_id, **progress}


# ---------------------------------------------------------------------------
# Session endpoints
# ---------------------------------------------------------------------------

@app.get("/api/topics/{topic_id}/sessions")
async def list_sessions(
    topic_id: str,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> dict:
    reg = _get_registry()
    if not reg.get_topic(topic_id):
        raise HTTPException(404, detail="Topic not found")
    sessions = reg.list_sessions(topic_id, limit=limit, offset=offset)
    return {"sessions": sessions, "limit": limit, "offset": offset}


@app.get("/api/topics/{topic_id}/sessions/{session_id}")
async def get_session(topic_id: str, session_id: str) -> dict:
    reg = _get_registry()
    session = reg.get_session(topic_id, session_id)
    if not session:
        raise HTTPException(404, detail="Session not found")

    # Inline report and insights content
    if session.get("report_path") and Path(session["report_path"]).exists():
        session["report_content"] = Path(session["report_path"]).read_text(encoding="utf-8")
    else:
        session["report_content"] = ""

    if session.get("insights_path") and Path(session["insights_path"]).exists():
        session["insights_content"] = Path(session["insights_path"]).read_text(encoding="utf-8")
    else:
        session["insights_content"] = ""

    return session


# ---------------------------------------------------------------------------
# Insights endpoint
# ---------------------------------------------------------------------------

@app.get("/api/topics/{topic_id}/insights")
async def get_latest_insights(topic_id: str) -> dict:
    reg = _get_registry()
    if not reg.get_topic(topic_id):
        raise HTTPException(404, detail="Topic not found")
    session = reg.get_latest_session(topic_id)
    content = ""
    if session and session.get("insights_path"):
        p = Path(session["insights_path"])
        if p.exists():
            content = p.read_text(encoding="utf-8")
    return {"topic_id": topic_id, "content": content, "session": session}


# ---------------------------------------------------------------------------
# Generate insights from selected papers (manual, cross-domain enrichment)
# ---------------------------------------------------------------------------

class SuggestCrossDomainRequest(BaseModel):
    paper_ids: list[str]
    live_search: bool = False  # also run live arXiv keyword search (opt-in)


class CrossDomainPaperIn(BaseModel):
    arxiv_id: str = ""
    title: str = ""
    authors: str = ""
    abstract: str = ""
    url: str = ""
    published: str = ""
    domain: str = ""


class GenerateInsightsRequest(BaseModel):
    paper_ids: list[str]
    cross_domain_papers: list[CrossDomainPaperIn] = []
    title: str = ""
    mode: str = "single"  # "single" (one-call) or "agentic" (multi-stage pipeline)


@app.post("/api/topics/{topic_id}/insights/suggest-cross-domain")
async def suggest_cross_domain_route(topic_id: str, body: SuggestCrossDomainRequest) -> dict:
    """Suggest cross-domain papers to enrich a selection (opus + arXiv). Synchronous."""
    reg = _get_registry()
    topic = reg.get_topic(topic_id)
    if not topic:
        raise HTTPException(404, detail="Topic not found")
    if not body.paper_ids:
        raise HTTPException(400, detail="paper_ids is required")
    store = Storage(_data_dir, topic_id)
    try:
        selected = [p for p in (store.get_arxiv(pid) for pid in body.paper_ids) if p]
    finally:
        store.close()
    if not selected:
        raise HTTPException(400, detail="None of the selected papers were found")

    # 1. Curated knowledge base via vector retrieval (default smart path).
    corpus = _corpus_store()
    try:
        kb = crossdomain.retrieve(selected, corpus, k=12)
    finally:
        corpus.close()

    candidates = list(kb)
    seen = {c["arxiv_id"] for c in candidates}

    # 2. Optional live arXiv keyword search (manual toggle).
    if body.live_search:
        topic_cfg = cfg_module.from_topic(topic, _base_cfg)
        for c in insights.suggest_cross_domain(selected, topic["name"], topic_cfg):
            if c.get("arxiv_id") in seen:
                continue
            seen.add(c["arxiv_id"])
            candidates.append({**c, "source": "live"})

    return {"candidates": candidates, "kb_count": len(kb), "live": bool(body.live_search)}


@app.post("/api/topics/{topic_id}/insights/generate", status_code=202)
async def generate_insights_route(topic_id: str, body: GenerateInsightsRequest) -> dict:
    """Generate an insights memo from selected (+ chosen cross-domain) papers.

    Creates a ``kind='manual'`` session and runs insights.generate in the
    background; poll GET .../sessions/{session_id} until status != 'running'.
    """
    reg = _get_registry()
    topic = reg.get_topic(topic_id)
    if not topic:
        raise HTTPException(404, detail="Topic not found")
    if not body.paper_ids:
        raise HTTPException(400, detail="paper_ids is required")

    session = reg.create_session(topic_id, kind="manual")
    sid = session["id"]
    session_dir = Path(_data_dir) / topic_id / "sessions" / sid
    topic_cfg = cfg_module.from_topic(topic, _base_cfg)
    topic_name = topic["name"]
    paper_ids = list(body.paper_ids)
    cross = [c.model_dump() for c in body.cross_domain_papers]
    mode = "agentic" if body.mode == "agentic" else "single"

    def _finish(updates: dict) -> None:
        updates["finished_at"] = datetime.now(timezone.utc).isoformat()
        reg.update_session(topic_id, sid, updates)

    def _run():
        store = Storage(_data_dir, topic_id)
        try:
            papers = [p for p in (store.get_arxiv(pid) for pid in paper_ids) if p]
            for c in cross:
                papers.append({
                    "arxiv_id": c.get("arxiv_id", ""),
                    "title": c.get("title", ""),
                    "authors": c.get("authors", ""),
                    "venue": c.get("domain", ""),
                    "abstract": c.get("abstract", ""),
                    "summary": c.get("abstract", ""),  # abstract stands in as summary
                    "key_insight": "", "method": "", "contribution": "",
                    "math_concepts": [],
                })
            if not papers:
                _finish({"status": "failed", "error_message": "No papers found"})
                return
            if mode == "agentic":
                path = insights.generate_agentic(
                    papers, topic_name, session_dir, topic_cfg,
                    progress_cb=lambda msg: log.info("insights[%s] %s", sid, msg),
                )
            else:
                path = insights.generate(papers, topic_name, session_dir, topic_cfg)
            if path:
                _finish({"status": "completed", "insights_path": str(path),
                         "paper_count": len(papers)})
            else:
                _finish({"status": "failed",
                         "error_message": "Insights generation failed"})
        except Exception as e:
            log.exception("Manual insights generation failed: %s", e)
            _finish({"status": "failed", "error_message": str(e)})
        finally:
            store.close()

    _brainstorm_executor.submit(_run)
    return {"session_id": sid, "status": "running"}


# ---------------------------------------------------------------------------
# Papers / Repos endpoints (per-topic paper library)
# ---------------------------------------------------------------------------

@app.get("/api/topics/{topic_id}/papers")
async def list_papers(
    topic_id: str,
    search: str = Query(default="", description="Search title/abstract/authors"),
    venue: str = Query(default="", description="Filter by venue"),
    source: str = Query(default="", description="Filter by source (arxiv, openalex, openreview)"),
    date_from: str = Query(default="", description="Filter papers published on or after this date (YYYY-MM-DD)"),
    date_to: str = Query(default="", description="Filter papers published on or before this date (YYYY-MM-DD)"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> dict:
    reg = _get_registry()
    if not reg.get_topic(topic_id):
        raise HTTPException(404, detail="Topic not found")
    store = Storage(_data_dir, topic_id)
    try:
        papers, total = store.get_all_arxiv(
            search=search, venue=venue, source=source,
            date_from=date_from, date_to=date_to,
            limit=limit, offset=offset,
        )
    finally:
        store.close()
    return {"papers": papers, "total": total, "limit": limit, "offset": offset}


# ---------------------------------------------------------------------------
# Refilter endpoints (must be BEFORE {arxiv_id:path} to avoid path capture)
# ---------------------------------------------------------------------------

class RefilterRequest(BaseModel):
    custom_instructions: str = ""
    min_quality: int = 3
    auto_delete: bool = False


_refilter_jobs: dict[str, dict] = {}


@app.post("/api/topics/{topic_id}/papers/refilter", status_code=202)
async def start_refilter(topic_id: str, body: RefilterRequest) -> dict:
    reg = _get_registry()
    topic = reg.get_topic(topic_id)
    if not topic:
        raise HTTPException(404, detail="Topic not found")

    if topic_id in _refilter_jobs and _refilter_jobs[topic_id].get("status") == "running":
        raise HTTPException(409, detail="Refilter already running for this topic")

    _refilter_jobs[topic_id] = {
        "status": "running",
        "total": 0,
        "processed": 0,
        "removed": 0,
    }

    topic_cfg = cfg_module.from_topic(topic, _base_cfg)

    def _run():
        try:
            store = Storage(_data_dir, topic_id)
            try:
                papers, total = store.get_all_arxiv(limit=10000, offset=0)
                _refilter_jobs[topic_id]["total"] = len(papers)

                def _on_batch(processed_count):
                    _refilter_jobs[topic_id]["processed"] = processed_count

                refiltered = refilter_papers(
                    papers,
                    topic_cfg,
                    topic_name=topic["name"],
                    keywords=topic.get("arxiv_keywords", []),
                    custom_instructions=body.custom_instructions,
                    on_batch_done=_on_batch,
                )

                # Update scores in DB
                for p in refiltered:
                    store.update_arxiv_quality(p["arxiv_id"], p["quality_score"])

                # Auto-delete if requested
                removed = 0
                if body.auto_delete:
                    removed = store.delete_arxiv_below_quality(body.min_quality)

                _refilter_jobs[topic_id].update({
                    "status": "completed",
                    "removed": removed,
                })
            finally:
                store.close()
        except Exception as e:
            log.exception("Refilter failed: %s", e)
            _refilter_jobs[topic_id]["status"] = "failed"

    _brainstorm_executor.submit(_run)
    return {"status": "started", "topic_id": topic_id}


@app.get("/api/topics/{topic_id}/papers/refilter")
async def get_refilter_status(topic_id: str) -> dict:
    reg = _get_registry()
    if not reg.get_topic(topic_id):
        raise HTTPException(404, detail="Topic not found")
    job = _refilter_jobs.get(topic_id, {"status": "idle", "total": 0, "processed": 0, "removed": 0})
    return {"topic_id": topic_id, **job}


# ---------------------------------------------------------------------------
# Backfill missing summaries (manual/seeded/batch-added papers have none)
# ---------------------------------------------------------------------------

_summary_backfill_jobs: dict[str, dict] = {}
_BACKFILL_CHUNK = 10


def _needs_summary(p: dict) -> bool:
    """A paper is unsummarized if it has no key_insight (pipeline papers do)."""
    return not (p.get("key_insight") or "").strip()


@app.post("/api/topics/{topic_id}/papers/backfill-summaries", status_code=202)
async def start_backfill_summaries(topic_id: str) -> dict:
    """Scan the library and summarize papers that have no summary yet (sonnet,
    background, batched). Poll GET for {status,total,processed}."""
    reg = _get_registry()
    topic = reg.get_topic(topic_id)
    if not topic:
        raise HTTPException(404, detail="Topic not found")
    if _summary_backfill_jobs.get(topic_id, {}).get("status") == "running":
        raise HTTPException(409, detail="Summary backfill already running for this topic")

    store = Storage(_data_dir, topic_id)
    try:
        papers, _ = store.get_all_arxiv(limit=10000, offset=0)
    finally:
        store.close()
    missing_ids = [p["arxiv_id"] for p in papers if _needs_summary(p)]

    if not missing_ids:
        _summary_backfill_jobs[topic_id] = {"status": "completed", "total": 0, "processed": 0}
        return {"status": "completed", "topic_id": topic_id, "total": 0, "processed": 0}

    _summary_backfill_jobs[topic_id] = {
        "status": "running", "total": len(missing_ids), "processed": 0,
    }
    topic_cfg = cfg_module.from_topic(topic, _base_cfg)

    def _run():
        job = _summary_backfill_jobs[topic_id]
        store = Storage(_data_dir, topic_id)
        try:
            for i in range(0, len(missing_ids), _BACKFILL_CHUNK):
                chunk_ids = missing_ids[i : i + _BACKFILL_CHUNK]
                chunk = [p for p in (store.get_arxiv(a) for a in chunk_ids) if p]
                if chunk:
                    summarize_papers(chunk, topic_cfg)  # batches internally
                    for p in chunk:
                        store.update_arxiv_summary(p["arxiv_id"], {
                            "summary": p.get("summary", ""),
                            "key_insight": p.get("key_insight", ""),
                            "method": p.get("method", ""),
                            "contribution": p.get("contribution", ""),
                            "math_concepts": p.get("math_concepts", []),
                            "venue": p.get("venue", ""),
                            "cited_works": p.get("cited_works", []),
                        })
                job["processed"] = min(job["total"], i + len(chunk_ids))
            job["status"] = "completed"
        except Exception as e:
            log.exception("Summary backfill failed: %s", e)
            job["status"] = "failed"
        finally:
            store.close()

    _brainstorm_executor.submit(_run)
    return {"status": "started", "topic_id": topic_id, "total": len(missing_ids)}


@app.get("/api/topics/{topic_id}/papers/backfill-summaries")
async def get_backfill_summaries_status(topic_id: str) -> dict:
    reg = _get_registry()
    if not reg.get_topic(topic_id):
        raise HTTPException(404, detail="Topic not found")
    job = _summary_backfill_jobs.get(topic_id, {"status": "idle", "total": 0, "processed": 0})
    return {"topic_id": topic_id, **job}


# ---------------------------------------------------------------------------
# Cross-domain knowledge base (global curated corpus: screen + vector retrieval)
# ---------------------------------------------------------------------------

_corpus_embed_jobs: dict = {}  # keyed by "corpus"


def _corpus_store() -> Storage:
    return Storage(_data_dir, crossdomain.CROSSDOMAIN_ID)


def _corpus_insert(store: Storage, meta: dict, domain: str, reason: str) -> str:
    """Insert a paper (from any source) into the corpus. Returns its key.

    Uses the source's own ``paper_id``/``doi`` (arXiv id, ``doi:…``, ``oa:…``,
    ``pdf:…``) so non-arXiv work is stored correctly.
    """
    key = meta.get("paper_id") or meta["arxiv_id"]
    store.insert_arxiv({
        "arxiv_id": key, "paper_id": key, "source": "crossdomain",
        "title": meta.get("title", ""), "authors": meta.get("authors", ""),
        "abstract": meta.get("abstract", ""),
        "url": meta.get("url", ""),
        "published": (meta.get("published", "") or "")[:10],
        "summary": meta.get("summary") or meta.get("abstract", ""),  # feeds embed text
        "key_insight": reason, "method": "", "contribution": "",
        "math_concepts": [], "venue": domain or meta.get("venue", ""), "cited_works": [],
        "quality_score": 0, "citation_count": meta.get("citation_count", 0),
        "doi": meta.get("doi", ""),
    })
    return key


def _submit_corpus_embed_job() -> None:
    if _corpus_embed_jobs.get("corpus", {}).get("status") == "running":
        return
    _corpus_embed_jobs["corpus"] = {"status": "running", "embedded": 0}

    def _run():
        store = _corpus_store()
        try:
            n = rag.ensure_embeddings(store)
            _corpus_embed_jobs["corpus"] = {
                "status": "completed", "embedded": n, "total": store.embedding_count(),
            }
        except Exception as e:
            log.exception("Corpus embedding failed: %s", e)
            _corpus_embed_jobs["corpus"] = {"status": "failed", "embedded": 0, "error": str(e)}
        finally:
            store.close()

    _brainstorm_executor.submit(_run)


class CorpusAddRequest(BaseModel):
    text: str = ""
    arxiv_ids: list[str] = []
    skip_screen: bool = False


class CorpusImportRequest(BaseModel):
    categories: list[str] = []
    keyword: str = ""
    max: int = 30


@app.get("/api/crossdomain/papers")
async def list_corpus_papers(
    search: str = Query(default=""),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> dict:
    store = _corpus_store()
    try:
        papers, total = store.get_all_arxiv(search=search, limit=limit, offset=offset)
        emb = store.embedding_count()
    finally:
        store.close()
    return {
        "papers": papers, "total": total, "limit": limit, "offset": offset,
        "embedding_count": emb,
        "embedding_job": _corpus_embed_jobs.get("corpus", {"status": "idle"}),
    }


@app.post("/api/crossdomain/papers", status_code=201)
async def add_corpus_papers(body: CorpusAddRequest) -> dict:
    """Add papers to the corpus from pasted IDs/URLs or an explicit id list.

    Each is screened (opus) unless ``skip_screen``; kept papers are inserted and
    a background embedding job is kicked. Per-item: added | skipped(reason) |
    duplicate | not_found | invalid | error.
    """
    tokens = [t for t in re.split(r"[\s,]+", body.text.strip()) if t]
    tokens += [s.strip() for s in body.arxiv_ids if s.strip()]
    if not tokens:
        raise HTTPException(400, detail="No URLs or arXiv IDs provided")
    seen_tok: set[str] = set()
    uniq = [t for t in tokens if not (t in seen_tok or seen_tok.add(t))]

    results: list[dict] = []
    added_ids: list[str] = []
    store = _corpus_store()
    try:
        # Classify each token: arXiv id/URL, PDF URL, or invalid.
        pending: list[tuple[str, str, str]] = []  # (token, kind, key)
        for tok in uniq:
            cid = arxiv.extract_arxiv_id(tok)
            if cid:
                kind, key, doi = "arxiv", cid, f"10.48550/arxiv.{cid}"
            elif pdf_source.is_pdf_url(tok):
                kind = "pdf"
                key = "pdf:" + hashlib.sha1(tok.strip().encode()).hexdigest()[:12]
                doi = ""
            else:
                results.append({"query": tok, "status": "invalid"})
                continue
            if store.is_paper_seen(key, doi):
                results.append({"query": tok, "status": "duplicate", "arxiv_id": key})
                continue
            pending.append((tok, kind, key))

        # Fetch metadata: arXiv in one batch, PDFs individually.
        arxiv_keys = [k for _, kind, k in pending if kind == "arxiv"]
        fetched: dict[str, dict] = {}
        fetch_failed = False
        if arxiv_keys:
            try:
                fetched = arxiv.fetch_many_by_id(arxiv_keys)
            except httpx.HTTPError as e:
                log.warning("Corpus add: arXiv fetch failed: %s", e)
                fetch_failed = True

        to_consider: list[tuple[str, dict]] = []  # (token, meta)
        for tok, kind, key in pending:
            if kind == "arxiv":
                if fetch_failed:
                    results.append({"query": tok, "status": "error", "arxiv_id": key})
                    continue
                meta = fetched.get(key)
            else:  # pdf — download + parse + LLM-extract metadata
                meta = pdf_source.fetch_pdf_paper(tok, _base_cfg)
            if not meta:
                results.append({"query": tok, "status": "not_found", "arxiv_id": key})
                continue
            to_consider.append((tok, meta))

        vmap: dict[str, dict] = {}
        if to_consider and not body.skip_screen:
            verdicts = crossdomain.screen_papers([m for _, m in to_consider], _base_cfg)
            vmap = {v["arxiv_id"]: v for v in verdicts}

        for tok, meta in to_consider:
            mkey = meta.get("paper_id") or meta["arxiv_id"]  # source's own id
            keep, domain, reason = True, "", ""
            if not body.skip_screen:
                v = vmap.get(mkey, {})
                keep, domain, reason = v.get("keep", True), v.get("domain", ""), v.get("reason", "")
            if mkey in added_ids:
                results.append({"query": tok, "status": "duplicate", "arxiv_id": mkey})
                continue
            if not keep:
                results.append({"query": tok, "status": "skipped", "arxiv_id": mkey,
                                "reason": reason, "title": meta.get("title", "")})
                continue
            _corpus_insert(store, meta, domain, reason)
            added_ids.append(mkey)
            results.append({"query": tok, "status": "added", "arxiv_id": mkey,
                            "title": meta.get("title", ""), "domain": domain})
    finally:
        store.close()

    if added_ids:
        _submit_corpus_embed_job()

    counts = {k: sum(1 for r in results if r["status"] == k)
              for k in ("added", "skipped", "duplicate", "not_found", "invalid", "error")}
    return {"results": results, "counts": counts, "embedding_started": bool(added_ids)}


@app.post("/api/crossdomain/import-search")
async def corpus_import_search(body: CorpusImportRequest) -> dict:
    """Find candidate corpus papers by arXiv category (or keyword), screen them,
    and return for review (NOT saved). Confirm by POSTing the chosen ids to
    /api/crossdomain/papers with skip_screen=true."""
    cats = [c.strip() for c in body.categories if c.strip()] or crossdomain.DEFAULT_IMPORT_CATEGORIES
    kw = body.keyword.strip()
    maxn = max(1, min(body.max, 100))
    try:
        if kw:
            # keyword → all-time relevance search (good for seminal work by concept)
            papers = arxiv.search_by_query(kw, max_results=maxn)
        else:
            # 3-pool gather (recent + historical/seminal + wildcard) — shared with
            # Math Insights so the importer surfaces classic theory, not just recent.
            r = max(1, maxn // 2)
            h = max(1, maxn // 3)
            papers = gather_cross_domain_papers(
                cats, lookback_days=365,
                max_recent=r, max_historical=h, max_wildcard=max(1, maxn - r - h),
            )
    except Exception as e:
        log.warning("Corpus import search failed: %s", e)
        raise HTTPException(502, detail="arXiv search failed; please try again.")

    store = _corpus_store()
    try:
        papers = [p for p in papers
                  if not store.is_paper_seen(p.get("paper_id", p["arxiv_id"]), p.get("doi", ""))]
    finally:
        store.close()
    if not papers:
        return {"candidates": []}

    verdicts = crossdomain.screen_papers(papers, _base_cfg)
    vmap = {v["arxiv_id"]: v for v in verdicts}
    candidates = [{
        "arxiv_id": p["arxiv_id"], "title": p.get("title", ""), "authors": p.get("authors", ""),
        "abstract": p.get("abstract", ""), "url": p.get("url", ""),
        "published": (p.get("published", "") or "")[:10],
        "keep": bool(vmap.get(p["arxiv_id"], {}).get("keep", False)),
        "domain": vmap.get(p["arxiv_id"], {}).get("domain", ""),
        "reason": vmap.get(p["arxiv_id"], {}).get("reason", ""),
    } for p in papers]
    return {"candidates": candidates}


@app.delete("/api/crossdomain/papers/{arxiv_id:path}", status_code=204)
async def delete_corpus_paper(arxiv_id: str) -> None:
    store = _corpus_store()
    try:
        deleted = store.delete_arxiv(arxiv_id)
        store.delete_embedding(arxiv_id)
    finally:
        store.close()
    if not deleted:
        raise HTTPException(404, detail="Paper not found")


@app.post("/api/crossdomain/embeddings", status_code=202)
async def build_corpus_embeddings() -> dict:
    _submit_corpus_embed_job()
    return {"status": "started"}


@app.get("/api/crossdomain/embeddings")
async def get_corpus_embeddings_status() -> dict:
    store = _corpus_store()
    try:
        _, total = store.get_all_arxiv(limit=1, offset=0)
        emb = store.embedding_count()
    finally:
        store.close()
    return {"paper_count": total, "embedding_count": emb,
            "job": _corpus_embed_jobs.get("corpus", {"status": "idle"})}


_corpus_import_jobs: dict = {}  # keyed by "import"


class ImportReportRequest(BaseModel):
    report_id: str


@app.post("/api/crossdomain/import-report", status_code=202)
async def import_report_to_corpus(body: ImportReportRequest) -> dict:
    """Promote a discovery report's papers (e.g. Math Insights) into the corpus:
    fetch metadata → screen → insert kept → embed, in the background."""
    reg = _get_registry()
    report = reg.get_discovery_report(body.report_id)
    if not report:
        raise HTTPException(404, detail="Report not found")
    if _corpus_import_jobs.get("import", {}).get("status") == "running":
        raise HTTPException(409, detail="A corpus import is already running")

    pj = report.get("papers_json") or []
    if isinstance(pj, str):
        try:
            pj = json.loads(pj)
        except (json.JSONDecodeError, TypeError):
            pj = []
    ids = [p["arxiv_id"] for p in pj if isinstance(p, dict) and p.get("arxiv_id")]
    if not ids:
        raise HTTPException(400, detail="Report has no papers to import")

    _corpus_import_jobs["import"] = {"status": "running", "total": len(ids),
                                     "added": 0, "skipped": 0, "processed": 0, "duplicate": 0}
    cfg = _base_cfg

    def _run():
        job = _corpus_import_jobs["import"]
        store = _corpus_store()
        try:
            todo = [i for i in ids if not store.is_paper_seen(i, f"10.48550/arxiv.{i}")]
            job["duplicate"] = len(ids) - len(todo)
            job["total"] = len(todo)
            added = skipped = processed = 0
            CHUNK = 40
            for s in range(0, len(todo), CHUNK):
                chunk_ids = todo[s : s + CHUNK]
                try:
                    fetched = arxiv.fetch_many_by_id(chunk_ids)
                except httpx.HTTPError as e:
                    log.warning("import-report fetch failed: %s", e)
                    fetched = {}
                metas = [fetched[i] for i in chunk_ids if i in fetched]
                if metas:
                    verdicts = {v["arxiv_id"]: v for v in crossdomain.screen_papers(metas, cfg)}
                    for m in metas:
                        v = verdicts.get(m["arxiv_id"], {})
                        if v.get("keep", True):
                            _corpus_insert(store, m, v.get("domain", ""), v.get("reason", ""))
                            added += 1
                        else:
                            skipped += 1
                processed += len(chunk_ids)
                job.update({"added": added, "skipped": skipped, "processed": processed})
            try:
                rag.ensure_embeddings(store)
            except Exception as e:
                log.warning("import-report embed failed: %s", e)
            job.update({"status": "completed", "added": added, "skipped": skipped, "processed": len(todo)})
        except Exception as e:
            log.exception("import-report failed: %s", e)
            job["status"] = "failed"
        finally:
            store.close()

    _brainstorm_executor.submit(_run)
    return {"status": "started", "total": len(ids)}


@app.get("/api/crossdomain/import-report")
async def get_import_report_status() -> dict:
    return {"job": _corpus_import_jobs.get("import", {"status": "idle"})}


_corpus_card_jobs: dict = {}  # keyed by "cards"


class ConceptCardsRequest(BaseModel):
    force: bool = False  # regenerate even for papers that already have a card


_CARD_PDF_MAX_PAGES = 400   # bound text extraction for very long books
_LONG_DOC_CHARS = 16000     # above this, split a PDF into per-section cards


def _insert_card_child(store: Storage, child_id: str, parent: dict, title: str, card: dict) -> None:
    store.insert_arxiv({
        "arxiv_id": child_id, "paper_id": child_id, "source": "crossdomain",
        "title": title[:300], "authors": parent.get("authors", ""),
        "abstract": card["summary"], "url": parent.get("url", ""),
        "published": parent.get("published", ""), "summary": card["summary"],
        "key_insight": parent.get("key_insight", ""), "method": "", "contribution": "",
        "math_concepts": card["math_concepts"], "venue": parent.get("venue", ""),
        "cited_works": [], "quality_score": 0, "citation_count": 0, "doi": "",
    })


def _single_card(store: Storage, aid: str, paper: dict, cfg: dict, text: str | None = None) -> bool:
    card = crossdomain.generate_concept_card(paper, cfg, text=text)
    if not card:
        return False
    store.update_arxiv_summary(aid, {"summary": card["summary"], "math_concepts": card["math_concepts"]})
    store.delete_embedding(aid)  # force re-embed with card text
    return True


def _card_one_paper(store: Storage, aid: str, cfg: dict) -> int:
    """Generate concept card(s) for one corpus paper; returns the # of cards made.

    A long top-level PDF is split into per-section CHILD entries (C: headings,
    B: chunks) that replace the parent; short docs / arXiv get a single card.
    """
    p = store.get_arxiv(aid)
    if not p:
        return 0

    # Long-doc splitting only for top-level PDF entries (not already-split children).
    full_text = None
    if aid.startswith("pdf:") and "#" not in aid and p.get("url"):
        full_text = pdf_source.download_and_extract_text(p["url"], max_pages=_CARD_PDF_MAX_PAGES)

    if full_text and len(full_text) > _LONG_DOC_CHARS:
        pieces = crossdomain.split_into_sections(full_text)
        if len(pieces) >= 2:
            parent_title = p.get("title", "")
            made = 0
            for j, piece in enumerate(pieces, 1):
                title = f"{parent_title} — {piece['title']}"
                card = crossdomain.generate_concept_card(
                    {"title": title, "abstract": ""}, cfg, text=piece["text"])
                if card:
                    _insert_card_child(store, f"{aid}#{j:02d}", p, title, card)
                    made += 1
            if made:
                store.delete_arxiv(aid)        # replace the parent with its sections
                store.delete_embedding(aid)
                return made
            # split produced nothing usable → fall through to a single card

    return 1 if _single_card(store, aid, p, cfg, text=full_text) else 0


@app.post("/api/crossdomain/concept-cards", status_code=202)
async def build_concept_cards(body: ConceptCardsRequest) -> dict:
    """Generate concept cards (main math content) for corpus papers, in the
    background. For PDF-sourced papers the body text is re-extracted for depth;
    cards feed both display and (after re-embed) retrieval."""
    if _corpus_card_jobs.get("cards", {}).get("status") == "running":
        raise HTTPException(409, detail="Concept-card generation already running")

    store = _corpus_store()
    try:
        papers, _ = store.get_all_arxiv(limit=10000, offset=0)
    finally:
        store.close()
    # "needs a card" = no math_concepts yet (set only by card generation)
    targets = [p["arxiv_id"] for p in papers if body.force or not p.get("math_concepts")]
    if not targets:
        _corpus_card_jobs["cards"] = {"status": "completed", "total": 0, "processed": 0, "generated": 0}
        return {"status": "completed", "total": 0}

    _corpus_card_jobs["cards"] = {"status": "running", "total": len(targets),
                                  "processed": 0, "generated": 0}
    cfg = _base_cfg

    def _run():
        job = _corpus_card_jobs["cards"]
        store = _corpus_store()
        try:
            generated = 0
            for i, aid in enumerate(targets):
                try:
                    generated += _card_one_paper(store, aid, cfg)
                except Exception as e:
                    log.warning("Concept card for %s failed: %s", aid, e)
                job.update({"processed": i + 1, "generated": generated})
            try:
                rag.ensure_embeddings(store)  # re-embed refreshed/new papers
            except Exception as e:
                log.warning("concept-cards re-embed failed: %s", e)
            job["status"] = "completed"
        except Exception as e:
            log.exception("concept-card generation failed: %s", e)
            job["status"] = "failed"
        finally:
            store.close()

    _brainstorm_executor.submit(_run)
    return {"status": "started", "total": len(targets)}


@app.get("/api/crossdomain/concept-cards")
async def get_concept_cards_status() -> dict:
    return {"job": _corpus_card_jobs.get("cards", {"status": "idle"})}


# ---------------------------------------------------------------------------
# Manual add-paper endpoints (must be BEFORE {arxiv_id:path} to avoid capture)
# ---------------------------------------------------------------------------

class PaperLookupRequest(BaseModel):
    query: str


class PaperCreateRequest(BaseModel):
    arxiv_id: str = ""          # raw arXiv id or URL; blank for a non-arXiv paper
    title: str
    authors: str = ""
    abstract: str = ""
    url: str = ""
    published: str = ""         # YYYY-MM-DD
    venue: str = ""
    summarize: bool = False

    @field_validator("title")
    @classmethod
    def _title_required(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("title is required")
        return v.strip()


@app.post("/api/topics/{topic_id}/papers/lookup")
async def lookup_paper(topic_id: str, body: PaperLookupRequest) -> dict:
    """Fetch arXiv metadata for preview without saving. Returns {found, paper}."""
    reg = _get_registry()
    if not reg.get_topic(topic_id):
        raise HTTPException(404, detail="Topic not found")
    query = body.query.strip()
    if not query:
        raise HTTPException(400, detail="query is required")
    try:
        paper = arxiv.fetch_by_id(query)
    except httpx.HTTPError as e:
        log.warning("arXiv lookup failed for %r: %s", query, e)
        raise HTTPException(
            502, detail="arXiv lookup failed; try again or enter details manually."
        )
    return {"found": paper is not None, "paper": paper}


def _submit_summarize_job(topic: dict, topic_id: str, arxiv_id: str) -> None:
    """Background-summarize a freshly added paper and write the fields back."""
    topic_cfg = cfg_module.from_topic(topic, _base_cfg)

    def _run():
        store = Storage(_data_dir, topic_id)
        try:
            paper = store.get_arxiv(arxiv_id)
            if not paper:
                return
            papers = [paper]
            summarize_papers(papers, topic_cfg)
            p = papers[0]
            store.update_arxiv_summary(arxiv_id, {
                "summary": p.get("summary", ""),
                "key_insight": p.get("key_insight", ""),
                "method": p.get("method", ""),
                "contribution": p.get("contribution", ""),
                "math_concepts": p.get("math_concepts", []),
                # keep the user's venue if the summarizer didn't infer one
                "venue": p.get("venue") or paper.get("venue", ""),
                "cited_works": p.get("cited_works", []),
            })
            log.info("Summarized manually-added paper %s (topic %s)", arxiv_id, topic_id)
        except Exception as e:
            log.exception("Summarize job for manual paper %s failed: %s", arxiv_id, e)
        finally:
            store.close()

    _brainstorm_executor.submit(_run)


@app.post("/api/topics/{topic_id}/papers", status_code=201)
async def add_paper(topic_id: str, body: PaperCreateRequest) -> dict:
    """Manually add a paper. arxiv_id (id or URL) is optional; blank = free-form.

    Dedups against existing papers (409 if present). When ``summarize`` is set,
    an LLM summarize job runs in the background and fills the structured fields.
    """
    reg = _get_registry()
    topic = reg.get_topic(topic_id)
    if not topic:
        raise HTTPException(404, detail="Topic not found")

    raw_id = body.arxiv_id.strip()
    arxiv_id = arxiv.extract_arxiv_id(raw_id) if raw_id else ""
    if raw_id and not arxiv_id:
        raise HTTPException(400, detail="Could not parse an arXiv ID from the provided value.")

    if arxiv_id:
        key = arxiv_id
        doi = f"10.48550/arxiv.{arxiv_id}"
        url = body.url.strip() or f"https://arxiv.org/abs/{arxiv_id}"
    else:
        # deterministic key so re-submitting the same paper dedups cleanly
        digest = hashlib.sha1(
            f"{body.title.strip()}|{body.url.strip()}".encode()
        ).hexdigest()[:10]
        key = f"manual:{digest}"
        doi = ""
        url = body.url.strip()

    store = Storage(_data_dir, topic_id)
    try:
        if store.is_paper_seen(key, doi):
            raise HTTPException(409, detail="This paper is already in the library.")
        paper = {
            "arxiv_id": key,
            "paper_id": key,
            "source": "manual",
            "title": body.title.strip(),
            "authors": body.authors.strip(),
            "abstract": body.abstract.strip(),
            "url": url,
            "published": body.published.strip(),
            "summary": "",
            "key_insight": "",
            "method": "",
            "contribution": "",
            "math_concepts": [],
            "venue": body.venue.strip(),
            "cited_works": [],
            "quality_score": 0,
            "citation_count": 0,
            "doi": doi,
        }
        store.insert_arxiv(paper)
        stored = store.get_arxiv(key)
    finally:
        store.close()

    if body.summarize:
        _submit_summarize_job(topic, topic_id, key)

    return {"paper": stored, "summarizing": bool(body.summarize)}


# ---------------------------------------------------------------------------
# Batch add papers (comma/whitespace-separated arXiv IDs or URLs)
# ---------------------------------------------------------------------------

class BatchAddRequest(BaseModel):
    text: str = ""
    summarize: bool = False


def _submit_batch_summarize_job(topic: dict, topic_id: str, arxiv_ids: list[str]) -> None:
    """Background-summarize a set of freshly batch-added papers (one job)."""
    topic_cfg = cfg_module.from_topic(topic, _base_cfg)
    ids = list(arxiv_ids)

    def _run():
        store = Storage(_data_dir, topic_id)
        try:
            papers = [p for p in (store.get_arxiv(a) for a in ids) if p]
            if not papers:
                return
            summarize_papers(papers, topic_cfg)  # batches internally
            for p in papers:
                store.update_arxiv_summary(p["arxiv_id"], {
                    "summary": p.get("summary", ""),
                    "key_insight": p.get("key_insight", ""),
                    "method": p.get("method", ""),
                    "contribution": p.get("contribution", ""),
                    "math_concepts": p.get("math_concepts", []),
                    "venue": p.get("venue", ""),
                    "cited_works": p.get("cited_works", []),
                })
            log.info("Summarized %d batch-added papers (topic %s)", len(papers), topic_id)
        except Exception as e:
            log.exception("Batch summarize job failed: %s", e)
        finally:
            store.close()

    _brainstorm_executor.submit(_run)


@app.post("/api/topics/{topic_id}/papers/batch", status_code=201)
async def add_papers_batch(topic_id: str, body: BatchAddRequest) -> dict:
    """Add many arXiv papers at once from comma/whitespace-separated IDs or URLs.

    Each entry is reported individually (added | duplicate | not_found | invalid
    | error). Only arXiv IDs/URLs are supported — we can't auto-fetch metadata
    for arbitrary links. Optional ``summarize`` runs one background job over all
    adds. arXiv is queried once (``id_list``) for the whole batch.
    """
    reg = _get_registry()
    topic = reg.get_topic(topic_id)
    if not topic:
        raise HTTPException(404, detail="Topic not found")

    tokens = [t for t in re.split(r"[\s,]+", body.text.strip()) if t]
    if not tokens:
        raise HTTPException(400, detail="No URLs or arXiv IDs provided")

    # dedup tokens, preserve order
    seen_tok: set[str] = set()
    uniq = [t for t in tokens if not (t in seen_tok or seen_tok.add(t))]

    results: list[dict] = []
    added_ids: list[str] = []
    store = Storage(_data_dir, topic_id)
    try:
        pending: list[tuple[str, str]] = []  # (token, cleaned_id)
        for tok in uniq:
            cid = arxiv.extract_arxiv_id(tok)
            if not cid:
                results.append({"query": tok, "status": "invalid"})
                continue
            if store.is_paper_seen(cid, f"10.48550/arxiv.{cid}"):
                results.append({"query": tok, "status": "duplicate", "arxiv_id": cid})
                continue
            pending.append((tok, cid))

        fetched: dict[str, dict] = {}
        fetch_failed = False
        if pending:
            try:
                fetched = arxiv.fetch_many_by_id([cid for _, cid in pending])
            except httpx.HTTPError as e:
                log.warning("Batch arXiv fetch failed: %s", e)
                fetch_failed = True

        for tok, cid in pending:
            if fetch_failed:
                results.append({"query": tok, "status": "error", "arxiv_id": cid})
                continue
            meta = fetched.get(cid)
            if not meta:
                results.append({"query": tok, "status": "not_found", "arxiv_id": cid})
                continue
            if cid in added_ids:  # same id appeared twice via different URLs
                results.append({"query": tok, "status": "duplicate", "arxiv_id": cid})
                continue
            store.insert_arxiv({
                "arxiv_id": cid,
                "paper_id": cid,
                "source": "manual",
                "title": meta.get("title", ""),
                "authors": meta.get("authors", ""),
                "abstract": meta.get("abstract", ""),
                "url": meta.get("url", f"https://arxiv.org/abs/{cid}"),
                "published": (meta.get("published", "") or "")[:10],
                "summary": "",
                "key_insight": "",
                "method": "",
                "contribution": "",
                "math_concepts": [],
                "venue": "",
                "cited_works": [],
                "quality_score": 0,
                "citation_count": 0,
                "doi": meta.get("doi", f"10.48550/arxiv.{cid}"),
            })
            added_ids.append(cid)
            results.append({"query": tok, "status": "added", "arxiv_id": cid,
                            "title": meta.get("title", "")})
    finally:
        store.close()

    if body.summarize and added_ids:
        _submit_batch_summarize_job(topic, topic_id, added_ids)

    counts = {k: sum(1 for r in results if r["status"] == k)
              for k in ("added", "duplicate", "not_found", "invalid", "error")}
    return {"results": results, "counts": counts,
            "summarizing": bool(body.summarize and added_ids)}


# ---------------------------------------------------------------------------
# Delete paper endpoint
# ---------------------------------------------------------------------------

@app.delete("/api/topics/{topic_id}/papers/{arxiv_id:path}", status_code=204)
async def delete_paper(topic_id: str, arxiv_id: str) -> None:
    reg = _get_registry()
    if not reg.get_topic(topic_id):
        raise HTTPException(404, detail="Topic not found")
    store = Storage(_data_dir, topic_id)
    try:
        deleted = store.delete_arxiv(arxiv_id)
    finally:
        store.close()
    if not deleted:
        raise HTTPException(404, detail="Paper not found")


@app.get("/api/topics/{topic_id}/papers/{arxiv_id:path}")
async def get_paper(topic_id: str, arxiv_id: str) -> dict:
    reg = _get_registry()
    if not reg.get_topic(topic_id):
        raise HTTPException(404, detail="Topic not found")
    store = Storage(_data_dir, topic_id)
    try:
        paper = store.get_arxiv(arxiv_id)
    finally:
        store.close()
    if not paper:
        raise HTTPException(404, detail="Paper not found")
    return paper


@app.get("/api/topics/{topic_id}/repos")
async def list_repos(
    topic_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> dict:
    reg = _get_registry()
    if not reg.get_topic(topic_id):
        raise HTTPException(404, detail="Topic not found")
    store = Storage(_data_dir, topic_id)
    try:
        repos, total = store.get_all_github(limit=limit, offset=offset)
    finally:
        store.close()
    return {"repos": repos, "total": total, "limit": limit, "offset": offset}


# ---------------------------------------------------------------------------
# Background executor (shared by brainstorm, refilter, discovery)
# ---------------------------------------------------------------------------

from concurrent.futures import ThreadPoolExecutor as _TPE
_brainstorm_executor = _TPE(max_workers=2)

# In-memory brainstorm progress: session_id → {stage, message, ...}
_brainstorm_progress: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Discovery endpoints
# ---------------------------------------------------------------------------

class DiscoveryCreate(BaseModel):
    type: str  # "trending" | "math" | "community"
    # Math insights options
    categories: list[str] | None = None
    wildcard_categories: list[str] | None = None
    lookback_days: int | None = None
    max_recent: int | None = None
    max_historical: int | None = None
    max_wildcard: int | None = None
    sample_size: int | None = None
    # Community ideas options
    keywords: list[str] | None = None
    platforms: list[str] | None = None
    max_results_per_platform: int | None = None


@app.post("/api/discovery", status_code=202)
async def start_discovery(body: DiscoveryCreate) -> dict:
    reg = _get_registry()
    if body.type not in ("trending", "math", "community"):
        raise HTTPException(400, detail="type must be 'trending', 'math', or 'community'")

    # Check if one is already running
    reports = reg.list_discovery_reports(report_type=body.type, limit=1)
    if reports and reports[0]["status"] == "running":
        raise HTTPException(409, detail=f"A {body.type} discovery is already running")

    opts: dict = {}
    if body.type == "math":
        if body.categories is not None:
            opts["categories"] = body.categories
        if body.wildcard_categories is not None:
            opts["wildcard_categories"] = body.wildcard_categories
        if body.lookback_days is not None:
            opts["lookback_days"] = body.lookback_days
        if body.max_recent is not None:
            opts["max_recent"] = body.max_recent
        if body.max_historical is not None:
            opts["max_historical"] = body.max_historical
        if body.max_wildcard is not None:
            opts["max_wildcard"] = body.max_wildcard
        if body.sample_size is not None:
            opts["sample_size"] = body.sample_size
    elif body.type == "community":
        if body.keywords is not None:
            opts["keywords"] = body.keywords
        if body.platforms is not None:
            opts["platforms"] = body.platforms
        if body.max_results_per_platform is not None:
            opts["max_results_per_platform"] = body.max_results_per_platform

    def _run():
        try:
            if body.type == "trending":
                run_trending(reg, _base_cfg)
            elif body.type == "math":
                run_math_insights(reg, _base_cfg, **opts)
            else:
                run_community_ideas(reg, _base_cfg, **opts)
        except Exception as e:
            log.exception("Discovery %s failed: %s", body.type, e)

    _brainstorm_executor.submit(_run)
    return {"status": "started", "type": body.type}


@app.get("/api/discovery")
async def list_discovery_reports(type: str | None = Query(default=None)) -> dict:
    reg = _get_registry()
    reports = reg.list_discovery_reports(report_type=type, limit=20)
    return {"reports": reports}


@app.get("/api/discovery/latest/{report_type}")
async def get_latest_discovery(report_type: str) -> dict:
    reg = _get_registry()
    if report_type not in ("trending", "math", "community"):
        raise HTTPException(400, detail="type must be 'trending' or 'math'")
    report = reg.get_latest_discovery_report(report_type)
    if not report:
        # Check if one is running
        reports = reg.list_discovery_reports(report_type=report_type, limit=1)
        if reports and reports[0]["status"] == "running":
            return reports[0]
        raise HTTPException(404, detail=f"No {report_type} discovery report found")
    return report


@app.get("/api/discovery/{report_id}")
async def get_discovery_report(report_id: str) -> dict:
    reg = _get_registry()
    report = reg.get_discovery_report(report_id)
    if not report:
        raise HTTPException(404, detail="Discovery report not found")
    return report


@app.post("/api/discovery/{report_id}/review")
async def review_discovery(report_id: str) -> dict:
    """Run LLM-based quality review on a completed discovery report."""
    reg = _get_registry()
    report = reg.get_discovery_report(report_id)
    if not report:
        raise HTTPException(404, detail="Discovery report not found")
    if report["status"] != "completed":
        raise HTTPException(409, detail="Can only review completed reports")

    result = review_discovery_report(reg, report_id, _base_cfg)
    if "error" in result:
        raise HTTPException(400, detail=result["error"])
    return result


@app.post("/api/discovery/{report_id}/regenerate", status_code=202)
async def regenerate_discovery(report_id: str) -> dict:
    """Re-run discovery for a low-quality report. Creates a new report of the same type."""
    reg = _get_registry()
    report = reg.get_discovery_report(report_id)
    if not report:
        raise HTTPException(404, detail="Discovery report not found")

    report_type = report["type"]

    # Check no running report of same type
    reports = reg.list_discovery_reports(report_type=report_type, limit=1)
    if reports and reports[0]["status"] == "running":
        raise HTTPException(409, detail=f"A {report_type} discovery is already running")

    def _run():
        try:
            if report_type == "trending":
                run_trending(reg, _base_cfg)
            else:
                run_math_insights(reg, _base_cfg)
        except Exception as e:
            log.exception("Discovery regeneration %s failed: %s", report_type, e)

    _brainstorm_executor.submit(_run)
    return {"status": "started", "type": report_type, "replacing": report_id}


# ---------------------------------------------------------------------------
# Brainstorm endpoints
# ---------------------------------------------------------------------------


class BrainstormCreate(BaseModel):
    mode: str = "auto"  # "auto" | "user"
    user_idea: str = ""
    run_code_verification: bool = False
    context_options: dict | None = None  # Optional toggles for context sources


@app.post("/api/topics/{topic_id}/brainstorm", status_code=202)
async def start_brainstorm(topic_id: str, body: BrainstormCreate) -> dict:
    reg = _get_registry()
    topic = reg.get_topic(topic_id)
    if not topic:
        raise HTTPException(404, detail="Topic not found")

    bs = reg.create_brainstorm_session(
        topic_id,
        mode=body.mode,
        user_idea=body.user_idea,
        run_code_verification=body.run_code_verification,
    )

    topic_cfg = cfg_module.from_topic(topic, _base_cfg)

    session_id = bs["id"]

    def _on_brainstorm_progress(stage: str, detail: dict) -> None:
        _brainstorm_progress[session_id] = {"stage": stage, **detail}

    def _run():
        try:
            result = run_brainstorm(
                topic_id=topic_id,
                topic_name=topic["name"],
                data_dir=_data_dir,
                cfg=topic_cfg,
                mode=body.mode,
                user_idea=body.user_idea,
                run_code_verification=body.run_code_verification,
                registry=reg,
                on_progress=_on_brainstorm_progress,
                context_options=body.context_options,
            )
            from datetime import datetime, timezone
            import json as _json
            reg.update_brainstorm_session(topic_id, bs["id"], {
                "status": "completed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "ideas_json": result.get("ideas", []),
                "literature_result": result.get("literature_result", ""),
                "logic_result": result.get("logic_result", ""),
                "code_result": result.get("code_result", ""),
                "review_result": _json.dumps(result.get("review_history", [])),
            })
        except Exception as e:
            log.exception("Brainstorm failed: %s", e)
            from datetime import datetime, timezone
            reg.update_brainstorm_session(topic_id, bs["id"], {
                "status": "failed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
            })
        finally:
            _brainstorm_progress.pop(session_id, None)

    _brainstorm_executor.submit(_run)
    return {"status": "started", "session_id": session_id}


@app.get("/api/topics/{topic_id}/brainstorm")
async def list_brainstorm_sessions(topic_id: str) -> dict:
    reg = _get_registry()
    if not reg.get_topic(topic_id):
        raise HTTPException(404, detail="Topic not found")
    sessions = reg.list_brainstorm_sessions(topic_id)
    return {"sessions": sessions}


@app.get("/api/topics/{topic_id}/brainstorm/{session_id}")
async def get_brainstorm_session(topic_id: str, session_id: str) -> dict:
    reg = _get_registry()
    bs = reg.get_brainstorm_session(topic_id, session_id)
    if not bs:
        raise HTTPException(404, detail="Brainstorm session not found")
    return bs


@app.get("/api/topics/{topic_id}/brainstorm/{session_id}/progress")
async def get_brainstorm_progress(topic_id: str, session_id: str) -> dict:
    progress = _brainstorm_progress.get(session_id, {})
    running = session_id in _brainstorm_progress
    return {"running": running, "session_id": session_id, **progress}


class PriorArtRequest(BaseModel):
    idea_index: int


@app.post("/api/topics/{topic_id}/brainstorm/{session_id}/prior-art")
async def check_idea_prior_art(
    topic_id: str, session_id: str, body: PriorArtRequest
) -> dict:
    """Run arXiv prior-art check for a specific brainstorm idea.

    Synchronous (~15-30s). Updates the idea's prior_art field in the session.
    """
    reg = _get_registry()
    topic = reg.get_topic(topic_id)
    if not topic:
        raise HTTPException(404, detail="Topic not found")

    bs = reg.get_brainstorm_session(topic_id, session_id)
    if not bs:
        raise HTTPException(404, detail="Brainstorm session not found")
    if bs["status"] != "completed":
        raise HTTPException(409, detail="Can only check prior art on completed sessions")

    ideas = bs.get("ideas_json") or []
    if body.idea_index < 0 or body.idea_index >= len(ideas):
        raise HTTPException(400, detail=f"idea_index out of range (0-{len(ideas) - 1})")

    idea = ideas[body.idea_index]
    topic_cfg = cfg_module.from_topic(topic, _base_cfg)

    result = check_prior_art(idea, topic_cfg)

    # Persist result into ideas_json
    ideas[body.idea_index]["prior_art"] = result
    reg.update_brainstorm_session(topic_id, session_id, {"ideas_json": ideas})

    return result


# ---------------------------------------------------------------------------
# Research Plan endpoints
# ---------------------------------------------------------------------------

class ResearchPlanCreate(BaseModel):
    idea: dict[str, Any]
    brainstorm_session_id: str = ""


@app.post("/api/topics/{topic_id}/research-plan", status_code=202)
async def start_research_plan(topic_id: str, body: ResearchPlanCreate) -> dict:
    reg = _get_registry()
    topic = reg.get_topic(topic_id)
    if not topic:
        raise HTTPException(404, detail="Topic not found")

    plan = reg.create_research_plan(
        topic_id,
        idea=body.idea,
        brainstorm_session_id=body.brainstorm_session_id,
    )

    topic_cfg = cfg_module.from_topic(topic, _base_cfg)

    def _run():
        try:
            def _on_section(section: str, content: str) -> None:
                reg.update_research_plan(topic_id, plan["id"], {section: content})

            result = generate_research_plan(
                topic_id=topic_id,
                topic_name=topic["name"],
                data_dir=_data_dir,
                cfg=topic_cfg,
                idea=body.idea,
                source_brainstorm_id=body.brainstorm_session_id,
                on_section_done=_on_section,
            )
            from datetime import datetime, timezone
            import json as _json
            reg.update_research_plan(topic_id, plan["id"], {
                "status": "completed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "introduction": result.get("introduction", ""),
                "related_work": result.get("related_work", ""),
                "methodology": result.get("methodology", ""),
                "experimental_design": result.get("experimental_design", ""),
                "expected_results": result.get("expected_results", ""),
                "timeline": result.get("timeline", ""),
                "review": result.get("review", ""),
                "full_markdown": result.get("full_markdown", ""),
                "review_history": _json.dumps(result.get("review_history", [])),
            })
        except Exception as e:
            log.exception("Research plan generation failed: %s", e)
            from datetime import datetime, timezone
            reg.update_research_plan(topic_id, plan["id"], {
                "status": "failed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
            })

    _brainstorm_executor.submit(_run)
    return {"status": "started", "plan_id": plan["id"]}


@app.get("/api/topics/{topic_id}/research-plan")
async def list_research_plans(topic_id: str) -> dict:
    reg = _get_registry()
    if not reg.get_topic(topic_id):
        raise HTTPException(404, detail="Topic not found")
    plans = reg.list_research_plans(topic_id)
    return {"plans": plans}


@app.get("/api/topics/{topic_id}/research-plan/{plan_id}")
async def get_research_plan(topic_id: str, plan_id: str) -> dict:
    reg = _get_registry()
    plan = reg.get_research_plan(topic_id, plan_id)
    if not plan:
        raise HTTPException(404, detail="Research plan not found")
    return plan


@app.get("/api/topics/{topic_id}/research-plan/{plan_id}/progress")
async def get_research_plan_progress(topic_id: str, plan_id: str) -> dict:
    """Lightweight progress endpoint for monitoring generation status."""
    reg = _get_registry()
    plan = reg.get_research_plan(topic_id, plan_id)
    if not plan:
        raise HTTPException(404, detail="Research plan not found")

    sections = [
        "introduction", "related_work", "methodology",
        "experimental_design", "expected_results", "timeline", "review",
    ]
    section_status = {}
    for s in sections:
        content = plan.get(s, "")
        section_status[s] = {
            "done": bool(content),
            "chars": len(content),
        }

    done_count = sum(1 for v in section_status.values() if v["done"])
    total_chars = sum(v["chars"] for v in section_status.values())

    return {
        "plan_id": plan_id,
        "status": plan["status"],
        "started_at": plan.get("started_at"),
        "finished_at": plan.get("finished_at"),
        "idea_title": plan.get("idea_title", ""),
        "sections_done": done_count,
        "sections_total": len(sections),
        "total_chars": total_chars,
        "sections": section_status,
    }


class ResearchPlanRefine(BaseModel):
    feedback: str = ""  # optional — empty means refine based on peer review only
    sections: list[str] | None = None  # specific sections to refine, or all


@app.post("/api/topics/{topic_id}/research-plan/{plan_id}/refine", status_code=202)
async def refine_plan(topic_id: str, plan_id: str, body: ResearchPlanRefine) -> dict:
    """Refine an existing research plan in-place based on peer review / user feedback."""
    reg = _get_registry()
    topic = reg.get_topic(topic_id)
    if not topic:
        raise HTTPException(404, detail="Topic not found")

    existing = reg.get_research_plan(topic_id, plan_id)
    if not existing:
        raise HTTPException(404, detail="Research plan not found")
    if existing["status"] != "completed":
        raise HTTPException(409, detail="Can only refine completed plans")

    # Mark as running (in-place update)
    reg.update_research_plan(topic_id, plan_id, {"status": "running"})

    topic_cfg = cfg_module.from_topic(topic, _base_cfg)

    prev_history = existing.get("review_history", [])
    if isinstance(prev_history, str):
        import json as _json2
        try:
            prev_history = _json2.loads(prev_history)
        except Exception:
            prev_history = []

    def _run():
        try:
            def _on_section(section: str, content: str) -> None:
                reg.update_research_plan(topic_id, plan_id, {section: content})

            result = refine_research_plan(
                topic_id=topic_id,
                topic_name=topic["name"],
                data_dir=_data_dir,
                cfg=topic_cfg,
                existing_plan=existing,
                user_feedback=body.feedback,
                sections_to_refine=body.sections,
                on_section_done=_on_section,
            )
            from datetime import datetime, timezone
            import json as _json
            # Accumulate review history
            new_history = list(prev_history)
            new_round = len(new_history) + 1
            new_history.append({
                "round": new_round,
                "review": result.get("review", ""),
                "feedback": body.feedback or None,
            })
            reg.update_research_plan(topic_id, plan_id, {
                "status": "completed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "introduction": result.get("introduction", ""),
                "related_work": result.get("related_work", ""),
                "methodology": result.get("methodology", ""),
                "experimental_design": result.get("experimental_design", ""),
                "expected_results": result.get("expected_results", ""),
                "timeline": result.get("timeline", ""),
                "review": result.get("review", ""),
                "full_markdown": result.get("full_markdown", ""),
                "review_history": _json.dumps(new_history),
            })
        except Exception as e:
            log.exception("Research plan refinement failed: %s", e)
            # Restore to completed so user can retry (in-place refine, no new record)
            reg.update_research_plan(topic_id, plan_id, {
                "status": "completed",
            })

    _brainstorm_executor.submit(_run)
    return {
        "status": "started",
        "plan_id": plan_id,
    }


# ---------------------------------------------------------------------------
# Chat endpoints
# ---------------------------------------------------------------------------

class ChatSessionCreate(BaseModel):
    title: str = ""

class ChatMessageCreate(BaseModel):
    content: str

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content must not be empty")
        return v

# In-memory chat progress: assistant_msg_id → {status, content?}
_chat_progress: dict[str, dict] = {}


# In-memory embedding progress: topic_id → {status, embedded, total}
_embedding_progress: dict[str, dict] = {}


@app.post("/api/topics/{topic_id}/embeddings", status_code=202)
async def build_embeddings(topic_id: str) -> dict:
    """Build/update the RAG embedding index for a topic's paper library."""
    from paper_tracker.rag import ensure_embeddings

    reg = _get_registry()
    if not reg.get_topic(topic_id):
        raise HTTPException(404, detail="Topic not found")

    if topic_id in _embedding_progress and _embedding_progress[topic_id].get("status") == "running":
        return _embedding_progress[topic_id]

    _embedding_progress[topic_id] = {"status": "running", "embedded": 0, "total": 0}

    def _run():
        try:
            store = Storage(_data_dir, topic_id)
            try:
                def _on_progress(done: int, total: int) -> None:
                    _embedding_progress[topic_id] = {
                        "status": "running", "embedded": done, "total": total,
                    }

                count = ensure_embeddings(store, on_progress=_on_progress)
                existing = store.embedding_count()
                _embedding_progress[topic_id] = {
                    "status": "completed", "embedded": count, "total": existing,
                }
            finally:
                store.close()
        except Exception as e:
            log.exception("Embedding build failed: %s", e)
            _embedding_progress[topic_id] = {"status": "failed", "error": str(e)}

    _brainstorm_executor.submit(_run)
    return {"status": "started", "topic_id": topic_id}


@app.get("/api/topics/{topic_id}/embeddings")
async def get_embeddings_status(topic_id: str) -> dict:
    """Check embedding index status for a topic."""
    reg = _get_registry()
    if not reg.get_topic(topic_id):
        raise HTTPException(404, detail="Topic not found")

    store = Storage(_data_dir, topic_id)
    try:
        total_papers, _ = store.get_all_arxiv(limit=1, offset=0)
        # get_all_arxiv returns (papers, total)
        _, paper_count = store.get_all_arxiv(limit=0, offset=0)
        emb_count = store.embedding_count()
    finally:
        store.close()

    progress = _embedding_progress.get(topic_id, {})
    return {
        "topic_id": topic_id,
        "paper_count": paper_count,
        "embedding_count": emb_count,
        "indexed": emb_count >= paper_count if paper_count > 0 else False,
        "progress": progress,
    }


@app.post("/api/topics/{topic_id}/chat", status_code=201)
async def create_chat_session(topic_id: str, body: ChatSessionCreate) -> dict:
    reg = _get_registry()
    if not reg.get_topic(topic_id):
        raise HTTPException(404, detail="Topic not found")
    session = reg.create_chat_session(topic_id, title=body.title)
    return session


@app.get("/api/topics/{topic_id}/chat")
async def list_chat_sessions(topic_id: str) -> dict:
    reg = _get_registry()
    if not reg.get_topic(topic_id):
        raise HTTPException(404, detail="Topic not found")
    sessions = reg.list_chat_sessions(topic_id)
    return {"sessions": sessions}


@app.get("/api/topics/{topic_id}/chat/{session_id}")
async def get_chat_session(topic_id: str, session_id: str) -> dict:
    reg = _get_registry()
    session = reg.get_chat_session(topic_id, session_id)
    if not session:
        raise HTTPException(404, detail="Chat session not found")
    messages = reg.list_chat_messages(topic_id, session_id)
    return {**session, "messages": messages}


@app.delete("/api/topics/{topic_id}/chat/{session_id}", status_code=204)
async def delete_chat_session(topic_id: str, session_id: str) -> None:
    reg = _get_registry()
    if not reg.delete_chat_session(topic_id, session_id):
        raise HTTPException(404, detail="Chat session not found")


@app.post("/api/topics/{topic_id}/chat/{session_id}/messages", status_code=202)
async def send_chat_message(topic_id: str, session_id: str, body: ChatMessageCreate) -> dict:
    reg = _get_registry()
    topic = reg.get_topic(topic_id)
    if not topic:
        raise HTTPException(404, detail="Topic not found")
    session = reg.get_chat_session(topic_id, session_id)
    if not session:
        raise HTTPException(404, detail="Chat session not found")

    # Save user message
    user_msg = reg.add_chat_message(topic_id, session_id, "user", body.content)

    # Create assistant placeholder
    assistant_msg = reg.add_chat_message(
        topic_id, session_id, "assistant", "", status="pending"
    )

    # Auto-set title from first user message
    if not session.get("title"):
        title = body.content[:60].strip()
        with reg._lock:
            reg._conn.execute(
                "UPDATE chat_sessions SET title = ? WHERE id = ?",
                (title, session_id),
            )
            reg._conn.commit()

    topic_cfg = cfg_module.from_topic(topic, _base_cfg)
    assistant_msg_id = assistant_msg["id"]

    _chat_progress[assistant_msg_id] = {"status": "pending"}

    def _run():
        try:
            reg.update_chat_message(assistant_msg_id, {"status": "generating"})
            _chat_progress[assistant_msg_id] = {"status": "generating"}

            # Get prior messages for context
            prior = reg.list_chat_messages(topic_id, session_id)
            # Exclude the pending assistant message itself
            prior = [m for m in prior if m["id"] != assistant_msg_id]

            result = generate_chat_response(
                topic_id=topic_id,
                topic_name=topic["name"],
                data_dir=_data_dir,
                cfg=topic_cfg,
                user_message=body.content,
                prior_messages=prior,
            )
            reg.update_chat_message(assistant_msg_id, {
                "content": result["content"],
                "cited_papers": result["cited_papers"],
                "status": "completed",
            })
            _chat_progress[assistant_msg_id] = {"status": "completed"}
        except Exception as e:
            log.exception("Chat response generation failed: %s", e)
            reg.update_chat_message(assistant_msg_id, {
                "content": "An error occurred while generating the response.",
                "status": "failed",
            })
            _chat_progress[assistant_msg_id] = {"status": "failed"}

    _brainstorm_executor.submit(_run)
    return {"user_msg_id": user_msg["id"], "assistant_msg_id": assistant_msg_id}


@app.get("/api/topics/{topic_id}/chat/{session_id}/messages/{msg_id}/progress")
async def get_chat_message_progress(topic_id: str, session_id: str, msg_id: str) -> dict:
    # Check in-memory progress first
    progress = _chat_progress.get(msg_id)
    if progress:
        if progress["status"] in ("completed", "failed"):
            _chat_progress.pop(msg_id, None)
        return {"msg_id": msg_id, **progress}

    # Fallback to DB
    reg = _get_registry()
    msg = reg.get_chat_message(msg_id)
    if not msg:
        raise HTTPException(404, detail="Message not found")
    return {"msg_id": msg_id, "status": msg["status"]}


# ---------------------------------------------------------------------------
# Deep Read endpoints
# ---------------------------------------------------------------------------

class DeepReadCreate(BaseModel):
    paper_id: str
    language: str = "en"

class DeepReadMessageCreate(BaseModel):
    content: str

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content must not be empty")
        return v

class DeepReadNotesUpdate(BaseModel):
    notes: str

# In-memory deep-read progress: session_id → {sections_done: [...], current_section, status}
_deep_read_progress: dict[str, dict] = {}

# In-memory deep-read QA progress: assistant_msg_id → {status}
_deep_read_qa_progress: dict[str, dict] = {}


@app.post("/api/topics/{topic_id}/deep-read", status_code=202)
async def start_deep_read(topic_id: str, body: DeepReadCreate) -> dict:
    """Start a deep reading analysis of a paper (5-stage background task)."""
    reg = _get_registry()
    topic = reg.get_topic(topic_id)
    if not topic:
        raise HTTPException(404, detail="Topic not found")

    # Verify paper exists
    store = Storage(_data_dir, topic_id)
    try:
        paper = store.get_arxiv(body.paper_id)
    finally:
        store.close()
    if not paper:
        raise HTTPException(404, detail=f"Paper not found: {body.paper_id}")

    session = reg.create_deep_read_session(
        topic_id, body.paper_id, paper_title=paper.get("title", ""), language=body.language,
    )
    session_id = session["id"]
    topic_cfg = cfg_module.from_topic(topic, _base_cfg)

    _deep_read_progress[session_id] = {
        "status": "running",
        "sections_done": [],
        "current_section": "motivation_analysis",
    }

    def _run():
        try:
            from paper_tracker.deep_read import generate_deep_read

            section_order = [
                "motivation_analysis", "method_analysis",
                "experiment_analysis", "contribution_analysis",
                "summary_card_json", "code_review",
            ]

            def _on_section(section_name: str, content: str) -> None:
                reg.update_deep_read_session(topic_id, session_id, {section_name: content})
                done = _deep_read_progress.get(session_id, {}).get("sections_done", [])
                done.append(section_name)
                idx = section_order.index(section_name) if section_name in section_order else -1
                next_section = section_order[idx + 1] if idx + 1 < len(section_order) else None
                _deep_read_progress[session_id] = {
                    "status": "running",
                    "sections_done": done,
                    "current_section": next_section,
                }

            result = generate_deep_read(
                topic_id=topic_id,
                topic_name=topic["name"],
                data_dir=_data_dir,
                cfg=topic_cfg,
                paper_id=body.paper_id,
                on_section_done=_on_section,
                language=body.language,
            )
            from datetime import datetime, timezone
            import json as _json
            reg.update_deep_read_session(topic_id, session_id, {
                "status": "completed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "motivation_analysis": result.get("motivation_analysis", ""),
                "method_analysis": result.get("method_analysis", ""),
                "experiment_analysis": result.get("experiment_analysis", ""),
                "contribution_analysis": result.get("contribution_analysis", ""),
                "summary_card_json": _json.dumps(result.get("summary_card_json", {})),
                "code_review": result.get("code_review", ""),
                "repo_urls": _json.dumps(result.get("repo_urls", [])),
                "repo_local_paths": _json.dumps(result.get("repo_local_paths", [])),
            })
            _deep_read_progress[session_id] = {
                "status": "completed",
                "sections_done": [
                    "motivation_analysis", "method_analysis",
                    "experiment_analysis", "contribution_analysis",
                    "summary_card_json", "code_review",
                ],
                "current_section": None,
            }
        except Exception as e:
            log.exception("Deep read generation failed: %s", e)
            from datetime import datetime, timezone
            reg.update_deep_read_session(topic_id, session_id, {
                "status": "failed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
            })
            _deep_read_progress[session_id] = {"status": "failed", "sections_done": [], "current_section": None}

    _brainstorm_executor.submit(_run)
    return {"status": "started", "session_id": session_id}


@app.get("/api/topics/{topic_id}/deep-read")
async def list_deep_read_sessions(
    topic_id: str,
    paper_id: str = Query(default=""),
) -> dict:
    reg = _get_registry()
    if not reg.get_topic(topic_id):
        raise HTTPException(404, detail="Topic not found")
    sessions = reg.list_deep_read_sessions(topic_id, paper_id=paper_id or None)
    return {"sessions": sessions}


@app.get("/api/topics/{topic_id}/deep-read/{session_id}")
async def get_deep_read_session(topic_id: str, session_id: str) -> dict:
    reg = _get_registry()
    session = reg.get_deep_read_session(topic_id, session_id)
    if not session:
        raise HTTPException(404, detail="Deep read session not found")
    messages = reg.list_deep_read_messages(topic_id, session_id)
    return {**session, "messages": messages}


@app.get("/api/topics/{topic_id}/deep-read/{session_id}/progress")
async def get_deep_read_progress(topic_id: str, session_id: str) -> dict:
    progress = _deep_read_progress.get(session_id)
    if progress:
        if progress["status"] in ("completed", "failed"):
            _deep_read_progress.pop(session_id, None)
        return {"session_id": session_id, **progress}
    # Fallback to DB
    reg = _get_registry()
    session = reg.get_deep_read_session(topic_id, session_id)
    if not session:
        raise HTTPException(404, detail="Deep read session not found")
    # Derive sections_done from non-empty fields
    done = []
    for s in ["motivation_analysis", "method_analysis", "experiment_analysis", "contribution_analysis", "code_review"]:
        if session.get(s):
            done.append(s)
    card = session.get("summary_card_json")
    if card and card != {} and card != "{}":
        done.append("summary_card_json")
    return {
        "session_id": session_id,
        "status": session["status"],
        "sections_done": done,
        "current_section": None,
    }


@app.delete("/api/topics/{topic_id}/deep-read/{session_id}", status_code=204)
async def delete_deep_read_session(topic_id: str, session_id: str) -> None:
    reg = _get_registry()
    if not reg.delete_deep_read_session(topic_id, session_id):
        raise HTTPException(404, detail="Deep read session not found")


@app.put("/api/topics/{topic_id}/deep-read/{session_id}/notes")
async def update_deep_read_notes(topic_id: str, session_id: str, body: DeepReadNotesUpdate) -> dict:
    reg = _get_registry()
    session = reg.get_deep_read_session(topic_id, session_id)
    if not session:
        raise HTTPException(404, detail="Deep read session not found")
    reg.update_deep_read_session(topic_id, session_id, {"user_notes": body.notes})
    return {"status": "saved"}


@app.post("/api/topics/{topic_id}/deep-read/{session_id}/messages", status_code=202)
async def send_deep_read_message(topic_id: str, session_id: str, body: DeepReadMessageCreate) -> dict:
    reg = _get_registry()
    topic = reg.get_topic(topic_id)
    if not topic:
        raise HTTPException(404, detail="Topic not found")
    session = reg.get_deep_read_session(topic_id, session_id)
    if not session:
        raise HTTPException(404, detail="Deep read session not found")
    if session["status"] != "completed":
        raise HTTPException(409, detail="Can only ask questions on completed deep reads")

    # Save user message
    user_msg = reg.add_deep_read_message(topic_id, session_id, "user", body.content)

    # Create assistant placeholder
    assistant_msg = reg.add_deep_read_message(
        topic_id, session_id, "assistant", "", status="pending",
    )
    assistant_msg_id = assistant_msg["id"]
    _deep_read_qa_progress[assistant_msg_id] = {"status": "pending"}

    topic_cfg = cfg_module.from_topic(topic, _base_cfg)

    def _run():
        try:
            from paper_tracker.deep_read import generate_deep_read_qa

            reg.update_deep_read_message(assistant_msg_id, {"status": "generating"})
            _deep_read_qa_progress[assistant_msg_id] = {"status": "generating"}

            # Load paper
            store = Storage(_data_dir, topic_id)
            try:
                paper = store.get_arxiv(session["paper_id"])
            finally:
                store.close()

            prior = reg.list_deep_read_messages(topic_id, session_id)
            prior = [m for m in prior if m["id"] != assistant_msg_id]

            existing_analysis = {
                "motivation_analysis": session.get("motivation_analysis", ""),
                "method_analysis": session.get("method_analysis", ""),
                "experiment_analysis": session.get("experiment_analysis", ""),
                "contribution_analysis": session.get("contribution_analysis", ""),
                "code_review": session.get("code_review", ""),
            }

            result = generate_deep_read_qa(
                paper=paper or {},
                existing_analysis=existing_analysis,
                prior_messages=prior,
                user_question=body.content,
                cfg=topic_cfg,
                language=session.get("language", "en"),
            )
            reg.update_deep_read_message(assistant_msg_id, {
                "content": result["content"],
                "status": "completed",
            })
            _deep_read_qa_progress[assistant_msg_id] = {"status": "completed"}
        except Exception as e:
            log.exception("Deep read QA failed: %s", e)
            reg.update_deep_read_message(assistant_msg_id, {
                "content": "An error occurred while generating the response.",
                "status": "failed",
            })
            _deep_read_qa_progress[assistant_msg_id] = {"status": "failed"}

    _brainstorm_executor.submit(_run)
    return {"user_msg_id": user_msg["id"], "assistant_msg_id": assistant_msg_id}


@app.get("/api/topics/{topic_id}/deep-read/{session_id}/messages/{msg_id}/progress")
async def get_deep_read_message_progress(topic_id: str, session_id: str, msg_id: str) -> dict:
    progress = _deep_read_qa_progress.get(msg_id)
    if progress:
        if progress["status"] in ("completed", "failed"):
            _deep_read_qa_progress.pop(msg_id, None)
        return {"msg_id": msg_id, **progress}
    reg = _get_registry()
    msg = reg.get_deep_read_message(msg_id)
    if not msg:
        raise HTTPException(404, detail="Message not found")
    return {"msg_id": msg_id, "status": msg["status"]}


# ---------------------------------------------------------------------------
# Translation endpoints
# ---------------------------------------------------------------------------

class TranslateRequest(BaseModel):
    source_type: str
    source_id: str
    field: str
    content: str
    language: str = "zh"


@app.post("/api/translate")
async def translate_content(body: TranslateRequest) -> dict:
    """Translate content and cache the result."""
    reg = _get_registry()

    # Check cache first
    cached = reg.get_translation(body.source_type, body.source_id, body.field, body.language)
    if cached:
        return {"translated": cached}

    # Translate via LLM
    from paper_tracker.llm import call_cli

    prompt = f"""Translate the following text into Chinese (简体中文).

Rules:
- Preserve all markdown formatting, LaTeX formulas, code blocks, and reference links exactly as-is
- Do NOT translate proper nouns, model names, dataset names, metric names, method names, or paper titles
- Translate naturally and fluently, not word-by-word
- Output ONLY the translated text, no explanations or preamble

Text to translate:
{body.content}"""

    translated = call_cli(prompt, _base_cfg, model="sonnet", timeout=300)
    if not translated:
        raise HTTPException(500, detail="Translation failed")

    reg.save_translation(body.source_type, body.source_id, body.field, body.language, translated)
    return {"translated": translated}


@app.get("/api/translate")
async def get_translation(
    source_type: str = Query(...),
    source_id: str = Query(...),
    field: str = Query(...),
    language: str = Query(default="zh"),
) -> dict:
    """Look up a cached translation."""
    reg = _get_registry()
    cached = reg.get_translation(source_type, source_id, field, language)
    if cached is None:
        raise HTTPException(404, detail="Translation not found")
    return {"translated": cached}


# ---------------------------------------------------------------------------
# Usage / Billing endpoints
# ---------------------------------------------------------------------------

@app.get("/api/usage")
async def get_usage(
    service: str = Query(default="", description="Filter by service: claude, codex, copilot"),
) -> dict:
    """Return CLI tool usage data from Claude Code, Codex, and GitHub Copilot."""
    from paper_tracker.usage import get_all_usage, get_claude_usage, get_codex_usage, get_copilot_usage

    if service:
        fetchers = {"claude": get_claude_usage, "codex": get_codex_usage, "copilot": get_copilot_usage}
        fetcher = fetchers.get(service)
        if not fetcher:
            raise HTTPException(400, detail=f"Unknown service: {service}")
        return {"services": [fetcher()]}

    return {"services": get_all_usage()}
