"""FastAPI REST API for the paper-tracker web service."""

from __future__ import annotations

import logging
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from paper_tracker import config as cfg_module
from paper_tracker.brainstorm import run_brainstorm, check_prior_art
from paper_tracker.discovery import run_trending, run_math_insights, run_community_ideas, review_discovery_report
from paper_tracker.research_plan import generate_research_plan, refine_research_plan
from paper_tracker.registry import Registry
from paper_tracker.scheduler import Scheduler
from paper_tracker.storage import Storage
from paper_tracker.summarizer import refilter_papers

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
- "arxiv_keywords": A list of 3-6 search keyword phrases for finding relevant papers on arXiv (each 2-4 words, covering key aspects)
- "arxiv_categories": A list of relevant arXiv categories (e.g. "cs.CV", "cs.AI", "cs.LG", "cs.CL", "cs.RO", "stat.ML")
- "github_keywords": A list of 2-4 keyword phrases for finding related GitHub repos

Output ONLY the JSON object, no markdown fences, no explanation."""

    cfg = {"summarizer": {"claude_path": "claude", "claude_model": "sonnet", "claude_timeout": 60}}
    raw = call_cli(prompt, cfg, model="sonnet", timeout=60)
    if raw:
        # Strip markdown fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```\w*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
        try:
            return _json.loads(raw)
        except _json.JSONDecodeError:
            log.warning("Failed to parse LLM config output: %s", raw[:300])
    return {}


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

    topic = reg.create_topic({
        "id": topic_id,
        "name": body.name,
        "description": generated.get("description", ""),
        "arxiv_keywords": generated.get("arxiv_keywords", [body.name]),
        "arxiv_categories": generated.get("arxiv_categories", ["cs.CV", "cs.AI", "cs.LG"]),
        "arxiv_lookback_days": 365,
        "github_keywords": generated.get("github_keywords", [body.name]),
        "github_lookback_days": 365,
        "schedule_cron": "",
        "enabled": True,
        "search_date_from": date_from,
        "search_date_to": date_to,
    })
    return topic


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

    translated = call_cli(prompt, _base_cfg, model="sonnet", timeout=120)
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
