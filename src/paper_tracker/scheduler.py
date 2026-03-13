"""APScheduler-based job manager for topic pipelines."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

if TYPE_CHECKING:
    from paper_tracker.registry import Registry

log = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=4)


class Scheduler:
    def __init__(self, registry: "Registry", data_dir: str, base_cfg: dict):
        self._registry = registry
        self._data_dir = data_dir
        self._base_cfg = base_cfg
        self._scheduler = BackgroundScheduler()
        self._running: dict[str, Future] = {}
        self._lock = threading.Lock()
        # Pipeline progress: topic_id → {stage, message, ...}
        self.progress: dict[str, dict] = {}

    def start(self) -> None:
        self._scheduler.start()
        # Load all enabled topics with a cron schedule
        for topic in self._registry.list_topics():
            if topic.get("enabled") and topic.get("schedule_cron"):
                self._register_cron(topic)
        log.info("Scheduler started")

    def stop(self) -> None:
        self._scheduler.shutdown(wait=False)
        log.info("Scheduler stopped")

    def _register_cron(self, topic: dict) -> None:
        cron = topic["schedule_cron"]
        topic_id = topic["id"]
        try:
            trigger = CronTrigger.from_crontab(cron)
            job_id = f"cron_{topic_id}"
            self._scheduler.add_job(
                self._run_topic,
                trigger=trigger,
                args=[topic_id],
                id=job_id,
                replace_existing=True,
            )
            log.info("Registered cron job '%s' for topic %s", cron, topic_id)
        except Exception as e:
            log.error("Failed to register cron for topic %s: %s", topic_id, e)

    def trigger_now(self, topic_id: str) -> bool:
        """Trigger an immediate run. Returns False if already running."""
        with self._lock:
            if topic_id in self._running and not self._running[topic_id].done():
                return False
            future = _executor.submit(self._run_topic, topic_id)
            self._running[topic_id] = future
            return True

    def is_running(self, topic_id: str) -> bool:
        with self._lock:
            f = self._running.get(topic_id)
            return f is not None and not f.done()

    def cancel(self, topic_id: str) -> bool:
        """Attempt to cancel a running job. Returns True if cancelled."""
        with self._lock:
            f = self._running.get(topic_id)
            if f and not f.done():
                cancelled = f.cancel()
                return cancelled
        return False

    def update_schedule(self, topic_id: str, cron: str) -> None:
        job_id = f"cron_{topic_id}"
        if cron:
            try:
                trigger = CronTrigger.from_crontab(cron)
                self._scheduler.add_job(
                    self._run_topic,
                    trigger=trigger,
                    args=[topic_id],
                    id=job_id,
                    replace_existing=True,
                )
                log.info("Updated cron '%s' for topic %s", cron, topic_id)
            except Exception as e:
                log.error("Invalid cron '%s' for topic %s: %s", cron, topic_id, e)
        else:
            self._remove_cron(topic_id)

    def _remove_cron(self, topic_id: str) -> None:
        job_id = f"cron_{topic_id}"
        if self._scheduler.get_job(job_id):
            self._scheduler.remove_job(job_id)
            log.info("Removed cron job for topic %s", topic_id)

    def _run_topic(self, topic_id: str) -> None:
        from paper_tracker import config as cfg_module
        from paper_tracker.main import run_pipeline

        topic = self._registry.get_topic(topic_id)
        if not topic:
            log.error("Topic %s not found", topic_id)
            return

        session = self._registry.create_session(topic_id)
        session_id = session["id"]

        session_dir = (
            Path(self._data_dir)
            / topic_id
            / "sessions"
            / session_id
        )

        topic_cfg = cfg_module.from_topic(topic, self._base_cfg)

        def _on_progress(stage: str, detail: dict) -> None:
            self.progress[topic_id] = {"stage": stage, **detail}

        try:
            _on_progress("starting", {"message": "Initializing pipeline..."})
            result = run_pipeline(
                topic_cfg=topic_cfg,
                session_id=session_id,
                topic_id=topic_id,
                topic_name=topic["name"],
                data_dir=self._data_dir,
                session_dir=session_dir,
                on_progress=_on_progress,
            )
            self._registry.update_session(topic_id, session_id, {
                "status": "completed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "paper_count": result["paper_count"],
                "repo_count": result["repo_count"],
                "report_path": result["report_path"],
                "insights_path": result["insights_path"],
            })
            self.progress.pop(topic_id, None)
        except Exception as e:
            log.exception("Pipeline failed for topic %s session %s: %s", topic_id, session_id, e)
            self._registry.update_session(topic_id, session_id, {
                "status": "failed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
            })
            self.progress.pop(topic_id, None)
