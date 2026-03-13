"""Main entry point: orchestrate the full paper-tracker pipeline."""

from __future__ import annotations

import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from paper_tracker import config
from paper_tracker.sources import arxiv, github
from paper_tracker.sources import openalex, openreview_api
from paper_tracker.storage import Storage
from paper_tracker import summarizer, report, insights


def _setup_logging(logs_dir: str) -> None:
    log_dir = Path(logs_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_file = log_dir / f"run-{today}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def run_pipeline(
    topic_cfg: dict,
    session_id: str,
    topic_id: str,
    topic_name: str,
    data_dir: str,
    session_dir: str | Path,
    on_progress: "callable | None" = None,
) -> dict:
    """Run the full pipeline for a single topic/session.

    Returns a dict with keys: paper_count, repo_count, report_path, insights_path, status.

    on_progress(stage, detail_dict) is called at each stage transition.
    """
    log = logging.getLogger("paper_tracker")
    log.info("=== Pipeline started: topic=%s session=%s ===", topic_id, session_id)

    def _progress(stage: str, **detail: object) -> None:
        if on_progress:
            on_progress(stage, detail)

    _progress("fetching", message="Searching sources...")

    store = Storage(data_dir, topic_id)
    search_cfg = topic_cfg.get("search", {})
    try:
        # 1. Fetch from sources in parallel
        source_names = ["arxiv", "github"]
        if search_cfg.get("openalex_enabled"):
            source_names.append("openalex")
        if search_cfg.get("openreview_enabled"):
            source_names.append("openreview")

        _progress("fetching", message=f"Fetching from {', '.join(source_names)}...",
                  sources_total=len(source_names), sources_done=0)

        futures: dict[str, object] = {}
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures["arxiv"] = pool.submit(arxiv.search, topic_cfg)
            futures["github"] = pool.submit(github.search, topic_cfg)
            if "openalex" in source_names:
                futures["openalex"] = pool.submit(openalex.search, topic_cfg)
            if "openreview" in source_names:
                futures["openreview"] = pool.submit(openreview_api.search, topic_cfg)

            raw_papers: list[dict] = []
            raw_repos: list[dict] = []
            sources_done = 0

            # Collect arXiv papers
            try:
                raw_papers.extend(futures["arxiv"].result())
                sources_done += 1
                _progress("fetching", message=f"arXiv: {len(raw_papers)} papers",
                          sources_total=len(source_names), sources_done=sources_done,
                          papers_fetched=len(raw_papers))
            except Exception as e:
                log.error("arXiv source failed: %s", e)
                sources_done += 1

            # Collect GitHub repos
            try:
                raw_repos = futures["github"].result()
                sources_done += 1
                _progress("fetching", message=f"GitHub: {len(raw_repos)} repos",
                          sources_total=len(source_names), sources_done=sources_done,
                          papers_fetched=len(raw_papers), repos_fetched=len(raw_repos))
            except Exception as e:
                log.error("GitHub source failed: %s", e)
                sources_done += 1

            # Collect OpenAlex papers
            if "openalex" in futures:
                try:
                    oa_papers = futures["openalex"].result()
                    raw_papers.extend(oa_papers)
                    sources_done += 1
                    _progress("fetching", message=f"OpenAlex: +{len(oa_papers)} papers",
                              sources_total=len(source_names), sources_done=sources_done,
                              papers_fetched=len(raw_papers))
                except Exception as e:
                    log.error("OpenAlex source failed: %s", e)
                    sources_done += 1

            # Collect OpenReview papers
            if "openreview" in futures:
                try:
                    or_papers = futures["openreview"].result()
                    raw_papers.extend(or_papers)
                    sources_done += 1
                    _progress("fetching", message=f"OpenReview: +{len(or_papers)} papers",
                              sources_total=len(source_names), sources_done=sources_done,
                              papers_fetched=len(raw_papers))
                except Exception as e:
                    log.error("OpenReview source failed: %s", e)
                    sources_done += 1

        log.info("Fetched %d total papers, %d GitHub repos", len(raw_papers), len(raw_repos))

        # 2. Cross-source dedup + DB dedup
        _progress("deduplicating", message=f"Deduplicating {len(raw_papers)} papers...",
                  papers_fetched=len(raw_papers))

        seen_in_batch: set[str] = set()
        new_papers: list[dict] = []
        for p in raw_papers:
            pid = p.get("paper_id", p["arxiv_id"])
            if pid in seen_in_batch:
                continue
            seen_in_batch.add(pid)
            if store.is_paper_seen(pid):
                continue
            # Also check raw arxiv_id for arXiv-origin papers
            if p.get("arxiv_id") != pid and store.is_arxiv_seen(p["arxiv_id"]):
                continue
            new_papers.append(p)
        new_repos = [r for r in raw_repos if not store.is_github_seen(r["repo_full_name"])]

        log.info("After dedup: %d new papers, %d new repos", len(new_papers), len(new_repos))

        result: dict = {
            "paper_count": len(new_papers),
            "repo_count": len(new_repos),
            "report_path": "",
            "insights_path": "",
            "status": "completed",
        }

        if not new_papers and not new_repos:
            log.info("No new papers/repos found.")
            _progress("completed", message="No new papers or repos found.",
                      papers_new=0, repos_new=0)
            return result

        # 3. Summarize
        _progress("summarizing", message=f"Summarizing {len(new_papers)} papers...",
                  papers_total=len(new_papers), papers_done=0)
        summarizer.summarize_papers(new_papers, topic_cfg)
        _progress("summarizing", message=f"Summarized {len(new_papers)} papers, filtering...",
                  papers_total=len(new_papers), papers_done=len(new_papers))
        summarizer.summarize_repos(new_repos, topic_cfg)

        # 3.5. Quality filter — remove low-quality / irrelevant papers
        _progress("filtering", message=f"Quality filtering {len(new_papers)} papers...")
        new_papers = summarizer.filter_papers_by_quality(
            new_papers, topic_cfg, topic_name, min_quality=3
        )
        result["paper_count"] = len(new_papers)

        if not new_papers and not new_repos:
            log.info("All papers filtered out and no new repos.")
            _progress("completed", message="All papers filtered out.",
                      papers_new=0, repos_new=len(new_repos))
            return result

        # 4. Persist to DB
        _progress("saving", message=f"Saving {len(new_papers)} papers, {len(new_repos)} repos...",
                  papers_new=len(new_papers), repos_new=len(new_repos))
        for p in new_papers:
            store.insert_arxiv(p)
        for r in new_repos:
            store.insert_github(r)

        # 5. Generate session report (three-section format)
        _progress("report", message="Generating report...",
                  papers_new=len(new_papers), repos_new=len(new_repos))
        report_path = report.generate(
            new_papers, new_repos, session_dir,
            topic_name=topic_name, cfg=topic_cfg,
        )
        if report_path:
            result["report_path"] = str(report_path)

        # 6. Generate cross-paper insights
        _progress("insights", message="Generating insights...",
                  papers_new=len(new_papers), repos_new=len(new_repos))
        insights_path = insights.generate(new_papers, topic_name, session_dir, topic_cfg)
        if insights_path:
            result["insights_path"] = str(insights_path)

    finally:
        store.close()

    _progress("completed", message=f"Done: {result['paper_count']} papers, {result['repo_count']} repos",
              papers_new=result["paper_count"], repos_new=result["repo_count"])
    log.info("=== Pipeline finished: topic=%s session=%s ===", topic_id, session_id)
    return result


def main() -> None:
    """Legacy CLI entry point — uses config.toml."""
    from paper_tracker.notifiers import toast, email

    cfg = config.load()
    _setup_logging(cfg["paths"]["logs_dir"])
    log = logging.getLogger("paper_tracker")
    log.info("=== Paper Tracker run started (legacy CLI) ===")

    data_dir = cfg["paths"]["data_dir"]
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    session_dir = Path(data_dir) / "legacy" / f"sessions/{today}_001"

    result = run_pipeline(
        topic_cfg=cfg,
        session_id=f"{today}_001",
        topic_id="legacy",
        topic_name="Paper Tracker (legacy)",
        data_dir=data_dir,
        session_dir=session_dir,
    )

    # Legacy notifications
    store = Storage(data_dir, "legacy")
    try:
        new_papers = store.get_unnotified_arxiv()
        new_repos = store.get_unnotified_github()

        if new_papers or new_repos:
            toast_title = f"Paper Tracker: {result['paper_count']} papers, {result['repo_count']} repos"
            toast_body = "New findings available. Check the report!"
            toast_ok = toast.notify(toast_title, toast_body, cfg)
            email_ok = email.notify(
                Path(result["report_path"]) if result["report_path"] else None,
                new_papers, new_repos, cfg,
            )

            if toast_ok or email_ok:
                for p in new_papers:
                    store.mark_arxiv_notified(p["arxiv_id"])
                for r in new_repos:
                    store.mark_github_notified(r["repo_full_name"])
    finally:
        store.close()

    log.info("=== Paper Tracker run finished ===")


if __name__ == "__main__":
    main()
