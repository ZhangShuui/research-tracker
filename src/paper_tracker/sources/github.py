"""Search GitHub REST API for relevant repositories."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import httpx

log = logging.getLogger(__name__)

_GITHUB_SEARCH_API = "https://api.github.com/search/repositories"


def search(cfg: dict) -> list[dict]:
    """Return list of repo dicts from GitHub matching configured keywords."""
    search_cfg = cfg["search"]
    keywords = search_cfg["github_keywords"]
    lookback = search_cfg["github_lookback_days"]
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback)
    cutoff_str = cutoff.strftime("%Y-%m-%d")

    repos: list[dict] = []

    for kw in keywords:
        query = f"{kw} pushed:>={cutoff_str}"
        params = {
            "q": query,
            "sort": "updated",
            "order": "desc",
            "per_page": 30,
        }
        log.info("GitHub query: %s", query)

        try:
            resp = httpx.get(
                _GITHUB_SEARCH_API,
                params=params,
                headers={"Accept": "application/vnd.github.v3+json"},
                timeout=30,
            )
            resp.raise_for_status()
        except httpx.HTTPError as e:
            log.error("GitHub API request failed for '%s': %s", kw, e)
            continue

        data = resp.json()
        for item in data.get("items", []):
            full_name = item["full_name"]
            # Avoid duplicates across keyword queries within same run
            if any(r["repo_full_name"] == full_name for r in repos):
                continue
            repos.append({
                "repo_full_name": full_name,
                "description": (item.get("description") or "")[:500],
                "url": item["html_url"],
                "stars": item.get("stargazers_count", 0),
                "pushed_at": item.get("pushed_at", ""),
                "summary": "",  # filled later
            })

    log.info("GitHub returned %d repos after dedup", len(repos))
    return repos
