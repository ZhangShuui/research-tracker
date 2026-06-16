"""Load config: from config.toml (legacy) or a topic dict from the registry."""

from __future__ import annotations

import tomllib
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load(path: Path | None = None) -> dict:
    """Return parsed config dict from config.toml. *path* defaults to <project_root>/config.toml."""
    if path is None:
        path = _PROJECT_ROOT / "config.toml"
    with open(path, "rb") as f:
        cfg = tomllib.load(f)

    # Resolve relative paths against project root
    paths = cfg.get("paths", {})
    for key in ("reports_dir", "data_dir", "logs_dir"):
        raw = paths.get(key, key.replace("_dir", ""))
        p = Path(raw)
        if not p.is_absolute():
            p = _PROJECT_ROOT / p
        paths[key] = str(p)
    cfg["paths"] = paths
    # Fill superpod defaults for any unset keys (host / project_root)
    cfg["superpod"] = {**_default_superpod(), **cfg.get("superpod", {})}
    return cfg


def from_topic(topic: dict, base_cfg: dict | None = None) -> dict:
    """Build a pipeline cfg dict from a registry topic dict.

    *base_cfg* provides summarizer/notify/paths defaults (from load()).
    If not supplied, defaults are used.
    """
    if base_cfg is None:
        try:
            base_cfg = load()
        except FileNotFoundError:
            base_cfg = _default_base_cfg()

    data_dir = base_cfg.get("paths", {}).get("data_dir", str(_PROJECT_ROOT / "data"))
    logs_dir = base_cfg.get("paths", {}).get("logs_dir", str(_PROJECT_ROOT / "logs"))

    return {
        "search": {
            "description": topic.get("description", ""),
            "arxiv_keywords": topic.get("arxiv_keywords", []),
            "arxiv_categories": topic.get("arxiv_categories", []),
            "arxiv_lookback_days": topic.get("arxiv_lookback_days", 2),
            "github_keywords": topic.get("github_keywords", []),
            "github_lookback_days": topic.get("github_lookback_days", 7),
            # OpenAlex
            "openalex_enabled": topic.get("openalex_enabled", False),
            "openalex_keywords": topic.get("openalex_keywords", []),
            "openalex_lookback_days": topic.get("openalex_lookback_days", 7),
            "openalex_venues": topic.get("openalex_venues", []),
            "openalex_max_results": topic.get("openalex_max_results", 200),
            # OpenReview
            "openreview_enabled": topic.get("openreview_enabled", False),
            "openreview_venues": topic.get("openreview_venues", []),
            "openreview_keywords": topic.get("openreview_keywords", []),
            "openreview_max_results": topic.get("openreview_max_results", 100),
            # Global date range override (overrides lookback_days when set)
            "search_date_from": topic.get("search_date_from", ""),
            "search_date_to": topic.get("search_date_to", ""),
            # Early relevance prefilter
            "prefilter_enabled": topic.get("prefilter_enabled", True),
            "prefilter_criteria": topic.get("prefilter_criteria", ""),
        },
        "summarizer": {**_default_summarizer(), **base_cfg.get("summarizer", {})},
        "notify": base_cfg.get("notify", {"toast": {"enabled": False}, "email": {"enabled": False}}),
        "paths": {
            "data_dir": data_dir,
            "logs_dir": logs_dir,
        },
    }


def _default_summarizer() -> dict:
    return {
        "claude_path": "claude",
        "claude_timeout": 600,
        "claude_model": "sonnet",
        "codex_path": "codex",
        "codex_timeout": 600,
        "copilot_path": "copilot",
        "copilot_timeout": 90,  # hard-capped at 90s in llm.call_copilot
        "copilot_model": "gemini-3.1-pro-preview",
        "truncation_length": 300,
    }


def _default_superpod() -> dict:
    return {
        "host": "superpod",                     # ~/.ssh/config alias for the cluster
        "project_root": "/project/visworld01",  # where reproduction work dirs are created
    }


def _default_base_cfg() -> dict:
    return {
        "paths": {
            "data_dir": str(_PROJECT_ROOT / "data"),
            "logs_dir": str(_PROJECT_ROOT / "logs"),
            "reports_dir": str(_PROJECT_ROOT / "reports"),
        },
        "summarizer": _default_summarizer(),
        "superpod": _default_superpod(),
        "notify": {"toast": {"enabled": False}, "email": {"enabled": False}},
    }
