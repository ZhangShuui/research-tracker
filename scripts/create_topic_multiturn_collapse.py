"""Create the 'LLM Multi-turn Response Collapse' topic via the paper-tracker API.

Usage:
    uv run python scripts/create_topic_multiturn_collapse.py [--api http://localhost:8000]

Requires the server to be running. Idempotent: if the topic already exists
(same slug), the script prints the existing topic and exits without re-creating.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


TOPIC_NAME = "LLM Multi-turn Response Collapse"
TOPIC_SLUG = "llm-multi-turn-response-collapse"

ARXIV_KEYWORDS = [
    "mode collapse",
    "persona drift",
    "sycophancy",
    "response diversity",
    "multi-turn dialogue",
    "dialogue degradation",
    "output homogeneity",
]

OPENREVIEW_VENUES = [
    "iclr2025", "iclr2024",
    "neurips2025", "neurips2024",
    "acl2024", "acl2025",
    "emnlp2024", "emnlp2025",
]

PREFILTER_CRITERIA = (
    "The paper must be about Large Language Models, LLM-based dialogue systems, "
    "or conversational AI. Papers about GANs, image generation, VAEs, or "
    "non-language mode collapse should be excluded even if they mention "
    '"mode collapse" or "diversity". Papers that use LLMs only as a tool '
    "(e.g. as an evaluator) are OUT unless the paper's core contribution is "
    "about LLM output behavior."
)

TOPIC_BODY = {
    "name": TOPIC_NAME,
    "description": (
        "Investigates how LLM responses degrade in multi-turn conversations — "
        "response template convergence, persona drift, diversity loss, and "
        "sycophancy-induced mode collapse."
    ),
    "arxiv_keywords": ARXIV_KEYWORDS,
    "arxiv_categories": ["cs.CL", "cs.AI", "cs.LG"],
    "arxiv_lookback_days": 730,
    "github_keywords": [
        "LLM diversity evaluation",
        "multi-turn benchmark",
        "dialogue evaluation",
    ],
    "github_lookback_days": 730,
    "schedule_cron": "",
    "enabled": True,
    "openalex_enabled": True,
    "openalex_keywords": ARXIV_KEYWORDS,
    "openalex_lookback_days": 730,
    "openalex_venues": [],
    "openalex_max_results": 200,
    "openreview_enabled": True,
    "openreview_venues": OPENREVIEW_VENUES,
    "openreview_keywords": [
        "mode collapse",
        "response diversity",
        "persona drift",
        "sycophancy",
    ],
    "openreview_max_results": 100,
    "search_date_from": "2024-04-21",
    "search_date_to": "2026-04-21",
    "prefilter_enabled": True,
    "prefilter_criteria": PREFILTER_CRITERIA,
}


def _get_json(url: str) -> object:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def _post_json(url: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://localhost:8000",
                        help="Base URL of the paper-tracker API")
    args = parser.parse_args()

    base = args.api.rstrip("/")

    # Idempotency check
    try:
        existing = _get_json(f"{base}/api/topics/{TOPIC_SLUG}")
        if existing:
            print(f"Topic already exists: {existing['id']} ({existing['name']})")
            print("No action taken. Delete it first if you want to recreate.")
            return 0
    except urllib.error.HTTPError as e:
        if e.code != 404:
            print(f"Unexpected error checking topic: {e}", file=sys.stderr)
            return 2

    # Create
    try:
        created = _post_json(f"{base}/api/topics", TOPIC_BODY)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        print(f"Create failed ({e.code}): {detail}", file=sys.stderr)
        return 1

    print(f"Created topic: {created['id']}")
    print(f"  name:               {created['name']}")
    print(f"  arxiv_keywords:     {created['arxiv_keywords']}")
    print(f"  openalex_enabled:   {created['openalex_enabled']}")
    print(f"  openreview_venues:  {created['openreview_venues']}")
    print(f"  prefilter_enabled:  {created['prefilter_enabled']}")
    print(f"  search_date_range:  {created['search_date_from']} → {created['search_date_to']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
