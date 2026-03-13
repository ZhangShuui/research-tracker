"""Generate cross-paper insights via shared LLM utility.

Produces a structured research insights report highlighting trends,
methodological connections, research gaps, and recommended reading.
"""

from __future__ import annotations

import logging
from pathlib import Path

from paper_tracker.llm import call_cli

log = logging.getLogger(__name__)

_INSIGHTS_PROMPT = """\
You are a senior research scientist writing a cross-paper insights memo for your \
lab group in the field of "{topic_name}".

You have {n} new papers. For each paper you have: title, key insight, method, \
contribution, and mathematical concepts.

{paper_summaries}

Write a structured insights report in Markdown with these sections:

## Key Trends
Identify 3-5 significant trends visible across these papers. For each trend, \
explain: (a) what the trend is, (b) which papers exemplify it, and (c) why it \
matters for the field. Be specific — name concrete techniques and metrics, not \
vague generalizations.

## Emerging Methods
Highlight 2-4 novel techniques or architectural patterns that appear in these papers. \
For each method: describe what problem it solves, how it works (briefly), and assess \
its potential impact. Note if multiple papers independently converge on similar approaches \
— that's a strong signal.

## Connections & Cross-Paper Themes
Identify non-obvious connections between papers. Look for:
- Papers that solve the same problem with different approaches (compare them)
- Papers that could be combined for a stronger result
- Contradictions or disagreements between papers' findings
- A shared mathematical framework appearing in different contexts

## Research Gaps & Opportunities
Based on what these papers do and don't address, identify 2-3 concrete research \
opportunities. For each: describe the gap, explain why it's important, and suggest \
a possible approach. These should be actionable ideas, not vague wishes.

## Recommended Reading
Rank the top 3 must-read papers and explain why each deserves priority attention. \
Consider: novelty of the approach, strength of results, potential for follow-up work, \
and relevance to the field direction."""


def generate(
    papers: list[dict],
    topic_name: str,
    session_dir: str | Path,
    cfg: dict,
) -> Path | None:
    """Generate insights.md for the session. Returns path or None on failure."""
    if not papers:
        log.info("No papers — skipping insights generation")
        return None

    summaries = []
    for p in papers:
        parts = [f"**{p['title']}**"]
        if p.get("venue"):
            parts[0] += f" ({p['venue']})"
        if p.get("key_insight"):
            parts.append(f"  Key insight: {p['key_insight']}")
        if p.get("method"):
            parts.append(f"  Method: {p['method']}")
        if p.get("contribution"):
            parts.append(f"  Contribution: {p['contribution']}")
        if p.get("summary"):
            parts.append(f"  Summary: {p['summary']}")
        if p.get("math_concepts"):
            parts.append(f"  Math: {', '.join(p['math_concepts'])}")
        summaries.append("\n".join(parts))

    paper_summaries = "\n\n".join(summaries)
    prompt = _INSIGHTS_PROMPT.format(
        n=len(papers),
        topic_name=topic_name,
        paper_summaries=paper_summaries,
    )

    raw = call_cli(prompt, cfg, timeout=240)
    if not raw:
        log.warning("Insights generation failed — no output from CLI")
        return None

    out_dir = Path(session_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "insights.md"
    path.write_text(raw, encoding="utf-8")
    log.info("Insights written to %s", path)
    return path
