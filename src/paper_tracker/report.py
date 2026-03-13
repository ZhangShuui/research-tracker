"""Generate a three-section Markdown session report.

Sections:
1. Executive Summary — LLM-generated high-level overview
2. Thematic Analysis — LLM-generated survey-style narrative grouped by method/theme
3. Paper Details — Structured per-paper appendix
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from paper_tracker.llm import call_cli

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts — designed for research-quality output
# ---------------------------------------------------------------------------

_EXEC_SUMMARY_PROMPT = """\
You are a senior research analyst writing an executive summary for a research team.

Topic: "{topic_name}"

Below are {n} newly discovered papers (title + key insight + method + contribution). \
Synthesize them into a concise executive summary (200-300 words) that covers:

1. **Most important discoveries**: What are the 2-3 breakthrough findings that a \
researcher in this field MUST know about? Be specific — name the papers.
2. **Field direction**: What trends or shifts do these papers collectively signal? \
Is the field moving toward a particular approach or away from another?
3. **Practical implications**: What should a practitioner or researcher do differently \
based on these findings? Any new tools, benchmarks, or techniques worth adopting?

Write in clear, direct academic prose. Cite papers by their title in bold. \
Do NOT use bullet points — write flowing paragraphs. \
Reply ONLY with the summary text, no headings or markdown formatting.

Papers:
{paper_block}"""

_THEMATIC_PROMPT = """\
You are a research analyst writing a mini survey-style review for a lab meeting.

Topic: "{topic_name}"

Below are {n} papers with their structured metadata. Write a thematic analysis \
(400-600 words) that:

1. **Groups papers by methodology or research theme** (NOT by individual paper). \
Identify 2-4 natural clusters. Give each cluster a descriptive heading.
2. **Within each theme**: explain the shared approach, compare how different papers \
address the problem differently, note agreements and disagreements, and highlight \
the most promising direction.
3. **Cross-theme connections**: Are there methodological bridges between clusters? \
Could combining approaches from different themes yield something novel?

Style guidelines:
- Write like a survey paper introduction — synthesize, don't just list
- Reference each paper by **title** (bold) when discussing it
- Include specific technical details (architectures, metrics, datasets) — avoid vague statements
- End with 1-2 sentences identifying the most significant open question or gap

Reply ONLY with the analysis in Markdown (use ## for theme headings). \
Do NOT include an overall title.

Papers:
{paper_block}"""


def _build_paper_block(papers: list[dict]) -> str:
    """Build a compact text representation of papers for LLM prompts."""
    lines = []
    for p in papers:
        parts = [f"**{p['title']}** [{p['arxiv_id']}]"]
        if p.get("key_insight"):
            parts.append(f"  Key insight: {p['key_insight']}")
        if p.get("method"):
            parts.append(f"  Method: {p['method']}")
        if p.get("contribution"):
            parts.append(f"  Contribution: {p['contribution']}")
        if p.get("venue"):
            parts.append(f"  Venue: {p['venue']}")
        if p.get("math_concepts"):
            parts.append(f"  Math: {', '.join(p['math_concepts'])}")
        if not p.get("key_insight") and p.get("summary"):
            parts.append(f"  Summary: {p['summary']}")
        lines.append("\n".join(parts))
    return "\n\n".join(lines)


def generate(
    papers: list[dict],
    repos: list[dict],
    session_dir: str | Path,
    *,
    topic_name: str = "Research",
    cfg: dict | None = None,
) -> Path | None:
    """Write a three-section Markdown report to session_dir/report.md."""
    if not papers and not repos:
        log.info("Nothing to report — skipping report generation")
        return None

    out_dir = Path(session_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "report.md"

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    lines: list[str] = [
        f"# Research Report — {topic_name}\n",
        f"*Generated {today} | {len(papers)} papers, {len(repos)} repos*\n",
    ]

    # --- Section 1: Executive Summary (LLM-generated) ---
    if papers and cfg:
        paper_block = _build_paper_block(papers)
        exec_prompt = _EXEC_SUMMARY_PROMPT.format(
            topic_name=topic_name,
            n=len(papers),
            paper_block=paper_block,
        )
        exec_summary = call_cli(exec_prompt, cfg, timeout=180)
        if exec_summary:
            lines.append("## Executive Summary\n")
            lines.append(exec_summary)
            lines.append("")
        else:
            log.warning("Executive summary generation failed — using fallback")
            lines.append("## Executive Summary\n")
            lines.append(
                f"This session discovered {len(papers)} new papers and "
                f"{len(repos)} new repositories in **{topic_name}**."
            )
            lines.append("")

    # --- Section 2: Thematic Analysis (LLM-generated) ---
    if papers and cfg and len(papers) >= 2:
        thematic_prompt = _THEMATIC_PROMPT.format(
            topic_name=topic_name,
            n=len(papers),
            paper_block=_build_paper_block(papers),
        )
        thematic = call_cli(thematic_prompt, cfg, timeout=180)
        if thematic:
            lines.append("## Thematic Analysis\n")
            lines.append(thematic)
            lines.append("")
        else:
            log.warning("Thematic analysis generation failed — skipping")

    # --- Section 3: Paper Details (structured appendix) ---
    if papers:
        lines.append(f"## Paper Details ({len(papers)})\n")
        for p in papers:
            lines.append(f"### {p['title']}\n")
            lines.append(f"- **ID**: [{p['arxiv_id']}]({p['url']})")
            lines.append(f"- **Authors**: {p['authors']}")
            lines.append(f"- **Published**: {p['published']}")
            if p.get("venue"):
                lines.append(f"- **Venue**: {p['venue']}")
            if p.get("key_insight"):
                lines.append(f"- **Key Insight**: {p['key_insight']}")
            if p.get("method"):
                lines.append(f"- **Method**: {p['method']}")
            if p.get("contribution"):
                lines.append(f"- **Contribution**: {p['contribution']}")
            lines.append(f"- **Summary**: {p['summary']}")
            if p.get("math_concepts"):
                lines.append(f"- **Math Concepts**: {', '.join(p['math_concepts'])}")
            if p.get("cited_works"):
                lines.append(f"- **Cited Works**: {'; '.join(p['cited_works'])}")
            lines.append("")

    if repos:
        lines.append(f"## GitHub Repos ({len(repos)})\n")
        for r in repos:
            lines.append(f"### [{r['repo_full_name']}]({r['url']})\n")
            lines.append(f"- **Stars**: {r['stars']}")
            lines.append(f"- **Last pushed**: {r['pushed_at']}")
            lines.append(f"- **Summary**: {r['summary']}\n")

    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Report written to %s", path)
    return path
