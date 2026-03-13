"""Discovery pipelines: Trending Themes + Math Insights.

Cross-topic discovery that aggregates papers from multiple sources
and uses LLM to extract themes and insights.
"""

from __future__ import annotations

import json
import logging
import random
import time
from datetime import datetime, timezone

from paper_tracker.llm import call_cli
from paper_tracker.registry import Registry
from paper_tracker.sources.arxiv import search_broad, search_random_era
from paper_tracker.sources.huggingface import fetch_daily_papers
from paper_tracker.sources.paperswithcode import fetch_trending as fetch_pwc_trending

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Trending Themes Pipeline
# ---------------------------------------------------------------------------

_TRENDING_CATEGORIES = ["cs.AI", "cs.LG", "cs.CV", "cs.CL"]

_TRENDING_PROMPT = """\
You are a research trend analyst. Below are {n_papers} recent papers from \
arXiv, HuggingFace Daily Papers, and Papers With Code.

## Papers
{papers_text}

## Source Statistics
{source_stats}

## Your Task
Identify 5-8 **trending research themes** from these papers. For EACH theme:

1. "title": A concise, descriptive theme name (3-8 words)
2. "representative_papers": List of 3-5 arXiv IDs that best represent this theme
3. "description": A 2-3 sentence description of what this theme is about, \
what problems it addresses, and why it's gaining traction
4. "trend_direction": One of "EMERGING" (brand new, <6 months), \
"ACCELERATING" (growing fast), "ESTABLISHED" (mature but active), \
"PEAKING" (lots of activity but possibly saturating)
5. "key_techniques": List of 3-5 specific techniques, architectures, or methods \
central to this theme

After all themes, add a "cross_theme_observations" object with:
- "overarching_narrative": A 2-3 sentence synthesis of the overall research landscape
- "unexpected_convergences": 2-3 cases where different themes are converging
- "landscape_gaps": 2-3 areas that seem underexplored given current trends

Reply ONLY with a JSON object: {{"themes": [...], "cross_theme_observations": {{...}}}}
No markdown fences."""


def _try_parse_json(text: str):
    """Best-effort JSON parse from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        for sc, ec in [("{", "}"), ("[", "]")]:
            start = text.find(sc)
            end = text.rfind(ec)
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except (json.JSONDecodeError, TypeError):
                    continue
    return None


def _format_papers_for_prompt(papers: list[dict], max_papers: int = 150) -> str:
    """Format papers into compact text for LLM prompt."""
    lines = []
    for p in papers[:max_papers]:
        source = p.get("source", "arxiv")
        published = p.get("published", "")[:10]  # YYYY-MM-DD
        pool = p.get("pool", "")
        tag = f"({source})"
        if pool and pool != "recent":
            tag = f"({source}, {pool})"
        if published:
            tag = f"({source}, {published})"
            if pool and pool != "recent":
                tag = f"({source}, {published}, {pool})"
        lines.append(
            f"[{p['arxiv_id']}] {tag} {p['title']}\n"
            f"  {p.get('abstract', '')[:300]}"
        )
    return "\n\n".join(lines)


def run_trending(registry: Registry, cfg: dict) -> dict:
    """Run the trending themes discovery pipeline."""
    log.info("=== Trending discovery started ===")

    report = registry.create_discovery_report("trending")
    report_id = report["id"]

    try:
        # Collect from all sources
        arxiv_papers = search_broad(_TRENDING_CATEGORIES, lookback_days=7, max_results=200)
        for p in arxiv_papers:
            p.setdefault("source", "arxiv")

        hf_papers = fetch_daily_papers()
        pwc_papers = fetch_pwc_trending(max_papers=50)

        # Deduplicate by arxiv_id
        seen: set[str] = set()
        all_papers: list[dict] = []
        source_counts: dict[str, int] = {"arxiv": 0, "huggingface": 0, "paperswithcode": 0}

        for p in arxiv_papers + hf_papers + pwc_papers:
            aid = p.get("arxiv_id", "")
            if aid and aid not in seen:
                seen.add(aid)
                all_papers.append(p)
                source_counts[p.get("source", "arxiv")] = source_counts.get(p.get("source", "arxiv"), 0) + 1

        log.info("Collected %d unique papers (arxiv=%d, hf=%d, pwc=%d)",
                 len(all_papers), source_counts.get("arxiv", 0),
                 source_counts.get("huggingface", 0), source_counts.get("paperswithcode", 0))

        # Update report with paper count
        papers_json = [
            {"arxiv_id": p["arxiv_id"], "title": p.get("title", ""), "source": p.get("source", "arxiv")}
            for p in all_papers
        ]
        registry.update_discovery_report(report_id, {
            "paper_count": len(all_papers),
            "papers_json": papers_json,
            "source_stats": source_counts,
        })

        # LLM clustering
        papers_text = _format_papers_for_prompt(all_papers)
        source_stats_text = "\n".join(f"- {k}: {v} papers" for k, v in source_counts.items())

        prompt = _TRENDING_PROMPT.format(
            n_papers=len(all_papers),
            papers_text=papers_text,
            source_stats=source_stats_text,
        )

        raw = call_cli(prompt, cfg, timeout=360)
        content = raw or "No trending themes could be generated."

        # Format as markdown report
        md = _format_trending_markdown(content, all_papers, source_counts)

        registry.update_discovery_report(report_id, {
            "status": "completed",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "content": md,
        })

        log.info("=== Trending discovery completed: %d papers analyzed ===", len(all_papers))
        return {"report_id": report_id, "paper_count": len(all_papers)}

    except Exception as e:
        log.exception("Trending discovery failed: %s", e)
        registry.update_discovery_report(report_id, {
            "status": "failed",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "content": f"Discovery failed: {e}",
        })
        return {"report_id": report_id, "error": str(e)}


def _format_trending_markdown(raw_content: str, papers: list[dict], source_counts: dict) -> str:
    """Convert LLM JSON output into readable markdown report."""
    header = (
        f"# Trending Research Themes\n\n"
        f"**Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"**Papers analyzed**: {len(papers)}\n"
        f"**Sources**: arXiv ({source_counts.get('arxiv', 0)}), "
        f"HuggingFace ({source_counts.get('huggingface', 0)}), "
        f"Papers With Code ({source_counts.get('paperswithcode', 0)})\n\n"
        f"---\n\n"
    )

    # Try to parse JSON and format nicely
    parsed = _try_parse_json(raw_content)
    if not parsed or not isinstance(parsed, dict):
        return header + raw_content

    body_parts = []
    themes = parsed.get("themes", [])
    for i, theme in enumerate(themes, 1):
        title = theme.get("title", f"Theme {i}")
        direction = theme.get("trend_direction", "")
        desc = theme.get("description", "")
        papers_list = theme.get("representative_papers", [])
        techniques = theme.get("key_techniques", [])

        direction_emoji = {
            "EMERGING": "🌱", "ACCELERATING": "🚀",
            "ESTABLISHED": "📊", "PEAKING": "📈",
        }.get(direction, "")

        section = f"## {i}. {title}\n\n"
        if direction:
            section += f"**Trend**: {direction_emoji} {direction}\n\n"
        if desc:
            section += f"{desc}\n\n"
        if techniques:
            section += "**Key Techniques**: " + " · ".join(f"`{t}`" for t in techniques) + "\n\n"
        if papers_list:
            section += "**Representative Papers**: " + ", ".join(f"[{p}](https://arxiv.org/abs/{p})" for p in papers_list) + "\n\n"

        body_parts.append(section)

    # Cross-theme observations
    obs = parsed.get("cross_theme_observations", {})
    if obs:
        body_parts.append("---\n\n## Cross-Theme Observations\n\n")
        if obs.get("overarching_narrative"):
            body_parts.append(f"### Overarching Narrative\n\n{obs['overarching_narrative']}\n\n")
        if obs.get("unexpected_convergences"):
            conv = obs["unexpected_convergences"]
            if isinstance(conv, list):
                body_parts.append("### Unexpected Convergences\n\n")
                for c in conv:
                    body_parts.append(f"- {c}\n")
                body_parts.append("\n")
            elif isinstance(conv, str):
                body_parts.append(f"### Unexpected Convergences\n\n{conv}\n\n")
        if obs.get("landscape_gaps"):
            gaps = obs["landscape_gaps"]
            if isinstance(gaps, list):
                body_parts.append("### Landscape Gaps\n\n")
                for g in gaps:
                    body_parts.append(f"- {g}\n")
                body_parts.append("\n")
            elif isinstance(gaps, str):
                body_parts.append(f"### Landscape Gaps\n\n{gaps}\n\n")

    return header + "".join(body_parts)


# ---------------------------------------------------------------------------
# Math Insights Pipeline
# ---------------------------------------------------------------------------

_MATH_CORE_CATEGORIES = [
    "math.ST", "stat.ML", "math.PR", "math.OC",
    "stat.TH", "stat.ME", "math.NA",
]

# Broader categories for serendipitous discovery (including non-ML areas)
_MATH_WILDCARD_CATEGORIES = [
    "math.CO",  # Combinatorics
    "math.AG",  # Algebraic Geometry
    "math.GT",  # Geometric Topology
    "math.DS",  # Dynamical Systems
    "math.FA",  # Functional Analysis
    "math.IT",  # Information Theory (= cs.IT)
    "math.LO",  # Logic
    "math.DG",  # Differential Geometry
    "math.RT",  # Representation Theory
    "math.CA",  # Classical Analysis and ODEs
    "math.AP",  # Analysis of PDEs
    "math.CT",  # Category Theory
    "physics.data-an",  # Data Analysis, Statistics and Probability
    "nlin.CD",  # Chaotic Dynamics
    "q-bio.QM",  # Quantitative Methods in Biology
]

_MATH_PROMPT = """\
You are a mathematical research analyst who specializes in finding unexpected \
connections between mathematics (including classical results) and modern machine learning.

Below are {n_papers} papers spanning different eras and mathematical areas — some recent, \
some from decades ago. The mix is intentional: we want fresh eyes on old ideas and \
cross-pollination between distant fields.

## Papers
{papers_text}

## Your Task

### Per-Paper Analysis
For each paper, provide:
1. "arxiv_id": The paper's arXiv ID
2. "core_concepts": The key theorems, lemmas, or mathematical concepts (1-2 sentences)
3. "math_techniques": List of 2-4 specific mathematical techniques used
4. "ml_applications": Potential applications to machine learning (1-2 sentences). \
Think VERY creatively — even if the paper seems unrelated to ML, find an angle. \
Could the proof technique inspire a new algorithm? Could the structure map onto \
a neural architecture? Could the theorem bound something useful?
5. "elegance_score": 1-5 (5 = beautifully novel or deeply insightful, 1 = routine)

### Synthesis
After all papers, provide a "synthesis" object with:
- "recurring_techniques": 3-5 mathematical techniques that appear across multiple papers \
or eras, showing the endurance of certain mathematical ideas
- "overlooked_connections": 2-3 connections between these math results and ML that the \
ML community likely hasn't noticed. Be bold — the best ideas come from distant analogies.
- "recommended_reading_path": An ordered list of 3-5 paper IDs for someone wanting \
to build an unusual but powerful mathematical toolkit for ML
- "wild_idea": One speculative but exciting research idea that combines insights \
from these papers with current ML challenges (3-5 sentences). The wilder the better, \
as long as there's a kernel of mathematical logic behind it.
- "time_travel_insight": One idea from an older paper (pre-2015) that was ahead of \
its time and could be directly applied to a 2025 ML problem (2-3 sentences).

Reply ONLY with a JSON object: {{"papers": [...], "synthesis": {{...}}}}
No markdown fences."""


def run_math_insights(
    registry: Registry,
    cfg: dict,
    *,
    categories: list[str] | None = None,
    wildcard_categories: list[str] | None = None,
    lookback_days: int | None = None,
    max_recent: int | None = None,
    max_historical: int | None = None,
    max_wildcard: int | None = None,
    sample_size: int | None = None,
) -> dict:
    """Run the math insights discovery pipeline.

    Collects papers from three pools:
    1. Recent math/stats papers (last 14 days) — what's new
    2. Historical papers from random eras (1994-2024) — serendipitous rediscovery
    3. Wildcard categories (topology, algebra, physics, etc.) — cross-pollination
    """
    core_cats = categories or _MATH_CORE_CATEGORIES
    wild_cats = wildcard_categories or _MATH_WILDCARD_CATEGORIES
    lb_days = lookback_days or 14
    mr_recent = max_recent or 100
    mr_historical = max_historical or 30
    mr_wildcard = max_wildcard or 15
    n_sample = sample_size or 25

    log.info("=== Math insights discovery started (cats=%s, lookback=%d, sample=%d) ===",
             core_cats, lb_days, n_sample)

    report = registry.create_discovery_report("math")
    report_id = report["id"]

    try:
        # Pool 1: Recent papers from core math/stats categories
        recent_papers = search_broad(core_cats, lookback_days=lb_days, max_results=mr_recent)
        for p in recent_papers:
            p.setdefault("source", "arxiv")
            p["pool"] = "recent"

        # Pool 2: Historical papers from random eras in core categories
        historical_papers = search_random_era(core_cats, max_results=mr_historical)
        for p in historical_papers:
            p.setdefault("source", "arxiv")
            p["pool"] = "historical"

        # Pool 3: Wildcard — random era, random non-standard categories
        wildcard_cats = random.sample(wild_cats, min(4, len(wild_cats)))
        wildcard_papers = search_random_era(wildcard_cats, max_results=mr_wildcard)
        for p in wildcard_papers:
            p.setdefault("source", "arxiv")
            p["pool"] = "wildcard"

        # Deduplicate
        seen: set[str] = set()
        all_papers: list[dict] = []
        for p in recent_papers + historical_papers + wildcard_papers:
            aid = p.get("arxiv_id", "")
            if aid and aid not in seen:
                seen.add(aid)
                all_papers.append(p)

        pool_counts = {
            "recent": sum(1 for p in all_papers if p.get("pool") == "recent"),
            "historical": sum(1 for p in all_papers if p.get("pool") == "historical"),
            "wildcard": sum(1 for p in all_papers if p.get("pool") == "wildcard"),
        }
        log.info("Math discovery collected %d papers (recent=%d, historical=%d, wildcard=%d)",
                 len(all_papers), pool_counts["recent"],
                 pool_counts["historical"], pool_counts["wildcard"])

        # Sample for analysis: mix from all pools (proportional to n_sample)
        n_recent = max(1, int(n_sample * 0.4))
        n_hist = max(1, int(n_sample * 0.4))
        n_wild = max(1, n_sample - n_recent - n_hist)
        sampled: list[dict] = []
        for pool_name, target in [("recent", n_recent), ("historical", n_hist), ("wildcard", n_wild)]:
            pool = [p for p in all_papers if p.get("pool") == pool_name]
            if len(pool) > target:
                sampled.extend(random.sample(pool, target))
            else:
                sampled.extend(pool)
        # If we still need more, fill from any pool
        remaining_ids = {p["arxiv_id"] for p in sampled}
        extras = [p for p in all_papers if p["arxiv_id"] not in remaining_ids]
        if len(sampled) < n_sample and extras:
            sampled.extend(random.sample(extras, min(n_sample - len(sampled), len(extras))))

        random.shuffle(sampled)

        # Update report with paper info
        papers_json = [
            {
                "arxiv_id": p["arxiv_id"],
                "title": p.get("title", ""),
                "source": "arxiv",
                "pool": p.get("pool", "recent"),
                "era": p.get("era", ""),
            }
            for p in all_papers
        ]
        source_counts = {
            "total": len(all_papers),
            **pool_counts,
        }
        registry.update_discovery_report(report_id, {
            "paper_count": len(all_papers),
            "papers_json": papers_json,
            "source_stats": source_counts,
        })

        # LLM analysis
        papers_text = _format_papers_for_prompt(sampled, max_papers=25)

        prompt = _MATH_PROMPT.format(
            n_papers=len(sampled),
            papers_text=papers_text,
        )

        raw = call_cli(prompt, cfg, timeout=360)
        content = raw or "No math insights could be generated."

        md = _format_math_markdown(content, all_papers, sampled)

        registry.update_discovery_report(report_id, {
            "status": "completed",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "content": md,
        })

        log.info("=== Math insights completed: %d sampled from %d (recent=%d, historical=%d, wildcard=%d) ===",
                 len(sampled), len(all_papers),
                 pool_counts["recent"], pool_counts["historical"], pool_counts["wildcard"])
        return {"report_id": report_id, "paper_count": len(all_papers)}

    except Exception as e:
        log.exception("Math insights discovery failed: %s", e)
        registry.update_discovery_report(report_id, {
            "status": "failed",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "content": f"Discovery failed: {e}",
        })
        return {"report_id": report_id, "error": str(e)}


def _format_math_markdown(raw_content: str, all_papers: list[dict], sampled: list[dict]) -> str:
    """Convert LLM JSON output into readable markdown report."""
    pool_counts = {}
    for p in sampled:
        pool = p.get("pool", "recent")
        pool_counts[pool] = pool_counts.get(pool, 0) + 1

    pool_str = ", ".join(f"{k}: {v}" for k, v in pool_counts.items()) if pool_counts else "mixed"
    cats = sorted(set(c for c in _MATH_CORE_CATEGORIES + _MATH_WILDCARD_CATEGORIES
                      if any(c in str(p.get("url", "")) for p in sampled)))
    cat_str = ", ".join(_MATH_CORE_CATEGORIES) + " + wildcards"

    header = (
        f"# Math & Statistics Insights for ML\n\n"
        f"**Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"**Total papers found**: {len(all_papers)}\n"
        f"**Papers analyzed**: {len(sampled)} ({pool_str})\n"
        f"**Categories**: {cat_str}\n\n"
        f"---\n\n"
    )

    parsed = _try_parse_json(raw_content)
    if not parsed or not isinstance(parsed, dict):
        return header + raw_content

    body_parts = []

    # Per-paper analysis
    paper_analyses = parsed.get("papers", [])
    if paper_analyses:
        body_parts.append("## Paper Analyses\n\n")
        for i, pa in enumerate(paper_analyses, 1):
            aid = pa.get("arxiv_id", f"Paper {i}")
            # Find title from sampled papers
            title = ""
            for sp in sampled:
                if sp.get("arxiv_id") == aid:
                    title = sp.get("title", "")
                    break
            elegance = pa.get("elegance_score", 0)
            stars = "★" * elegance + "☆" * (5 - elegance) if isinstance(elegance, int) else ""

            section = f"### {i}. [{aid}](https://arxiv.org/abs/{aid})"
            if title:
                section += f" — {title}"
            section += "\n\n"

            if stars:
                section += f"**Elegance**: {stars} ({elegance}/5)\n\n"

            if pa.get("core_concepts"):
                section += f"**Core Concepts**: {pa['core_concepts']}\n\n"
            if pa.get("math_techniques"):
                techniques = pa["math_techniques"]
                if isinstance(techniques, list):
                    section += "**Techniques**: " + " · ".join(f"`{t}`" for t in techniques) + "\n\n"
                else:
                    section += f"**Techniques**: {techniques}\n\n"
            if pa.get("ml_applications"):
                section += f"**ML Applications**: {pa['ml_applications']}\n\n"

            body_parts.append(section)

    # Synthesis
    synthesis = parsed.get("synthesis", {})
    if synthesis:
        body_parts.append("---\n\n## Synthesis\n\n")

        if synthesis.get("recurring_techniques"):
            rt = synthesis["recurring_techniques"]
            body_parts.append("### Recurring Techniques\n\n")
            if isinstance(rt, list):
                for t in rt:
                    body_parts.append(f"- {t}\n")
            else:
                body_parts.append(f"{rt}\n")
            body_parts.append("\n")

        if synthesis.get("overlooked_connections"):
            oc = synthesis["overlooked_connections"]
            body_parts.append("### Overlooked Connections\n\n")
            if isinstance(oc, list):
                for c in oc:
                    body_parts.append(f"- {c}\n")
            else:
                body_parts.append(f"{oc}\n")
            body_parts.append("\n")

        if synthesis.get("recommended_reading_path"):
            rp = synthesis["recommended_reading_path"]
            body_parts.append("### Recommended Reading Path\n\n")
            if isinstance(rp, list):
                for j, paper_id in enumerate(rp, 1):
                    body_parts.append(f"{j}. [{paper_id}](https://arxiv.org/abs/{paper_id})\n")
            else:
                body_parts.append(f"{rp}\n")
            body_parts.append("\n")

        if synthesis.get("wild_idea"):
            body_parts.append(f"### Wild Idea 💡\n\n{synthesis['wild_idea']}\n\n")

        if synthesis.get("time_travel_insight"):
            body_parts.append(f"### Time Travel Insight ⏳\n\n{synthesis['time_travel_insight']}\n\n")

    return header + "".join(body_parts)


# ---------------------------------------------------------------------------
# Community Ideas Pipeline
# ---------------------------------------------------------------------------

_COMMUNITY_PROMPT = """\
You are a research idea analyst who specializes in extracting promising research \
directions from online community discussions (HackerNews, Reddit, forums, blogs).

Below are {n_posts} discussion snippets from various online communities about AI/ML topics.

## Community Discussions
{posts_text}

## Your Task
Identify 5-10 **research-worthy ideas** from these community discussions. \
Focus on ideas that:
- Address real-world pain points practitioners mention
- Suggest novel combinations of techniques
- Identify underserved problems or overlooked opportunities
- Challenge established approaches with practical evidence

For EACH idea:
1. "title": A concise, descriptive title (3-8 words)
2. "source_posts": List of 1-3 source titles/URLs that inspired this idea
3. "problem": What real-world problem or pain point does this address? (2-3 sentences)
4. "proposed_direction": A specific research direction to explore (2-3 sentences)
5. "why_community": Why does the community perspective matter here — what do practitioners \
see that academics might miss? (1-2 sentences)
6. "feasibility": One of "QUICK_WIN" (could test in a week), "PROJECT" (1-3 months), \
"AMBITIOUS" (6+ months)
7. "excitement_score": 1-5 (5 = widely discussed and high impact, 1 = niche mention)

After all ideas, add a "meta" object with:
- "hot_topics": 3-5 topics the community is most actively discussing right now
- "pain_points": 3-5 recurring frustrations or unmet needs
- "emerging_tools": 2-3 new tools, libraries, or frameworks gaining traction
- "contrarian_takes": 2-3 opinions that go against mainstream academic consensus

Reply ONLY with a JSON object: {{"ideas": [...], "meta": {{...}}}}
No markdown fences."""


def run_community_ideas(
    registry: Registry,
    cfg: dict,
    *,
    keywords: list[str] | None = None,
    platforms: list[str] | None = None,
    max_results_per_platform: int = 15,
) -> dict:
    """Run community ideas discovery pipeline.

    Searches HackerNews, Reddit, and general web for AI/ML discussions,
    then uses LLM to extract research ideas from community perspectives.
    """
    from paper_tracker.sources.web import search_hackernews, search_web

    search_keywords = keywords or [
        "machine learning research idea",
        "deep learning breakthrough",
        "LLM limitations practical",
        "AI research direction",
        "reinforcement learning application",
    ]
    active_platforms = platforms or ["hackernews", "reddit", "web"]

    log.info("=== Community ideas discovery started (keywords=%d, platforms=%s) ===",
             len(search_keywords), active_platforms)

    report = registry.create_discovery_report("community")
    report_id = report["id"]

    try:
        all_posts: list[dict] = []
        seen_titles: set[str] = set()
        source_counts: dict[str, int] = {}

        for kw in search_keywords:
            # HackerNews
            if "hackernews" in active_platforms:
                hn_results = search_hackernews(kw, max_results=max_results_per_platform)
                for r in hn_results:
                    key = r.get("title", "").lower().strip()[:60]
                    if key and key not in seen_titles:
                        seen_titles.add(key)
                        r["keyword"] = kw
                        all_posts.append(r)
                        source_counts["hackernews"] = source_counts.get("hackernews", 0) + 1
                time.sleep(0.5)

            # Reddit
            if "reddit" in active_platforms:
                reddit_results = search_web(
                    f"{kw} site:reddit.com",
                    max_results=max_results_per_platform,
                )
                for r in reddit_results:
                    r["source"] = "reddit"
                    key = r.get("title", "").lower().strip()[:60]
                    if key and key not in seen_titles:
                        seen_titles.add(key)
                        r["keyword"] = kw
                        all_posts.append(r)
                        source_counts["reddit"] = source_counts.get("reddit", 0) + 1
                time.sleep(1.0)

            # General web
            if "web" in active_platforms:
                web_results = search_web(
                    f"{kw} discussion opinion blog",
                    max_results=max_results_per_platform,
                )
                for r in web_results:
                    key = r.get("title", "").lower().strip()[:60]
                    if key and key not in seen_titles:
                        seen_titles.add(key)
                        r["keyword"] = kw
                        all_posts.append(r)
                        src = r.get("source", "web")
                        source_counts[src] = source_counts.get(src, 0) + 1
                time.sleep(1.0)

        log.info("Community discovery collected %d posts from %s",
                 len(all_posts), source_counts)

        # Update report with post counts
        posts_json = [
            {
                "arxiv_id": r.get("url", ""),
                "title": r.get("title", ""),
                "source": r.get("source", "web"),
            }
            for r in all_posts
        ]
        registry.update_discovery_report(report_id, {
            "paper_count": len(all_posts),
            "papers_json": posts_json,
            "source_stats": source_counts,
        })

        if not all_posts:
            registry.update_discovery_report(report_id, {
                "status": "completed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "content": "No community discussions found. Check your search keywords and internet connectivity.",
            })
            return {"report_id": report_id, "post_count": 0}

        # Format posts for LLM
        posts_text = _format_posts_for_prompt(all_posts, max_posts=80)

        prompt = _COMMUNITY_PROMPT.format(
            n_posts=min(len(all_posts), 80),
            posts_text=posts_text,
        )

        raw = call_cli(prompt, cfg, timeout=360)
        content = raw or "No community ideas could be generated."

        md = _format_community_markdown(content, all_posts, source_counts)

        registry.update_discovery_report(report_id, {
            "status": "completed",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "content": md,
        })

        log.info("=== Community ideas completed: %d posts analyzed ===", len(all_posts))
        return {"report_id": report_id, "post_count": len(all_posts)}

    except Exception as e:
        log.exception("Community ideas discovery failed: %s", e)
        registry.update_discovery_report(report_id, {
            "status": "failed",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "content": f"Discovery failed: {e}",
        })
        return {"report_id": report_id, "error": str(e)}


def _format_posts_for_prompt(posts: list[dict], max_posts: int = 80) -> str:
    """Format community posts into compact text for LLM prompt."""
    # Sort by score/engagement if available
    scored = sorted(posts, key=lambda p: p.get("score", 0) + p.get("comments", 0), reverse=True)
    lines = []
    for p in scored[:max_posts]:
        source = p.get("source", "web")
        title = p.get("title", "")
        snippet = p.get("snippet", "")
        score = p.get("score", 0)
        comments = p.get("comments", 0)
        url = p.get("url", "")

        meta_parts = [source]
        if score:
            meta_parts.append(f"score:{score}")
        if comments:
            meta_parts.append(f"comments:{comments}")
        meta = ", ".join(meta_parts)

        entry = f"[{meta}] {title}"
        if url:
            entry += f"\n  URL: {url}"
        if snippet and snippet not in ("(external link)", "(link post)"):
            entry += f"\n  {snippet[:400]}"
        lines.append(entry)
    return "\n\n".join(lines)


def _format_community_markdown(raw_content: str, posts: list[dict], source_counts: dict) -> str:
    """Convert LLM JSON output into readable markdown report."""
    header = (
        f"# Community Research Ideas\n\n"
        f"**Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"**Discussions analyzed**: {len(posts)}\n"
        f"**Sources**: " + ", ".join(f"{k} ({v})" for k, v in source_counts.items()) + "\n\n"
        f"---\n\n"
    )

    parsed = _try_parse_json(raw_content)
    if not parsed or not isinstance(parsed, dict):
        return header + raw_content

    body_parts = []

    # Ideas
    ideas = parsed.get("ideas", [])
    for i, idea in enumerate(ideas, 1):
        title = idea.get("title", f"Idea {i}")
        feasibility = idea.get("feasibility", "")
        excitement = idea.get("excitement_score", 0)
        problem = idea.get("problem", "")
        direction = idea.get("proposed_direction", "")
        why = idea.get("why_community", "")
        sources = idea.get("source_posts", [])

        feas_emoji = {
            "QUICK_WIN": "⚡", "PROJECT": "🔬", "AMBITIOUS": "🚀",
        }.get(feasibility, "")
        stars = "★" * excitement + "☆" * (5 - excitement) if isinstance(excitement, int) else ""

        section = f"## {i}. {title}\n\n"
        if feasibility:
            section += f"**Feasibility**: {feas_emoji} {feasibility}"
        if stars:
            section += f"  |  **Excitement**: {stars} ({excitement}/5)"
        section += "\n\n"
        if problem:
            section += f"**Problem**: {problem}\n\n"
        if direction:
            section += f"**Research Direction**: {direction}\n\n"
        if why:
            section += f"**Community Insight**: {why}\n\n"
        if sources:
            section += "**Inspired by**: "
            if isinstance(sources, list):
                section += " · ".join(str(s) for s in sources)
            else:
                section += str(sources)
            section += "\n\n"

        body_parts.append(section)

    # Meta
    meta = parsed.get("meta", {})
    if meta:
        body_parts.append("---\n\n## Community Pulse\n\n")

        if meta.get("hot_topics"):
            ht = meta["hot_topics"]
            body_parts.append("### Hot Topics\n\n")
            if isinstance(ht, list):
                for t in ht:
                    body_parts.append(f"- {t}\n")
            else:
                body_parts.append(f"{ht}\n")
            body_parts.append("\n")

        if meta.get("pain_points"):
            pp = meta["pain_points"]
            body_parts.append("### Pain Points\n\n")
            if isinstance(pp, list):
                for p in pp:
                    body_parts.append(f"- {p}\n")
            else:
                body_parts.append(f"{pp}\n")
            body_parts.append("\n")

        if meta.get("emerging_tools"):
            et = meta["emerging_tools"]
            body_parts.append("### Emerging Tools\n\n")
            if isinstance(et, list):
                for t in et:
                    body_parts.append(f"- {t}\n")
            else:
                body_parts.append(f"{et}\n")
            body_parts.append("\n")

        if meta.get("contrarian_takes"):
            ct = meta["contrarian_takes"]
            body_parts.append("### Contrarian Takes\n\n")
            if isinstance(ct, list):
                for t in ct:
                    body_parts.append(f"- {t}\n")
            else:
                body_parts.append(f"{ct}\n")
            body_parts.append("\n")

    return header + "".join(body_parts)


# ---------------------------------------------------------------------------
# Quality Review
# ---------------------------------------------------------------------------

_QUALITY_REVIEW_TRENDING_PROMPT = """\
You are a quality reviewer for a research trend analysis report.

## Report Content
{content}

## Report Metadata
- Papers analyzed: {paper_count}
- Sources: {source_stats}

## Quality Criteria
Score this report (0-100) based on:
1. **Theme count** (15pts): Are there 5-8 distinct, non-overlapping themes?
2. **Theme depth** (20pts): Does each theme have representative papers, description, \
trend direction, and key techniques?
3. **Cross-theme observations** (15pts): Are there overarching narratives, unexpected \
convergences, and landscape gaps?
4. **Specificity** (20pts): Does the report cite specific papers, methods, and results \
rather than making vague claims?
5. **Source diversity** (10pts): Are papers from multiple sources (arXiv, HuggingFace, \
Papers With Code) referenced?
6. **Actionability** (10pts): Could a researcher use this to identify promising research \
directions?
7. **Formatting** (10pts): Is the report well-structured and readable?

## Output
Reply ONLY with a JSON object:
{{
  "quality_score": <0-100>,
  "flags": [
    {{"issue": "<short description>", "severity": "low|medium|high"}}
  ],
  "summary": "<1-2 sentence overall assessment>"
}}
No markdown fences."""

_QUALITY_REVIEW_MATH_PROMPT = """\
You are a quality reviewer for a mathematical insights report.

## Report Content
{content}

## Report Metadata
- Papers analyzed: {paper_count}

## Quality Criteria
Score this report (0-100) based on:
1. **Paper coverage** (15pts): Are enough papers analyzed in depth (ideally 20-25)?
2. **Mathematical rigor** (20pts): Are core concepts, theorems, and techniques \
described accurately?
3. **ML connections** (20pts): Are the connections to ML creative, specific, and \
plausible (not hand-wavy)?
4. **Synthesis quality** (15pts): Does the synthesis identify recurring techniques, \
overlooked connections, and a reading path?
5. **Wild idea** (10pts): Is there a creative, specific wild idea that combines \
insights from the papers?
6. **Specificity** (10pts): Are specific papers, techniques, and applications named?
7. **Formatting** (10pts): Is the report well-structured and readable?

## Output
Reply ONLY with a JSON object:
{{
  "quality_score": <0-100>,
  "flags": [
    {{"issue": "<short description>", "severity": "low|medium|high"}}
  ],
  "summary": "<1-2 sentence overall assessment>"
}}
No markdown fences."""


def review_discovery_report(registry: Registry, report_id: str, cfg: dict) -> dict:
    """Run LLM-based quality review on a completed discovery report.

    Returns dict with quality_score (0-100) and quality_flags (list of issues).
    Also updates the report in the registry.
    """
    report = registry.get_discovery_report(report_id)
    if not report:
        return {"error": "Report not found"}
    if report["status"] != "completed":
        return {"error": "Can only review completed reports"}

    content = report.get("content", "")
    if not content:
        result = {"quality_score": 0, "flags": [{"issue": "Empty content", "severity": "high"}], "summary": "Report has no content."}
        registry.update_discovery_report(report_id, {
            "quality_score": 0,
            "quality_flags": result["flags"],
        })
        return result

    # Choose prompt based on report type
    if report["type"] == "trending":
        prompt = _QUALITY_REVIEW_TRENDING_PROMPT.format(
            content=content[:6000],
            paper_count=report.get("paper_count", 0),
            source_stats=", ".join(f"{k}: {v}" for k, v in report.get("source_stats", {}).items()),
        )
    else:
        prompt = _QUALITY_REVIEW_MATH_PROMPT.format(
            content=content[:6000],
            paper_count=report.get("paper_count", 0),
        )

    raw = call_cli(prompt, cfg, timeout=120)

    if not raw:
        result = {"quality_score": -1, "flags": [{"issue": "Review LLM call failed", "severity": "high"}], "summary": "Could not run quality review."}
        return result

    # Parse response
    try:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        # Try to extract JSON
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1:
            try:
                parsed = json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                parsed = None
        else:
            parsed = None

    if not parsed:
        return {"quality_score": -1, "flags": [{"issue": "Could not parse review output", "severity": "medium"}], "summary": "Review output was not valid JSON."}

    quality_score = parsed.get("quality_score", -1)
    flags = parsed.get("flags", [])
    summary = parsed.get("summary", "")

    # Update registry
    registry.update_discovery_report(report_id, {
        "quality_score": quality_score,
        "quality_flags": flags,
    })

    log.info("Quality review for %s: score=%d, flags=%d", report_id, quality_score, len(flags))

    return {
        "quality_score": quality_score,
        "flags": flags,
        "summary": summary,
    }
