"""Generate cross-paper insights via shared LLM utility.

Produces a structured research insights report highlighting trends,
methodological connections, research gaps, and recommended reading.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from paper_tracker.llm import call_cli
from paper_tracker.sources import arxiv

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


def _write_papers_manifest(session_dir: str | Path, papers: list[dict]) -> Path | None:
    """Persist provenance for brainstorm: ``[Pn]`` index -> paper metadata + card/abstract.

    Written next to insights.md as ``insights_papers.json`` so a later brainstorm
    run can show the model which papers fed the memo (by index) and let it look up
    their originals on demand. Index order matches the ``[Pn]`` citation handles.
    Best-effort — never fails the memo.
    """
    try:
        entries = [
            {
                "index": f"P{i + 1}",
                "id": p.get("arxiv_id") or p.get("paper_id") or "",
                "source": p.get("source", ""),
                "title": p.get("title", ""),
                "authors": p.get("authors", ""),
                "venue": p.get("venue", ""),
                "url": p.get("url", ""),
                "abstract": p.get("abstract", ""),
                "summary": p.get("summary", ""),
                "key_insight": p.get("key_insight", ""),
                "math_concepts": p.get("math_concepts", []),
            }
            for i, p in enumerate(papers)
        ]
        path = Path(session_dir) / "insights_papers.json"
        path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("Wrote insights paper manifest (%d papers) to %s", len(entries), path)
        return path
    except Exception as e:
        log.warning("Failed to write insights paper manifest: %s", e)
        return None


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

    raw = call_cli(prompt, cfg, timeout=1200)
    if not raw:
        log.warning("Insights generation failed — no output from CLI")
        return None

    out_dir = Path(session_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "insights.md"
    path.write_text(raw, encoding="utf-8")
    _write_papers_manifest(out_dir, papers)
    log.info("Insights written to %s", path)
    return path


_CROSS_DOMAIN_PROMPT = """\
You are a research scientist hunting for cross-pollination between fields.

A colleague working on "{topic_name}" selected these papers:

{paper_list}

Propose {n} papers from OTHER fields — mathematics, statistics, physics, biology, \
economics, or other CS subfields — whose ideas, tools, or theory could meaningfully \
enrich, generalize, or connect to the selected work. Favor non-obvious but genuine bridges.

For each, give a focused arXiv search query (3-8 words) likely to surface such work.

Reply ONLY with a JSON array; each element:
{{"query": "<arxiv search query>", "domain": "<field/category, e.g. math.OC>", \
"rationale": "<one sentence: why this bridge matters>"}}"""


def _parse_json_array(raw: str) -> list:
    """Best-effort parse of a JSON array from an LLM reply (tolerates fences/prose)."""
    if not raw:
        return []
    text = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        pass
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            return []
    return []


def _parse_json_obj(raw: str) -> dict:
    """Best-effort parse of a JSON object from an LLM reply (tolerates fences/prose)."""
    if not raw:
        return {}
    text = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    text = re.sub(r"\s*```$", "", text)
    try:
        d = json.loads(text)
        return d if isinstance(d, dict) else {}
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                d = json.loads(m.group(0))
                return d if isinstance(d, dict) else {}
            except json.JSONDecodeError:
                return {}
    return {}


def suggest_cross_domain(
    selected_papers: list[dict],
    topic_name: str,
    cfg: dict,
    *,
    max_candidates: int = 12,
    max_queries: int = 6,
    per_query: int = 3,
) -> list[dict]:
    """Suggest cross-domain (math / other-field) papers to enrich the selection.

    Uses an LLM (opus) to propose bridge search queries, then runs each through
    arXiv. Returns candidate dicts (arxiv_id, title, authors, abstract, url,
    published, domain, rationale). Returns [] gracefully on LLM/search failure.
    """
    if not selected_papers:
        return []

    lines = []
    for p in selected_papers[:40]:
        bits = [f"- {p.get('title', '')}"]
        if p.get("key_insight"):
            bits.append(f"insight: {p['key_insight']}")
        if p.get("math_concepts"):
            bits.append(f"math: {', '.join(p['math_concepts'])}")
        lines.append(" | ".join(bits))

    prompt = _CROSS_DOMAIN_PROMPT.format(
        topic_name=topic_name, n=max_queries, paper_list="\n".join(lines)
    )
    bridges = _parse_json_array(call_cli(prompt, cfg, timeout=300) or "")
    if not bridges:
        log.warning("Cross-domain suggestion: no bridges proposed")
        return []

    selected_ids = {p.get("arxiv_id") for p in selected_papers}
    selected_ids |= {p.get("paper_id") for p in selected_papers}

    candidates: list[dict] = []
    seen: set[str] = set()
    for b in bridges[:max_queries]:
        if not isinstance(b, dict):
            continue
        query = (b.get("query") or "").strip()
        if not query:
            continue
        domain = (b.get("domain") or "").strip()
        rationale = (b.get("rationale") or "").strip()
        try:
            hits = arxiv.search_by_query(query, max_results=per_query)
        except Exception as e:  # network / parse — skip this bridge, keep the rest
            log.warning("Cross-domain search failed for %r: %s", query, e)
            continue
        for h in hits:
            aid = h.get("arxiv_id", "")
            if not aid or aid in seen or aid in selected_ids:
                continue
            seen.add(aid)
            candidates.append({
                "arxiv_id": aid,
                "title": h.get("title", ""),
                "authors": h.get("authors", ""),
                "abstract": h.get("abstract", ""),
                "url": h.get("url", ""),
                "published": (h.get("published", "") or "")[:10],
                "domain": domain,
                "rationale": rationale,
            })
            if len(candidates) >= max_candidates:
                return candidates
    return candidates


# ===========================================================================
# Agentic multi-stage insights pipeline (alternative to single-call generate())
# ===========================================================================
#
# Instead of stuffing every paper into one prompt, it:
#   1. outlines themes (soft clustering — a bridging paper may sit in several)
#   2. synthesizes each theme in parallel (focused, bounded context)
#   3. runs a dedicated CROSS-THEME connection pass so the most valuable
#      (non-obvious, span-multiple-themes) insights are NOT siloed by clustering
#   4. reduces to the final 5-section memo
#   5. audits it (reflection + deterministic citation grounding)
#   6. revises adaptively if the audit finds substantive issues
#
# Modeled on AutoSurvey (outline -> parallel -> integrate), SurveyG (group by
# theme, don't concatenate full text), SciSage (reflector), and MVSS (citation
# grounding) — adapted from survey-WRITING to insight-MINING (hence the extra
# cross-theme pass, which survey pipelines don't need). No single prompt ever
# holds all papers' full content. Every stage degrades gracefully so a transient
# LLM/egress failure yields a (possibly reduced) memo rather than nothing.

_MAX_WORKERS = 4         # parallel theme-synthesis workers
_FALLBACK_BATCH = 12     # papers per group when outlining fails (-> map-reduce)
_THEME_MAX = 18          # sub-batch a theme larger than this

_CITE_RE = re.compile(r"\[(P\d+)\]")


_OUTLINE_PROMPT = """\
You are organizing {n} papers in the field of "{topic_name}" to mine cross-paper \
INSIGHTS (trends, emerging methods, non-obvious connections, research gaps).

Group them into 3-7 coherent themes. A paper that bridges themes MAY appear in \
more than one. Give each theme a short title and a one-line analytical angle.

Papers (cite by the bracketed label):
{paper_list}

Reply ONLY with a JSON object:
{{"themes": [{{"title": "<short theme title>", "angle": "<one-line angle>", "papers": ["P1", "P5", ...]}}, ...]}}"""


_THEME_PROMPT = """\
You are a senior researcher in "{topic_name}" writing a focused brief on ONE theme \
for a cross-paper insights memo.

Theme: {title}
Angle: {angle}

Papers in this theme (cite using the [Pn] labels):
{cards}

Write a concise Markdown brief covering, for THIS theme only:
- the key trend(s) and what concretely drives them (name techniques / metrics)
- notable or novel methods (note if papers independently converge)
- internal connections, comparisons, and any contradictions between these papers
- 1-2 concrete gaps / opportunities this theme exposes
- which 1-2 papers here are must-reads and why
Cite every claim with the relevant [Pn]. Be specific, not generic."""


_CONNECTIONS_PROMPT = """\
You are mining NON-OBVIOUS connections that SPAN different themes in a body of work \
on "{topic_name}" — exactly the cross-cutting insights that get lost when papers are \
analyzed theme-by-theme.

Per-theme briefs already written:
{briefs}

Full paper list (nothing here is invisible to you; cite by [Pn]):
{paper_list}

Find connections that cross theme boundaries. Look specifically for:
- the same problem solved differently in different themes (compare them)
- a method in one theme that could fill a gap in another
- a shared mathematical / conceptual framework recurring across contexts
- contradictions or tensions between themes' findings
Reply with a concise Markdown list of 3-6 cross-theme connections, each citing the \
relevant [Pn]. If you genuinely find none, reply with exactly "None found"."""


_REDUCE_PROMPT = """\
You are a senior research scientist writing the final cross-paper INSIGHTS memo for \
your lab in "{topic_name}", integrating the analysis below.

Per-theme briefs:
{briefs}

Cross-theme connections:
{connections}

Full paper list (for reference; cite by [Pn]):
{paper_list}

Write the final memo in Markdown with EXACTLY these sections:

## Key Trends
3-5 significant trends across all papers; for each: what it is, which [Pn] exemplify it, why it matters. Be specific.

## Emerging Methods
2-4 novel techniques / patterns; what problem each solves, briefly how it works, potential impact; flag independent convergence.

## Connections & Cross-Paper Themes
The non-obvious links — prioritize the CROSS-THEME connections above (same problem / different approach, combinable results, contradictions, shared frameworks). Cite [Pn].

## Research Gaps & Opportunities
2-3 concrete, actionable opportunities grounded in what the papers do and don't address; for each: the gap, why it matters, a possible approach.

## Recommended Reading
Rank the top 3 must-read papers ([Pn]) and justify each (novelty, results, follow-up potential, relevance).

Cite claims with [Pn]. Do not invent papers or use labels outside the list."""


_AUDIT_PROMPT = """\
You are a critical reviewer auditing an insights memo for "{topic_name}" before release.

{invalid_note}Memo:
{memo}

Papers actually available (cite by [Pn]):
{paper_list}

Audit for: (a) structural issues — a major theme missing, over-generalized / vague \
claims, redundancy, shallow gaps; (b) grounding — claims not supported by the cited \
paper(s), or citations to labels that don't exist.

Reply ONLY with a JSON object:
{{"ok": true|false, "issues": ["<concrete, actionable fix>", ...]}}
Set "ok" true only if the memo needs no substantive changes."""


_REVISE_PROMPT = """\
Revise this insights memo for "{topic_name}", fixing the issues below. Keep the five \
section headers and the [Pn] citation style. Only use labels that exist in the paper \
list. Output the full revised memo in Markdown — nothing else.

Issues to fix:
{issues}

Paper list (valid [Pn] labels):
{paper_list}

Current memo:
{memo}"""


def _paper_oneliner(label: str, p: dict) -> str:
    """Compact one-line representation (used wherever ALL papers must be visible)."""
    hint = (p.get("key_insight") or p.get("summary") or p.get("abstract") or "").strip()
    hint = re.sub(r"\s+", " ", hint)[:160]
    title = (p.get("title") or "Untitled").strip()
    return f"[{label}] {title}" + (f" — {hint}" if hint else "")


def _paper_card(label: str, p: dict) -> str:
    """Full distilled card (only fed into the paper's own theme synthesis)."""
    head = f"[{label}] **{p.get('title', 'Untitled')}**"
    if p.get("venue"):
        head += f" ({p['venue']})"
    parts = [head]
    for field, lbl in (("key_insight", "Key insight"), ("method", "Method"),
                       ("contribution", "Contribution"), ("summary", "Summary")):
        if p.get(field):
            parts.append(f"  {lbl}: {p[field]}")
    if p.get("math_concepts"):
        parts.append(f"  Math: {', '.join(p['math_concepts'])}")
    return "\n".join(parts)


def _ordered_labels(text: str) -> list[str]:
    """Distinct [Pn] labels in first-appearance order."""
    seen: list[str] = []
    s: set[str] = set()
    for m in _CITE_RE.findall(text):
        if m not in s:
            s.add(m)
            seen.append(m)
    return seen


def _outline(labeled, id_map, topic_name, cfg):
    """Stage 1: theme outline with soft assignment. Falls back to fixed batches.

    Always returns a non-empty list of ``{title, angle, labels}`` covering every
    paper (un-themed papers are swept into an "Additional papers" group).
    """
    paper_list = "\n".join(_paper_oneliner(lbl, p) for lbl, p in labeled)
    raw = call_cli(
        _OUTLINE_PROMPT.format(n=len(labeled), topic_name=topic_name, paper_list=paper_list),
        cfg, timeout=600,
    )
    obj = _parse_json_obj(raw or "")
    themes = []
    for t in (obj.get("themes") or []):
        if not isinstance(t, dict):
            continue
        labels = [str(x) for x in (t.get("papers") or []) if str(x) in id_map]
        if not labels:
            continue
        themes.append({
            "title": (t.get("title") or "Theme").strip() or "Theme",
            "angle": (t.get("angle") or "").strip(),
            "labels": labels,
        })

    if not themes:  # outline unavailable -> degrade to fixed-batch map-reduce
        log.warning("Insights outline unavailable — falling back to fixed batches")
        all_labels = [lbl for lbl, _ in labeled]
        return [
            {"title": f"Group {i + 1}", "angle": "", "labels": all_labels[s:s + _FALLBACK_BATCH]}
            for i, s in enumerate(range(0, len(all_labels), _FALLBACK_BATCH))
        ]

    # ensure full coverage: papers in no theme go into an "Additional papers" group
    covered = {l for t in themes for l in t["labels"]}
    missing = [lbl for lbl, _ in labeled if lbl not in covered]
    for i, s in enumerate(range(0, len(missing), _FALLBACK_BATCH)):
        themes.append({
            "title": "Additional papers" if i == 0 else f"Additional papers {i + 1}",
            "angle": "", "labels": missing[s:s + _FALLBACK_BATCH],
        })
    return themes


def _synthesize_theme(theme, id_map, topic_name, cfg):
    """Stage 2: one theme brief. Sub-batches very large themes; stub on failure."""
    labels = theme["labels"]
    title = theme["title"]
    briefs = []
    for s in range(0, len(labels), _THEME_MAX):
        chunk = labels[s:s + _THEME_MAX]
        cards = "\n\n".join(_paper_card(l, id_map[l]) for l in chunk)
        raw = call_cli(
            _THEME_PROMPT.format(topic_name=topic_name, title=title,
                                 angle=theme.get("angle", ""), cards=cards),
            cfg, timeout=600,
        )
        if raw and raw.strip():
            briefs.append(raw.strip())
    if briefs:
        return f"### {title}\n\n" + "\n\n".join(briefs)
    # deterministic stub so the reduce step still has material from this theme
    bullets = "\n".join(
        f"- [{l}] {id_map[l].get('title', '')}: "
        f"{(id_map[l].get('key_insight') or id_map[l].get('summary') or '').strip()}".rstrip(": ")
        for l in labels
    )
    return f"### {title}\n\n{bullets}"


def _fallback_memo(topic_name, briefs_text, connections):
    """If the reduce call fails, assemble a non-empty memo from the briefs."""
    parts = [f"# Insights: {topic_name}", ""]
    if connections:
        parts += ["## Connections & Cross-Paper Themes", connections, ""]
    parts += ["## Themes", briefs_text]
    return "\n".join(parts)


def _audit_and_revise(memo, id_map, topic_name, paper_list, cfg, progress):
    """Stages 4-5: reflect + ground-check, then revise only if needed."""
    cited = set(_CITE_RE.findall(memo))
    invalid = sorted(cited - set(id_map))
    invalid_note = (
        f"NOTE: these cited labels do NOT exist and must be removed: {', '.join(invalid)}\n\n"
        if invalid else ""
    )
    raw = call_cli(
        _AUDIT_PROMPT.format(topic_name=topic_name, memo=memo,
                             paper_list=paper_list, invalid_note=invalid_note),
        cfg, timeout=600,
    )
    audit = _parse_json_obj(raw or "")
    issues = [str(x).strip() for x in (audit.get("issues") or []) if str(x).strip()]
    if invalid:  # hard grounding rule — always fix hallucinated citations
        issues.append(f"Remove citations to non-existent labels: {', '.join(invalid)}")

    audit_ok = bool(audit.get("ok", True))
    if (audit_ok and not invalid) or not issues:
        return memo  # clean, or audit gave nothing actionable -> keep the draft

    progress("revising per audit")
    revised = call_cli(
        _REVISE_PROMPT.format(topic_name=topic_name,
                              issues="\n".join(f"- {i}" for i in issues),
                              paper_list=paper_list, memo=memo),
        cfg, timeout=1200,
    )
    return (revised or "").strip() or memo


def _append_references(memo, id_map):
    """Append a ## References section mapping cited [Pn] -> title (venue) — url."""
    cited = [l for l in _ordered_labels(memo) if l in id_map]
    if not cited:
        return memo
    lines = ["", "## References", ""]
    for lbl in cited:
        p = id_map[lbl]
        title = p.get("title", "Untitled")
        venue = f" ({p['venue']})" if p.get("venue") else ""
        url = f" — {p['url']}" if p.get("url") else ""
        lines.append(f"- [{lbl}] {title}{venue}{url}")
    return memo.rstrip() + "\n" + "\n".join(lines) + "\n"


def generate_agentic(
    papers: list[dict],
    topic_name: str,
    session_dir: str | Path,
    cfg: dict,
    *,
    progress_cb=None,
) -> Path | None:
    """Multi-stage agentic insights pipeline. Returns insights.md path or None.

    Drop-in alternative to :func:`generate` (same first four args), selected via
    the route's ``mode="agentic"``. See the module-level comment for the design.
    """
    if not papers:
        log.info("No papers — skipping agentic insights generation")
        return None

    def _progress(msg):
        if progress_cb:
            try:
                progress_cb(msg)
            except Exception:  # progress reporting must never break generation
                pass
        log.info("Insights[agentic]: %s", msg)

    labeled = [(f"P{i + 1}", p) for i, p in enumerate(papers)]
    id_map = {lbl: p for lbl, p in labeled}
    paper_list = "\n".join(_paper_oneliner(lbl, p) for lbl, p in labeled)

    # Stage 1 — outline / soft clustering
    _progress(f"outlining themes over {len(papers)} papers")
    themes = _outline(labeled, id_map, topic_name, cfg)

    # Stage 2 — parallel per-theme synthesis
    _progress(f"synthesizing {len(themes)} themes in parallel")
    briefs: list[str | None] = [None] * len(themes)
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        futs = {ex.submit(_synthesize_theme, t, id_map, topic_name, cfg): i
                for i, t in enumerate(themes)}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                briefs[i] = fut.result()
            except Exception as e:  # one theme failing must not sink the run
                log.warning("Theme synthesis failed (%s): %s", themes[i]["title"], e)
                briefs[i] = f"### {themes[i]['title']}\n\n(synthesis unavailable)"
    briefs_text = "\n\n".join(b for b in briefs if b)

    # Stage 2.5 — cross-theme connection pass (the anti-silo step)
    connections = ""
    if len(themes) > 1:
        _progress("mining cross-theme connections")
        raw = call_cli(
            _CONNECTIONS_PROMPT.format(topic_name=topic_name, briefs=briefs_text,
                                       paper_list=paper_list),
            cfg, timeout=600,
        )
        connections = (raw or "").strip()
        if connections.lower().rstrip(".") == "none found":
            connections = ""

    # Stage 3 — reduce / integrate
    _progress("integrating final memo")
    memo = call_cli(
        _REDUCE_PROMPT.format(topic_name=topic_name, briefs=briefs_text,
                              connections=connections or "(none surfaced)",
                              paper_list=paper_list),
        cfg, timeout=1200,
    )
    memo = (memo or "").strip()
    if not memo:
        log.warning("Insights reduce failed — assembling fallback memo from briefs")
        memo = _fallback_memo(topic_name, briefs_text, connections)

    # Stages 4-5 — audit (reflection + grounding) then adaptive revise
    _progress("auditing (reflection + grounding)")
    memo = _audit_and_revise(memo, id_map, topic_name, paper_list, cfg, _progress)

    # Stage 6 — references + write
    memo = _append_references(memo, id_map)
    out_dir = Path(session_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "insights.md"
    path.write_text(memo, encoding="utf-8")
    _write_papers_manifest(out_dir, papers)
    _progress("done")
    log.info("Agentic insights written to %s", path)
    return path
