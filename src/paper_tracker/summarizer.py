"""Batch summarizer: Claude CLI → Codex CLI → truncation fallback.

Instead of one CLI call per item, we batch all items into a single prompt
to drastically reduce wall-clock time.

Prompt design informed by:
- Elicit's structured data extraction (per-field targeted questions)
- Chain-of-Ideas (CoI) progressive analysis structure
- Academic systematic review best practices
"""

from __future__ import annotations

import json
import logging
import os
import subprocess

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paper analysis prompt — Elicit-style structured extraction
# ---------------------------------------------------------------------------

_BATCH_PAPER_PROMPT = """\
You are a senior research analyst conducting a systematic review. \
For each paper below, perform careful structured extraction.

IMPORTANT: Read each abstract thoroughly. Do NOT hallucinate information not present \
in the abstract. If a field cannot be determined from the abstract alone, use "" for \
strings or [] for lists.

For each paper, extract:

1. "summary": A precise 2-3 sentence summary covering: (a) the problem addressed, \
(b) the proposed approach, and (c) the main result or finding. Be specific — include \
quantitative results when mentioned.

2. "key_insight": The single most important finding or contribution in one sentence. \
What would a researcher remember about this paper? Focus on what is NEW, not what is \
well-known.

3. "method": The core technical approach in one sentence. Name specific architectures, \
algorithms, or techniques (e.g. "Diffusion Transformer with cross-attention conditioning" \
not just "deep learning"). Include the key design choice that differentiates this method.

4. "contribution": What this paper adds relative to prior work, in one sentence. \
Frame as "Unlike X which does Y, this paper Z" or "First to achieve X" or \
"Extends X by adding Y".

5. "math_concepts": Up to 5 key mathematical concepts, loss functions, or formal \
notations central to the method (e.g. ["classifier-free guidance scale w", \
"DDPM forward process q(x_t|x_0)", "FID score", "DiT block architecture"]). \
Use precise notation where possible. Return [] if the abstract is not technical enough.

6. "venue": Publication venue and year if mentioned or clearly inferable from the \
abstract text (e.g. "NeurIPS 2025", "CVPR 2025"). Return "" if unknown — do NOT guess.

7. "cited_works": Up to 5 notable prior works referenced in the abstract, formatted as \
"Author et al. Year (ShortTitle)" (e.g. "Ho et al. 2020 (DDPM)"). Return [] if none \
are explicitly mentioned.

Reply ONLY with a JSON array. No markdown fences, no explanation, no extra text.
Each object must have keys: "id", "summary", "key_insight", "method", "contribution", \
"math_concepts", "venue", "cited_works".

Papers:
{items}"""

# ---------------------------------------------------------------------------
# Repo summary prompt
# ---------------------------------------------------------------------------

_BATCH_REPO_PROMPT = """\
You are a research engineer. For each GitHub repository below, write a concise \
1-sentence summary explaining: what problem it solves, what technique/framework it \
uses, and who would benefit from it.

Reply ONLY with a JSON array of objects, each with "id" and "summary" keys. \
No markdown fences, no extra text.

Repos:
{items}"""

_BATCH_SIZE = 20  # max items per CLI call to stay within context limits


def _call_cli(prompt: str, cfg: dict) -> str | None:
    """Try Claude CLI first, then Codex CLI. Returns raw stdout or None."""
    scfg = cfg["summarizer"]

    # --- Claude CLI ---
    claude_path = scfg.get("claude_path", "claude")
    claude_model = scfg.get("claude_model", "sonnet")
    timeout = scfg.get("claude_timeout", 120)
    timeout = max(timeout, 120)

    env = os.environ.copy()
    env.pop("CLAUDE_CODE_ENTRYPOINT", None)
    env.pop("CLAUDECODE", None)

    try:
        cmd = [claude_path, "-p", "-"]
        if claude_model:
            cmd.extend(["--model", claude_model])
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True, text=True, timeout=timeout, env=env,
        )
        if result.returncode == 0 and result.stdout.strip():
            log.info("Batch summarized via Claude CLI")
            return result.stdout.strip()
        log.warning("Claude CLI returned code %d", result.returncode)
    except FileNotFoundError:
        log.warning("Claude CLI not found at '%s'", claude_path)
    except subprocess.TimeoutExpired:
        log.warning("Claude CLI timed out after %ds", timeout)

    # --- Codex CLI ---
    codex_path = scfg.get("codex_path", "codex")
    codex_timeout = scfg.get("codex_timeout", 30)
    codex_timeout = max(codex_timeout, 120)

    env2 = os.environ.copy()
    env2.pop("OPENAI_API_KEY", None)
    env2.pop("OPENAI_BASE_URL", None)

    try:
        result = subprocess.run(
            [codex_path, "exec", prompt],
            capture_output=True, text=True, timeout=codex_timeout, env=env2,
        )
        if result.returncode == 0 and result.stdout.strip():
            log.info("Batch summarized via Codex CLI")
            return result.stdout.strip()
        log.warning("Codex CLI returned code %d", result.returncode)
    except FileNotFoundError:
        log.warning("Codex CLI not found at '%s'", codex_path)
    except subprocess.TimeoutExpired:
        log.warning("Codex CLI timed out after %ds", codex_timeout)

    return None


def _parse_json_array(raw: str) -> list[dict]:
    """Best-effort parse a JSON array from LLM output (handles markdown fences)."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        arr = json.loads(text)
        if isinstance(arr, list):
            return arr
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    log.warning("Failed to parse JSON array from CLI output")
    return []


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


_PAPER_FIELDS = ("summary", "key_insight", "method", "contribution",
                 "math_concepts", "venue", "cited_works")


def summarize_papers(papers: list[dict], cfg: dict) -> None:
    """Summarize papers in-place using batch CLI calls. Mutates paper dict fields."""
    if not papers:
        return
    max_len = cfg["summarizer"].get("truncation_length", 300)

    for batch_start in range(0, len(papers), _BATCH_SIZE):
        batch = papers[batch_start : batch_start + _BATCH_SIZE]
        items_text = "\n".join(
            f"[{p.get('paper_id', p['arxiv_id'])}] {p['title']}: {p['abstract'][:500]}"
            for p in batch
        )
        prompt = _BATCH_PAPER_PROMPT.format(items=items_text)

        raw = _call_cli(prompt, cfg)
        if raw:
            parsed = _parse_json_array(raw)
            lookup = {str(d.get("id", "")): d for d in parsed}
            for p in batch:
                pid = p.get("paper_id", p["arxiv_id"])
                if pid in lookup:
                    entry = lookup[pid]
                    if entry.get("summary"):
                        p["summary"] = entry["summary"]
                    p["key_insight"] = entry.get("key_insight") or ""
                    p["method"] = entry.get("method") or ""
                    p["contribution"] = entry.get("contribution") or ""
                    p["math_concepts"] = entry.get("math_concepts") or []
                    p["venue"] = entry.get("venue") or ""
                    p["cited_works"] = entry.get("cited_works") or []

        # Fallback: ensure all fields exist
        for p in batch:
            if not p.get("summary"):
                p["summary"] = _truncate(p["abstract"], max_len)
            for field in ("key_insight", "method", "contribution"):
                if field not in p:
                    p[field] = ""
            for field in ("math_concepts", "cited_works"):
                if field not in p:
                    p[field] = []
            if "venue" not in p:
                p["venue"] = ""


def summarize_repos(repos: list[dict], cfg: dict) -> None:
    """Summarize repos in-place using batch CLI calls. Mutates repo['summary']."""
    if not repos:
        return
    max_len = cfg["summarizer"].get("truncation_length", 300)

    for batch_start in range(0, len(repos), _BATCH_SIZE):
        batch = repos[batch_start : batch_start + _BATCH_SIZE]
        items_text = "\n".join(
            f"[{r['repo_full_name']}] {r['description'][:300]}"
            for r in batch
        )
        prompt = _BATCH_REPO_PROMPT.format(items=items_text)

        raw = _call_cli(prompt, cfg)
        if raw:
            parsed = _parse_json_array(raw)
            lookup = {str(d.get("id", "")): d.get("summary", "") for d in parsed}
            for r in batch:
                if r["repo_full_name"] in lookup and lookup[r["repo_full_name"]]:
                    r["summary"] = lookup[r["repo_full_name"]]

        for r in batch:
            if not r.get("summary"):
                r["summary"] = _truncate(r["description"], max_len)


# ---------------------------------------------------------------------------
# Paper quality filtering
# ---------------------------------------------------------------------------

_QUALITY_FILTER_BATCH_SIZE = 10  # smaller batches for more attention per paper

_QUALITY_FILTER_PROMPT = """\
You are a senior ML researcher reviewing papers for a research survey on "{topic_name}".
The survey focuses on these keywords: {keywords}

For each paper below, assess on THREE dimensions:
1. **Relevance** to "{topic_name}" (is it directly about this topic, or tangential?)
2. **Methodological rigor** (is the approach well-motivated, technically sound?)
3. **Novelty** (does it propose something genuinely new, or is it incremental?)

Score each paper 1-5:
- 5: Top-tier venue quality, highly relevant, novel contribution with rigorous methodology
- 4: Good quality, clearly relevant, solid methodology with meaningful contribution
- 3: Acceptable quality, somewhat relevant, standard methodology
- 2: Low quality or marginally relevant (tangentially related, weak methodology, \
incremental with no clear advance)
- 1: Not relevant, very low quality, or appears to be spam/placeholder

IMPORTANT CRITERIA:
- Papers from top venues (NeurIPS, ICML, ICLR, CVPR, ECCV, AAAI, ACL, EMNLP) get +1 boost
- Papers that are only tangentially related to "{topic_name}" should score ≤2
- Papers with no clear technical contribution should score ≤2
- Workshop papers or preprints with vague claims should score ≤3
- Papers with clear methodology AND clear results should score ≥3
- Survey/review papers should score ≥3 only if they cover "{topic_name}" directly

Reply ONLY with a JSON array. Each object must have:
- "id": the arxiv_id
- "quality": integer 1-5
- "rationale": one sentence explaining the score

No markdown fences, no extra text.

Papers:
{items}"""


def filter_papers_by_quality(
    papers: list[dict],
    cfg: dict,
    topic_name: str,
    min_quality: int = 3,
    keywords: list[str] | None = None,
) -> list[dict]:
    """Score papers on quality/relevance and filter out low-quality ones.

    Adds 'quality_score' field to each paper. Returns only papers with
    quality >= min_quality. Papers that fail to get scored are kept (benefit of doubt).
    """
    from paper_tracker.llm import call_cli as llm_call_cli

    if not papers:
        return papers

    kw_str = ", ".join(keywords) if keywords else topic_name
    log.info("Quality filtering %d papers for topic '%s' (min_quality=%d)",
             len(papers), topic_name, min_quality)

    # Score in smaller batches with more context per paper
    for batch_start in range(0, len(papers), _QUALITY_FILTER_BATCH_SIZE):
        batch = papers[batch_start : batch_start + _QUALITY_FILTER_BATCH_SIZE]
        items_text = "\n".join(
            f"[{p.get('paper_id', p['arxiv_id'])}] {p['title']}\n"
            f"  Abstract: {p.get('abstract', '')[:800]}\n"
            f"  Method: {p.get('method', '')}\n"
            f"  Contribution: {p.get('contribution', '')}"
            for p in batch
        )
        prompt = _QUALITY_FILTER_PROMPT.format(
            topic_name=topic_name, keywords=kw_str, items=items_text
        )

        raw = llm_call_cli(prompt, cfg, model="opus", timeout=180)
        if raw:
            parsed = _parse_json_array(raw)
            lookup = {str(d.get("id", "")): d for d in parsed}
            for p in batch:
                pid = p.get("paper_id", p["arxiv_id"])
                entry = lookup.get(pid)
                if entry is not None:
                    score = entry.get("quality", 3)
                    p["quality_score"] = max(1, min(5, int(score)))

        # Default: unscored papers get benefit of doubt
        for p in batch:
            if "quality_score" not in p:
                p["quality_score"] = 3

    # Filter
    kept = [p for p in papers if p["quality_score"] >= min_quality]
    removed = len(papers) - len(kept)
    if removed:
        log.info("Quality filter removed %d/%d papers (kept %d with score >= %d)",
                 removed, len(papers), len(kept), min_quality)
        for p in papers:
            if p["quality_score"] < min_quality:
                log.debug("  Filtered out: [%s] %s (score=%d)",
                          p["arxiv_id"], p["title"][:60], p["quality_score"])
    else:
        log.info("Quality filter: all %d papers passed (score >= %d)", len(papers), min_quality)

    return kept


# ---------------------------------------------------------------------------
# Re-filter existing papers
# ---------------------------------------------------------------------------

_REFILTER_PROMPT = """\
You are a senior ML researcher re-evaluating papers in a research library on "{topic_name}".
Keywords: {keywords}

{custom_section}

For each paper below, re-assess its quality and relevance. Score each paper 1-5:
- 5: Top-tier venue quality, highly relevant, novel contribution with rigorous methodology
- 4: Good quality, clearly relevant, solid methodology with meaningful contribution
- 3: Acceptable quality, somewhat relevant, standard methodology
- 2: Low quality or marginally relevant
- 1: Not relevant, very low quality

CRITERIA:
- Relevance to "{topic_name}" is the primary factor
- Methodological rigor and novelty are secondary factors
- Top venues get +1 boost
- Tangentially related papers should score ≤2

Reply ONLY with a JSON array. Each object must have:
- "id": the arxiv_id
- "quality": integer 1-5
- "rationale": one sentence explaining the score

No markdown fences, no extra text.

Papers:
{items}"""


def refilter_papers(
    papers: list[dict],
    cfg: dict,
    topic_name: str,
    keywords: list[str] | None = None,
    custom_instructions: str = "",
    on_batch_done: callable | None = None,
) -> list[dict]:
    """Re-score existing papers. Returns papers with updated quality_score.

    Does NOT filter — caller decides what to do with scores.
    on_batch_done(processed_so_far) is called after each batch completes.
    """
    from paper_tracker.llm import call_cli as llm_call_cli

    if not papers:
        return papers

    kw_str = ", ".join(keywords) if keywords else topic_name
    custom_section = ""
    if custom_instructions:
        custom_section = (
            f"CUSTOM INSTRUCTIONS (highest priority):\n{custom_instructions}\n"
        )

    log.info("Re-filtering %d papers for topic '%s'", len(papers), topic_name)

    processed = 0
    for batch_start in range(0, len(papers), _QUALITY_FILTER_BATCH_SIZE):
        batch = papers[batch_start : batch_start + _QUALITY_FILTER_BATCH_SIZE]
        items_text = "\n".join(
            f"[{p.get('paper_id', p['arxiv_id'])}] {p['title']}\n"
            f"  Abstract: {p.get('abstract', '')[:800]}\n"
            f"  Method: {p.get('method', '')}\n"
            f"  Contribution: {p.get('contribution', '')}"
            for p in batch
        )
        prompt = _REFILTER_PROMPT.format(
            topic_name=topic_name,
            keywords=kw_str,
            custom_section=custom_section,
            items=items_text,
        )

        raw = llm_call_cli(prompt, cfg, model="opus", timeout=180)
        if raw:
            parsed = _parse_json_array(raw)
            lookup = {str(d.get("id", "")): d for d in parsed}
            for p in batch:
                pid = p.get("paper_id", p["arxiv_id"])
                entry = lookup.get(pid)
                if entry is not None:
                    score = entry.get("quality", p.get("quality_score", 3))
                    p["quality_score"] = max(1, min(5, int(score)))

        processed += len(batch)
        if on_batch_done:
            on_batch_done(processed)

    log.info("Re-filtering complete for %d papers", len(papers))
    return papers
