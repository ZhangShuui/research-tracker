# Early Relevance Prefilter + Multi-turn Response Collapse Topic — Design Spec

## Problem

Two related issues motivate this change:

1. **Wasted summarizer cost**: the pipeline runs per-paper structured extraction (`summarizer.summarize_papers`, sonnet, ~1 call/paper) *before* relevance filtering. For a topic like "LLM multi-turn response collapse", the keyword `mode collapse` also matches many GAN/diffusion papers — all of them pay the full extraction cost and only get rejected afterwards by `filter_papers_by_quality`.
2. **Model tier drift**: `config.toml` and `config.py` currently default the summarizer to `opus`. The user wants the "subagent tier" (batch, per-paper) to use sonnet, reserving opus for synthesis steps (report, insights, brainstorm, research_plan, deep_read, chat).

This spec also defines the first of three planned LLM-collapse topics (A, B, C); only topic A is created now.

## Solution

Three independent changes, done in order:

1. Point the summarizer subagent at sonnet.
2. Insert a coarse relevance **prefilter** between dedup and summarize that reads only title + abstract, runs batched sonnet calls, and drops off-topic papers before they reach the expensive per-paper summarizer.
3. Create topic A ("LLM Multi-turn Response Collapse") with `prefilter_criteria` that explicitly rejects GAN / image-generation uses of "mode collapse".

Topics B (RLHF diversity loss) and C (model collapse from synthetic training data) are **out of scope** for this spec and will be created in later specs once topic A has produced a first session.

## Non-goals

- No changes to `report.py`, `insights.py`, `brainstorm.py`, `research_plan.py`, `deep_read.py`, `chat.py`, `discovery.py` — these stay on opus.
- No change to the existing post-summarize `filter_papers_by_quality` step. Prefilter is **additive**, not a replacement.
- No retroactive re-filter of already-stored papers.
- Frontend changes to the Topic edit form are optional and deferred (backend + script creation is enough to ship topic A).

## Change 1 — Summarizer model → sonnet

Two edits:

- `config.toml` line 25: `claude_model = "opus"` → `claude_model = "sonnet"`
- `src/paper_tracker/config.py` `_default_summarizer()` line 80: `"claude_model": "opus"` → `"claude_model": "sonnet"`

Other call sites already specify their model explicitly (opus for synthesis paths, sonnet for a JSON-repair step in brainstorm.py:3788), so these two edits are sufficient.

## Change 2 — Early relevance prefilter

### Pipeline placement

Insert between step 2 (dedup) and step 3 (summarize) in `main.py`:

```
fetch → dedup → PREFILTER (new) → summarize → quality_filter → save → report → insights
```

Progress stage: `prefiltering`.

### Function signature

New function in `src/paper_tracker/summarizer.py`:

```python
def prefilter_by_relevance(
    papers: list[dict],
    cfg: dict,
    topic_name: str,
    description: str = "",
    criteria: str = "",
    keywords: list[str] | None = None,
    batch_size: int = 25,
) -> list[dict]:
    """Drop papers whose title+abstract is clearly off-topic.

    Uses sonnet on (title + abstract) only, batched. Cheaper than summarize
    (~20x) and runs before it. Papers that fail to be scored are kept
    (benefit-of-doubt, matching filter_papers_by_quality).

    Returns the subset of papers that passed (and the dropped ones get
    a 'prefilter_reason' attached for logging only — not persisted).
    """
```

### Prompt shape

```
You are screening papers for a research library on "{topic_name}".
Description: {description}
Keywords: {keywords}

{criteria_section}  # present only if criteria is non-empty

For each paper below (title + abstract), answer: is this paper AT LEAST
plausibly about the topic above? Err on the side of keeping borderline
papers — a later pass will re-evaluate more carefully.

Reply ONLY with a JSON array. Each object: {"id": str, "relevant": bool,
"reason": str (<=12 words)}. No markdown fences.

Papers:
{items}  # [id] title \n abstract (truncated to 600 chars) per paper
```

- Abstract truncated to 600 chars in the prompt (vs 800 in quality filter) — cheaper, and the prompt is binary so less context is needed.
- `items` uses the same `[id] title / Abstract:` layout as `filter_papers_by_quality`, so the JSON-parse helper `_parse_json_array` can be reused.

### Failure modes

- LLM returns nothing / invalid JSON → **keep all papers in that batch** (benefit of doubt).
- Individual paper not mentioned in the returned JSON → **keep**.
- `relevant = false` with missing/empty reason → still drop, log with reason `""`.

### Logging

At INFO level: `"Prefilter removed X/Y papers (kept Z)"`. At DEBUG level: one line per dropped paper with its reason.

### Config plumbing

Topic registry gains two new fields:

| Field | Type | Default | Meaning |
|---|---|---|---|
| `prefilter_enabled` | bool | `true` | Master switch. If false, step is skipped. |
| `prefilter_criteria` | str | `""` | Extra constraints interpolated into the prompt (see prompt shape — appears as `{criteria_section}`). Empty = derived from description + keywords only. |

Propagation:

- `server.py` `TopicCreate` / `TopicUpdate`: add `prefilter_enabled` and `prefilter_criteria` fields.
- `server.py` `quick_create_topic`: no Pydantic change (body is still just `name`), but the registry insert dict at the bottom of the function gains the two fields with their defaults (`True`, `""`).
- `registry.py`: add columns; reuse the existing migration pattern (the `Registry` class already auto-migrates; follow that pattern).
- `config.py` `from_topic()`: add `"description"`, `"prefilter_enabled"`, `"prefilter_criteria"` to the `search` sub-dict.
- `main.py`: call `prefilter_by_relevance` only if `search_cfg.get("prefilter_enabled", True)`.

### Cost estimate

For a raw batch of 500 papers:

| Stage | Before | After |
|---|---|---|
| Prefilter | — | 20 sonnet calls (batch=25) |
| Summarize (per-paper sonnet) | 500 | ~200 (assuming 60% drop rate) |
| Quality filter (batch=10, opus) | 50 | ~20 |

Net: roughly 60% reduction in summarizer calls on topics with a lot of cross-domain keyword noise like "mode collapse".

## Change 3 — Topic A

Creation via a committed script `scripts/create_topic_multiturn_collapse.py` that POSTs to `/api/topics`. Keeping the config in git makes re-creation and tweaks auditable.

### Topic A config

| Field | Value |
|---|---|
| `name` | `LLM Multi-turn Response Collapse` |
| `description` | `Investigates how LLM responses degrade in multi-turn conversations — response template convergence, persona drift, diversity loss, and sycophancy-induced mode collapse.` |
| `schedule_cron` | *(empty, manual runs)* |
| `enabled` | `true` |
| `search_date_from` | `2024-04-21` |
| `search_date_to` | `2026-04-21` |
| `arxiv_categories` | `["cs.CL", "cs.AI", "cs.LG"]` |
| `arxiv_keywords` | `["mode collapse", "persona drift", "sycophancy", "response diversity", "multi-turn dialogue", "dialogue degradation", "output homogeneity"]` |
| `arxiv_lookback_days` | `730` (falls back if date range ignored) |
| `github_keywords` | `["LLM diversity evaluation", "multi-turn benchmark", "dialogue evaluation"]` |
| `github_lookback_days` | `730` |
| `openalex_enabled` | `true` |
| `openalex_keywords` | same as arxiv_keywords |
| `openalex_venues` | `[]` (rely on date + keyword filter) |
| `openalex_max_results` | `200` |
| `openreview_enabled` | `true` |
| `openreview_venues` | `["iclr2025", "iclr2024", "neurips2025", "neurips2024", "acl2024", "acl2025", "emnlp2024", "emnlp2025"]` |
| `openreview_keywords` | `["mode collapse", "response diversity", "persona drift", "sycophancy"]` |
| `openreview_max_results` | `100` |
| `prefilter_enabled` | `true` |
| `prefilter_criteria` | *(see below)* |

### `prefilter_criteria` for topic A

```
The paper must be about Large Language Models, LLM-based dialogue systems, or
conversational AI. Papers about GANs, image generation, VAEs, or non-language
mode collapse should be excluded even if they mention "mode collapse" or
"diversity". Papers that use LLMs only as a tool (e.g. as an evaluator) are
OUT unless the paper's core contribution is about LLM output behavior.
```

### Known trade-offs

- Keyword `mode collapse` will match GAN papers on arXiv; the prefilter catches these.
- Keywords `sycophancy`, `persona drift` are narrow enough that recall may be low; `response diversity` + `multi-turn dialogue` broaden the net.
- Topics B (RLHF diversity loss, alignment tax) and C (model collapse on synthetic data) are deliberately excluded to avoid paper double-counting across topics.

## Implementation order

1. Change 1 (model config, 2 edits) — commit standalone.
2. Change 2 (prefilter function + pipeline wiring + registry migration + config plumbing) — commit standalone.
3. Change 3 (topic creation script + run it) — commit standalone.
4. First session run of topic A, review output, tune keywords if needed (not part of this spec).

Frontend edits to expose `prefilter_enabled` / `prefilter_criteria` in the Topic edit form are deferred.

## Testing

- Unit test for `prefilter_by_relevance` with a mocked `call_cli` returning known JSON — assert that `relevant=false` papers are dropped and `relevant=true` / unscored papers are kept.
- Integration smoke: run the pipeline end-to-end on topic A; verify `prefiltering` stage appears in progress log and some papers are dropped with a visible reason.
- No test for the topic-creation script beyond "POST returns 201 and topic appears in `GET /api/topics`".
