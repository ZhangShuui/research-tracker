"""Research Plan generation pipeline: multi-stage section-by-section generation + self-review + peer review.

Generates publication-quality research plans targeting top ML venues (NeurIPS/ICML/ICLR/CVPR).
Each section uses a specialized prompt. Experimental Design uses structured JSON forcing.
After generation, a consistency review catches cross-section contradictions.
Final stage simulates 3 peer reviewers using ICLR scoring criteria.

Design informed by:
- AI Scientist v2 (Sakana AI): per-section generation tips
- Idea2Plan: structured research plan generation
- NeurIPS Paper Checklist: rigor & reproducibility standards
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from paper_tracker.llm import call_cli
from paper_tracker.storage import Storage

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paper Context Management
# ---------------------------------------------------------------------------


def _build_paper_catalog(papers: list[dict]) -> str:
    """Build a compact catalog of ALL papers (titles + 1-line summaries) for Claude to read."""
    lines = []
    for i, p in enumerate(papers):
        title = p.get("title", "Untitled")
        summary = p.get("summary", p.get("key_insight", ""))
        # Keep each entry to ~1 line for compactness
        if summary:
            lines.append(f"{i+1}. **{title}** — {summary}")
        else:
            lines.append(f"{i+1}. **{title}**")
    return "\n".join(lines)


_PAPER_CURATION_PROMPT = """\
You are a research assistant selecting and summarizing papers relevant to a research idea.

## Research Idea
Title: {title}
Problem: {problem}
Method: {method}

## Paper Library ({n_papers} papers)
{paper_catalog}

## Task
From the papers above, select the ones most relevant to this research idea and produce \
a focused research context. Output THREE sections:

### Key Papers (5-8 papers)
For each, write 2-3 sentences explaining: what the paper does, and how it specifically \
relates to or contrasts with the proposed idea. These are the papers that should be cited \
and positioned against in the research plan.

### Supporting Papers (8-12 papers)
For each, write 1 sentence on relevance. These provide broader context (baselines, \
related techniques, datasets).

### Additional References
Just list titles of any other potentially citable papers (1 per line, no description).

IMPORTANT:
- Focus on papers about: diffusion/video generation acceleration, speculative decoding, \
inference efficiency, temporal consistency — topics directly related to the idea.
- Skip papers that are clearly irrelevant (e.g., medical imaging, protein design, etc.).
- Be concise. Total output should be under 4000 words.

Reply with Markdown only. No meta-commentary."""


def curate_papers_for_idea(papers: list[dict], idea: dict, cfg: dict) -> str:
    """Use Claude to select and summarize the most relevant papers for a research idea.

    This replaces hard-coded keyword scoring with intelligent LLM-based curation.
    Returns a formatted paper context string ready to embed in stage prompts.
    """
    if not papers:
        return "(No papers in library yet)"

    catalog = _build_paper_catalog(papers)
    prompt = _PAPER_CURATION_PROMPT.format(
        title=idea.get("title", ""),
        problem=idea.get("problem", ""),
        method=idea.get("method", ""),
        n_papers=len(papers),
        paper_catalog=catalog,
    )

    log.info("Paper curation: %d papers, catalog %d chars, prompt %d chars",
             len(papers), len(catalog), len(prompt))

    result = call_cli(prompt, cfg, timeout=180)
    if result:
        log.info("Paper curation done: %d chars of curated context", len(result))
        return result

    log.warning("Paper curation failed, falling back to first 15 papers")
    # Fallback: use first 15 papers as-is
    lines = []
    for p in papers[:15]:
        title = p.get("title", "Untitled")
        summary = p.get("summary", "")
        lines.append(f"**{title}**\n  Summary: {summary}")
    return "\n\n".join(lines)


def _build_prior_art_context(idea: dict) -> str:
    """Build prior art context string from idea's prior_art field (if present)."""
    pa = idea.get("prior_art")
    if not pa:
        return ""

    lines = ["## Prior Art Analysis (from arXiv-wide search)"]
    lines.append(f"**Maturity Level**: {pa.get('maturity_level', 'UNKNOWN')} "
                 f"({pa.get('total_related', 0)} related papers found)")
    lines.append(f"**Recommendation**: {pa.get('recommendation', 'N/A')}")

    assessment = pa.get("novelty_assessment", "")
    if assessment:
        lines.append(f"**Novelty Assessment**: {assessment}")

    reason = pa.get("recommendation_reason", "")
    if reason:
        lines.append(f"**Recommendation Reason**: {reason}")

    prior = pa.get("prior_works", [])
    if prior:
        lines.append("")
        lines.append("### Prior Works (foundational)")
        for w in prior:
            arxiv_id = w.get("arxiv_id", "")
            title = w.get("title", "Unknown")
            rel = w.get("relevance", "")
            entry = f"- **{title}** (arXiv:{arxiv_id})"
            if rel:
                entry += f" — {rel}"
            lines.append(entry)

    similar = pa.get("similar_works", [])
    if similar:
        lines.append("")
        lines.append("### Similar Works (high overlap)")
        for w in similar:
            arxiv_id = w.get("arxiv_id", "")
            title = w.get("title", "Unknown")
            overlap = w.get("overlap", "")
            entry = f"- **{title}** (arXiv:{arxiv_id})"
            if overlap:
                entry += f" — {overlap}"
            lines.append(entry)

    return "\n".join(lines)


def _build_review_context(idea: dict) -> str:
    """Build review context string from idea's review field (from brainstorm review loop).

    Extracts scores, weaknesses, and strengths so the research plan generation
    can address known weaknesses from the start.
    """
    review = idea.get("review")
    if not review or not isinstance(review, dict):
        return ""

    lines = ["## Brainstorm Review Feedback"]
    lines.append("The following review was conducted during the brainstorm stage. "
                 "Address the identified weaknesses in your plan — especially novelty "
                 "concerns, which are structural and hardest to fix later.\n")

    # Scores
    scores = []
    for dim in ("novelty", "feasibility", "clarity", "impact", "overall"):
        val = review.get(dim)
        if val is not None:
            scores.append(f"{dim}={val}")
    if scores:
        lines.append(f"**Scores**: {', '.join(scores)}")

    # Strengths
    strengths = review.get("strengths", [])
    if strengths:
        lines.append("\n**Strengths** (preserve these):")
        for s in strengths:
            lines.append(f"- {s}")

    # Weaknesses — flag novelty score explicitly
    novelty_score = review.get("novelty")
    if novelty_score is not None and novelty_score <= 6:
        lines.append(f"\n**NOVELTY WARNING** (score: {novelty_score}/10): "
                     "The brainstorm review flagged novelty as a concern. "
                     "The Introduction MUST clearly articulate what is genuinely new "
                     "(apply the 'X+Y' test — can the contribution be reduced to combining "
                     "two known techniques?). The Related Work MUST explicitly position "
                     "against the closest prior work. The Methodology MUST highlight the "
                     "non-obvious insight that makes this more than incremental.")

    weaknesses = review.get("weaknesses", [])
    if weaknesses:
        lines.append("\n**Weaknesses** (must address in plan):")
        for w in weaknesses:
            lines.append(f"- {w}")

    # Revision instructions
    instructions = review.get("revision_instructions", [])
    if instructions:
        lines.append("\n**Revision Instructions**:")
        for inst in instructions:
            lines.append(f"- {inst}")

    return "\n".join(lines)


def _strip_leading_heading(text: str) -> str:
    """Remove any leading ## heading lines from section content to prevent duplication."""
    lines = text.strip().split("\n")
    while lines and re.match(r"^#{1,3}\s+\d*\.?\s*", lines[0]):
        lines.pop(0)
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Stage 1: Introduction & Motivation
# ---------------------------------------------------------------------------

_INTRODUCTION_PROMPT = """\
You are a senior ML researcher writing the Introduction section of a top-venue \
(NeurIPS/ICML/ICLR) research plan.

## Research Idea
Title: {title}
Problem: {problem}
Motivation: {motivation}
Method: {method}

## Paper Context (curated from {n_papers} papers in library)
{paper_summaries}

{prior_art_context}

## Instructions
Write a complete "Introduction & Motivation" section with these requirements:

1. **Problem Statement**: Open with 1-2 precise sentences defining the problem.
2. **Timeliness**: Explain why NOW is the right time to solve this (cite recent papers from the library).
3. **Limitations of Existing Work**: Use the pattern "Unlike X which does Y, this work Z" to \
position against 2-3 specific papers from the library. Be concrete about what they cannot do.
4. **Core Hypothesis**: State the central claim or hypothesis clearly.
5. **Contributions**: List 3-4 specific, concrete contributions (not vague). Each should be \
independently verifiable.

IMPORTANT for claims: Be precise about what guarantees the method provides. If the method \
is approximate rather than exact/lossless, say so explicitly. Do not overclaim — reviewers \
will check every theoretical claim against the actual algorithm.

If Prior Art Analysis is provided above, USE it to:
- Explicitly position against the most relevant prior works and similar works by name
- Address the maturity level honestly — if the area is MATURE or SATURATED, emphasize what \
specifically differentiates this work from existing approaches
- Ground your novelty claims in the prior art assessment

Use academic tone. Reference papers from the library by title. Do NOT include the section heading \
itself (no "## 1. Introduction") — just the content.

Reply with Markdown content only. No meta-commentary."""


# ---------------------------------------------------------------------------
# Stage 2: Related Work
# ---------------------------------------------------------------------------

_RELATED_WORK_PROMPT = """\
You are writing the Related Work section of a top-venue ML research plan.

## Research Idea
Title: {title}
Problem: {problem}
Method: {method}

## Paper Context (curated from {n_papers} papers in library)
{paper_summaries}

{prior_art_context}

## Instructions
Write a comprehensive Related Work section organized as follows:

1. **Group by methodology/theme** (NOT chronologically). Create 3-5 thematic groups.
2. For each group:
   - Summarize the group's approach (2-3 sentences)
   - Cite specific papers from the library by title
   - Identify concrete limitations of this group's approaches
   - Explain how the proposed work addresses these limitations
3. End with a **positioning paragraph** that summarizes how this work differs from ALL \
prior work along at least 2 dimensions (e.g., scale, modality, architecture, objective).

If Prior Art Analysis is provided above, you MUST:
- Incorporate the listed prior works and similar works into appropriate thematic groups. \
These are papers found via global arXiv search specifically for this idea — they are \
high-relevance and should be cited.
- Use the overlap/relevance descriptions to explain how this work differs from each.
- Address the maturity assessment — if the field is GROWING or MATURE, map out the landscape \
thoroughly to show the reviewer you understand the crowded space.

Be specific — name methods, architectures, and results. Avoid vague statements like \
"previous work has limitations."

Do NOT include the section heading itself (no "## 2. Related Work") — just the content.

Reply with Markdown content only. No meta-commentary."""


# ---------------------------------------------------------------------------
# Stage 3: Methodology
# ---------------------------------------------------------------------------

_METHODOLOGY_PROMPT = """\
You are writing the Methodology section of a top-venue ML paper plan.

## Research Idea
Title: {title}
Problem: {problem}
Method: {method}
Experiment Plan: {experiment_plan}

## Paper Context (curated from library)
{paper_summaries}

## Instructions
Write a rigorous Methodology section with:

1. **Formal Problem Definition**: Use LaTeX-style math notation (e.g., $\\mathcal{{L}}$, \
$\\mathbb{{R}}^n$, $\\theta^*$). Define input space, output space, and the learning objective.

2. **Architecture / Algorithm Description**: Provide pseudocode-level detail. If proposing a \
neural architecture, describe each component (encoder, decoder, attention, loss, etc.). \
Use numbered equations for key formulas.

3. **Design Decisions**: For each major design choice, list:
   - The chosen approach
   - 1-2 alternatives considered
   - Why this choice is superior (cite evidence if possible)

4. **Complexity Analysis**: Time and space complexity of the key operations. Compare with \
baseline complexity where relevant.

5. **Loss Function**: Write the complete loss function with all terms, weights, and regularization. \
Explain each term.

## CRITICAL SELF-CHECK REQUIREMENTS
Before finalizing, verify:
- Every variable in the algorithm pseudocode is defined and used consistently
- If you claim "lossless" or "exact", verify the algorithm includes a correction mechanism \
(e.g., rejection sampling) that provably recovers the target distribution. If not, use \
"approximate" or "bounded-error" instead. Default to conservative claims.
- Hyperparameter values mentioned here (e.g., model sizes, max sequence lengths, window sizes) \
MUST be consistent with what appears in experimental design later. Use specific values and \
note them clearly.
- Each equation's variables match the pseudocode's variable names
- FLOPs/compute cost claims MUST be backed by per-operation analysis. If claiming X is N× \
cheaper than Y, show the calculation (attention FLOPs, FFN FLOPs, memory costs).
- Threshold/hyperparameter definitions MUST be unambiguous: state the exact scale (e.g., \
"τ ∈ [0, 1] where lower means stricter") and give a specific default value with justification.
- COMPUTE CONSTRAINT: The researcher has 2-3 nodes of 8×H800 GPUs (16-24 GPUs total). \
Design methods that are feasible within this budget. If proposing distributed training, \
limit to 2-3 nodes. Do not assume unlimited compute.

Use LaTeX notation for all math. Include at least one algorithm in pseudocode format (use \
numbered steps). Be concrete enough that a graduate student could implement this.

Do NOT include the section heading itself (no "## 3. Methodology") — just the content.

Reply with Markdown content only. No meta-commentary."""


# ---------------------------------------------------------------------------
# Stage 4: Experimental Design (JSON → Markdown)
# ---------------------------------------------------------------------------

_EXPERIMENT_JSON_PROMPT = """\
You are designing the experimental plan for a top-venue ML paper.

## Research Idea
Title: {title}
Problem: {problem}
Method: {method}
Initial Experiment Plan: {experiment_plan}

## Methodology Section (MUST be consistent with)
{methodology_summary}

## Paper Context (curated from library)
{paper_summaries}

## Instructions
Output a JSON object with this EXACT schema. All hyperparameter values, model sizes, and \
design choices MUST be consistent with the Methodology Section above.

{{
  "research_questions": [
    {{
      "id": "RQ1",
      "question": "...",
      "hypothesis": "..."
    }}
  ],
  "datasets": [
    {{
      "name": "...",
      "size": "...",
      "split": "train/val/test ratio",
      "preprocessing": "...",
      "justification": "why this dataset"
    }}
  ],
  "baselines": [
    {{
      "name": "...",
      "reference": "paper title from library if applicable",
      "known_result": "reported metric if known",
      "justification": "why compare against this"
    }}
  ],
  "metrics": [
    {{
      "name": "...",
      "formula": "mathematical formula",
      "justification": "why this metric"
    }}
  ],
  "ablation_studies": [
    {{
      "component_removed": "...",
      "replacement_strategy": "...",
      "expected_impact": "..."
    }}
  ],
  "hyperparameters": {{
    "search_space": {{"param_name": "range or choices"}},
    "search_method": "grid/random/bayesian",
    "budget": "number of trials"
  }},
  "compute": {{
    "hardware": "GPU type and count",
    "estimated_hours": "...",
    "reproducibility": "number of seeds, confidence intervals, significance test"
  }}
}}

IMPORTANT:
- The hyperparameter search space MUST include the exact values used in the Methodology \
section. For example, if the methodology uses max_length=8, the search space must include \
8 in its range.
- Baselines MUST include at least one "deployment-realistic" baseline (the configuration \
practitioners actually use, e.g., 20-step DPM-Solver++ rather than 50-step DDIM). Impact \
claims without comparison to practical baselines are not publishable.
- Include at least one parallel/iterative baseline if the method involves speculative or \
parallel computation (e.g., ParaDiGMS, Picard iterations).

## COMPUTE RESOURCE CONSTRAINTS (CRITICAL)
The researcher has access to 2-3 nodes of 8×H800 GPUs (16-24 GPUs total). \
All compute requirements MUST fit within this budget:
- Training: Design experiments that complete within 2-3 nodes × 8 GPUs. \
If a model needs more than 24 GPUs, scale down the model or use efficient methods \
(gradient checkpointing, mixed precision, DeepSpeed ZeRO).
- Do NOT assume access to 64+ GPU clusters, TPU pods, or datacenter-scale compute.
- Total training time for ALL experiments (including baselines and ablations) should \
be achievable within 1-2 weeks on 2-3 H800 nodes.
- If a method inherently requires massive compute (e.g., training a foundation model \
from scratch), propose a scaled-down variant or fine-tuning approach that fits the budget.
- Be honest about limitations imposed by compute constraints — state them explicitly \
rather than handwaving.

Be specific — use real dataset names, real model names, real metric formulas.
Reply ONLY with valid JSON. No markdown, no explanation."""


_EXPERIMENT_MARKDOWN_PROMPT = """\
Convert this experimental design JSON into a well-formatted Markdown section for a research plan.

## JSON Data
{json_data}

## Instructions
Write a clear, readable Experimental Design section with these subsections:

1. **Research Questions**: Number each RQ and state the hypothesis
2. **Datasets**: Table format with name, size, split, preprocessing, justification
3. **Baselines**: Table format with model name, reference, known results
4. **Evaluation Metrics**: Each metric with its formula and justification
5. **Ablation Studies**: Table showing component, replacement, expected impact
6. **Hyperparameter Search**: Search space, method, budget
7. **Compute Budget & Reproducibility**: Hardware, time estimate, seeds, statistical tests

Use Markdown tables where appropriate. Be precise and specific.

Do NOT include a top-level section heading (no "## 4. Experimental Design") — start \
directly with the subsections.

Reply with Markdown content only. No meta-commentary."""


# ---------------------------------------------------------------------------
# Stage 5: Expected Results & Timeline
# ---------------------------------------------------------------------------

_RESULTS_TIMELINE_PROMPT = """\
You are writing the final sections of a top-venue ML research plan.

## Research Idea
Title: {title}

## Prior Sections (for context)
### Introduction
{introduction}

### Methodology (summary)
{methodology_summary}

### Experimental Design (summary)
{experiment_summary}

## Instructions
Write TWO sections separated by the exact line "---SECTION_BREAK---":

### PART 1: Expected Results & Risk Analysis
For each research question from the experimental design:
1. State the **hypothesized result** with expected magnitude (e.g., "We expect 3-5% improvement \
in BLEU score over the strongest baseline")
2. Identify **failure modes**: What could go wrong? What if the hypothesis is wrong?
3. Provide **contingency plans**: If the main approach fails, what is plan B?
4. State **limitations**: What does this approach NOT claim to solve?
5. Define **scope boundaries**: Under what conditions does this work NOT apply?

### PART 2: Timeline & Milestones
Create a realistic research timeline with:
- 4-6 phases (e.g., Literature Review, Implementation, Experimentation, Analysis, Writing)
- Each phase has: duration, deliverables, and validation criteria
- Include buffer time for unexpected issues
- Mark critical decision points

Use a Markdown table for the timeline. Be realistic about timescales.

COMPUTE CONSTRAINT: The researcher has 2-3 nodes of 8×H800 GPUs (16-24 GPUs total). \
All timeline estimates, expected training durations, and resource requirements MUST be \
realistic for this budget. Do NOT assume access to larger clusters.

Do NOT include any section headings like "## 5." or "## 6." — just the content for each part.
Separate the two parts with EXACTLY this line on its own: ---SECTION_BREAK---

Reply with Markdown content only. No meta-commentary."""


# ---------------------------------------------------------------------------
# Stage 6: Consistency Review (new)
# ---------------------------------------------------------------------------

_CONSISTENCY_REVIEW_PROMPT = """\
You are a meticulous technical editor checking a research plan for internal consistency.

## Research Plan Sections

### Introduction
{introduction}

### Methodology
{methodology}

### Experimental Design
{experimental_design}

### Expected Results
{expected_results}

## Task
Check for ONLY these specific types of inconsistencies:

1. **Hyperparameter conflicts**: Values mentioned in Methodology (e.g., max window size, \
model capacity ratio, threshold values) vs. values in Experimental Design's hyperparameter \
search space. List each conflict.
2. **Overclaims**: Claims in Introduction (e.g., "lossless", "exact", "provably") that \
are not supported by the actual algorithm in Methodology. List each.
3. **Missing baselines**: Methods discussed in Related Work/Introduction that should be \
baselines but are absent from Experimental Design.
4. **Metric-claim mismatch**: Contributions claimed in Introduction but no corresponding \
metric or experiment to verify them.

For each issue found, output:
- **Section**: where the problem is
- **Issue**: what's wrong
- **Fix**: specific text to change or add

If no issues are found for a category, say "None found."

Reply with Markdown. Be concise — only flag genuine inconsistencies, not style preferences."""


# ---------------------------------------------------------------------------
# Stage 7: Peer Review Simulation (Multi-Reviewer Pipeline)
# ---------------------------------------------------------------------------
# Design informed by:
# - AI Scientist (Sakana AI): independent ensemble + iterative reflection
# - AgentReview (EMNLP 2024): reviewer persona system + score calibration
# - MARG (Allen AI): specialized reviewer groups per dimension
# - Stanford Agentic Reviewer: prior art integration
#
# Key difference: we review RESEARCH PLANS, not finished papers.
# ---------------------------------------------------------------------------

_PLAN_CONTEXT_BLOCK = """\
## Complete Research Plan
# {title}

## 1. Introduction & Motivation
{introduction}

## 2. Related Work
{related_work}

## 3. Methodology
{methodology}

## 4. Experimental Design
{experimental_design}

## 5. Expected Results & Risk Analysis
{expected_results}

{timeline}

{prior_art_block}"""

_PLAN_CALIBRATION = """\
## CRITICAL — Research Plan Calibration
You are reviewing a RESEARCH PLAN, NOT a finished paper. Calibrate accordingly:
- Plans do NOT have preliminary results, complete proofs, or ablation tables. \
Do NOT penalize for their absence.
- Evaluate on: soundness of design, feasibility of execution, completeness of \
experimental plan, honesty of claims.
- A well-designed plan with concrete methodology, appropriate baselines, and \
clear experimental design should score 6-7 (Weak Accept range).
- Reserve 4-5 (Borderline/Weak Reject) for plans with fundamental methodology \
flaws, missing critical baselines, or clearly overclaimed contributions.
- Reserve 1-3 for plans that are technically unsound or have no clear contribution.
- DO penalize: overclaiming, inconsistent numbers across sections, missing \
practical baselines, underspecified training, vague methodology.
- Most research plans get scores ≤5. Do not assign 7+ unless the plan \
demonstrates exceptional rigor AND novelty."""

_REVIEWER_TECHNICAL_PROMPT = """\
You are a rigorous methodology expert reviewing a research plan submitted to \
a top ML venue (ICLR/NeurIPS). You have deep expertise in algorithms, \
optimization theory, and mathematical modeling. You are skeptical of \
hand-wavy arguments and demand precise formulations.

{plan_context}

{calibration}

{review_history_block}

## Your Focus Areas
1. **Mathematical correctness**: Are all equations, objectives, and algorithms \
internally consistent? Check variable definitions, gradient formulas, KL \
divergence directions, loss function terms.
2. **Algorithm soundness**: Could the pseudocode be implemented as written? \
Are there missing steps, ambiguous operations, or undefined edge cases?
3. **Theoretical claims**: Does the plan overclaim? If it says "lossless", \
"exact", or "provably", is there a mechanism that actually guarantees this? \
Default to flagging unsubstantiated theoretical claims.
4. **Complexity analysis**: Are FLOPs/memory estimates correct? Do scaling \
claims hold under the stated compute constraints?

## Output Format
Provide your review as a JSON object with these exact keys:
```json
{{
  "summary": "2-3 sentence summary of the plan and your overall assessment",
  "strengths": [
    {{"id": "S1", "point": "be specific, cite equations/sections", "section_refs": ["methodology"]}}
  ],
  "weaknesses": [
    {{"id": "W1", "point": "be specific, cite equations/sections", "section_refs": ["methodology", "experimental_design"], "severity": "major"}}
  ],
  "questions": ["question for authors 1", "..."],
  "suggestions": [
    {{"id": "A1", "action": "actionable suggestion", "target_sections": ["methodology"], "linked_weakness": "W1"}}
  ],
  "soundness": <1-4>,
  "presentation": <1-4>,
  "contribution": <1-4>,
  "overall": <1-10>,
  "confidence": <1-5>
}}
```
Valid `section_refs` / `target_sections`: `introduction`, `related_work`, `methodology`, \
`experimental_design`, `expected_results`, `timeline`.
Severity: `major` (fundamental flaw), `minor` (should fix), `nitpick` (nice to have).

THOUGHT: Before writing the JSON, think step-by-step about each equation and \
algorithm in the Methodology section. Check every variable is defined, every \
claim is supported, and every complexity bound is derived.

Reply with your reasoning in a THOUGHT section, then the JSON. No markdown fences \
around the JSON."""

_REVIEWER_EMPIRICAL_PROMPT = """\
You are a meticulous empiricist reviewing a research plan submitted to a top \
ML venue (ICLR/NeurIPS). You have extensive experience running large-scale \
experiments and know exactly what it takes to produce reproducible results. \
You are suspicious of missing baselines and unrealistic compute estimates.

{plan_context}

{calibration}

{review_history_block}

## Your Focus Areas
1. **Baselines**: Are ALL relevant baselines included? Is there a \
"deployment-realistic" baseline (what practitioners actually use, not just \
academic comparisons)? Are there missing recent methods that a reviewer \
would flag?
2. **Datasets & evaluation**: Are the chosen datasets appropriate? Is the \
test set large enough for statistical significance? Are the metrics \
comprehensive (not cherry-picked)?
3. **Reproducibility**: Could a graduate student reproduce this from the plan? \
Are hyperparameters specified? Is the training procedure unambiguous?
4. **Compute feasibility**: Do the estimated GPU-hours fit the stated \
hardware constraints? Account for ALL runs: main experiments, baselines, \
ablations, hyperparameter sweeps, multiple seeds.
5. **Ablation completeness**: Does every claimed contribution have a \
corresponding ablation study?

## Output Format
Provide your review as a JSON object with these exact keys:
```json
{{
  "summary": "2-3 sentence summary of the plan and your overall assessment",
  "strengths": [
    {{"id": "S1", "point": "cite specific baselines/datasets/numbers", "section_refs": ["experimental_design"]}}
  ],
  "weaknesses": [
    {{"id": "W1", "point": "cite specific missing baselines/issues", "section_refs": ["experimental_design"], "severity": "major"}}
  ],
  "questions": ["question for authors 1", "..."],
  "suggestions": [
    {{"id": "A1", "action": "actionable suggestion", "target_sections": ["experimental_design"], "linked_weakness": "W1"}}
  ],
  "soundness": <1-4>,
  "presentation": <1-4>,
  "contribution": <1-4>,
  "overall": <1-10>,
  "confidence": <1-5>
}}
```
Valid `section_refs` / `target_sections`: `introduction`, `related_work`, `methodology`, \
`experimental_design`, `expected_results`, `timeline`.
Severity: `major` (fundamental flaw), `minor` (should fix), `nitpick` (nice to have).

THOUGHT: Before writing the JSON, enumerate every baseline mentioned, every \
dataset, and every compute estimate. Check if the total compute budget is \
consistent across sections. Identify any missing comparisons.

Reply with your reasoning in a THOUGHT section, then the JSON. No markdown fences \
around the JSON."""

_REVIEWER_NOVELTY_PROMPT = """\
You are a visionary researcher and area expert reviewing a research plan \
submitted to a top ML venue (ICLR/NeurIPS). You care deeply about whether \
the work pushes the field forward or merely combines existing ideas. You \
are well-read and can spot when a "novel contribution" is actually a \
well-known technique with a new name.

{plan_context}

{calibration}

{review_history_block}

## Your Focus Areas
1. **Novelty assessment**: Is the core idea genuinely new, or is it an \
incremental combination of existing techniques? Could the contribution be \
summarized as "X + Y" where X and Y are both known?
2. **Significance**: If this plan succeeds, would the results matter to the \
community? Would practitioners adopt this? Would it open new research \
directions?
3. **Positioning**: Does the plan clearly differentiate from the closest \
prior work? Is the Related Work comprehensive or does it conveniently \
omit competing approaches?
4. **Motivation clarity**: Is the problem well-motivated? Does the plan \
convince you that this problem NEEDS solving right now?
5. **Conceptual unity**: Do the components form a coherent, unified \
contribution, or is this a bag of tricks held together by a thin narrative?

## Output Format
Provide your review as a JSON object with these exact keys:
```json
{{
  "summary": "2-3 sentence summary of the plan and your overall assessment",
  "strengths": [
    {{"id": "S1", "point": "cite specific novelty/impact arguments", "section_refs": ["introduction"]}}
  ],
  "weaknesses": [
    {{"id": "W1", "point": "cite specific prior work overlap", "section_refs": ["related_work", "introduction"], "severity": "major"}}
  ],
  "questions": ["question for authors 1", "..."],
  "suggestions": [
    {{"id": "A1", "action": "actionable suggestion", "target_sections": ["introduction"], "linked_weakness": "W1"}}
  ],
  "soundness": <1-4>,
  "presentation": <1-4>,
  "contribution": <1-4>,
  "overall": <1-10>,
  "confidence": <1-5>
}}
```
Valid `section_refs` / `target_sections`: `introduction`, `related_work`, `methodology`, \
`experimental_design`, `expected_results`, `timeline`.
Severity: `major` (fundamental flaw), `minor` (should fix), `nitpick` (nice to have).

THOUGHT: Before writing the JSON, identify the single core contribution. \
Then ask: has this been done before? How is this different from the top 3 \
most similar papers in Related Work?

Reply with your reasoning in a THOUGHT section, then the JSON. No markdown fences \
around the JSON."""

_REVIEWER_REFLECTION_PROMPT = """\
You just wrote the following review of a research plan. Re-read it carefully \
and improve it.

## Your Previous Review
{previous_review_json}

## The Research Plan (for reference)
{plan_context}

## Reflection Instructions
1. **Accuracy check**: Did you make any factual errors about what the plan \
actually proposes? Re-read the methodology and correct any misstatements.
2. **Fairness check**: Were any of your criticisms unfair for a research \
PLAN (as opposed to a finished paper)? Remove criticisms that demand \
completed experiments or proofs.
3. **Missed issues**: On re-reading, did you miss any significant technical \
problems, missing baselines, or overclaims?
4. **Constructiveness**: For each weakness, have you suggested a concrete fix? \
Add suggestions where missing.
5. **Score calibration**: Re-examine your scores. Are they consistent with \
your stated strengths and weaknesses? A plan with 3 major weaknesses should \
not score 7+.
6. **Section tagging**: Verify each weakness and strength has accurate \
`section_refs`. Verify each suggestion has accurate `target_sections` and \
a valid `linked_weakness` ID.

Output your refined review as a JSON object with the SAME structured keys as before \
(strengths/weaknesses as objects with id/point/section_refs, suggestions with \
id/action/target_sections/linked_weakness). \
If nothing needs changing, return the same JSON unchanged.

Reply with a brief REFLECTION section explaining what you changed, then the \
updated JSON. No markdown fences around the JSON."""

_META_REVIEW_PROMPT = """\
You are an Area Chair at a top ML venue (ICLR/NeurIPS). You must synthesize \
{n_reviews} independent reviewer assessments into a single meta-review.

## Research Plan Title
{title}

## Individual Reviews

{reviews_text}

## Instructions
Write a meta-review that:

1. **Score table**: Average scores across all reviewers (use a Markdown table).
2. **Consensus strengths**: Points that 2+ reviewers agree on (cite which reviewers).
3. **Consensus weaknesses**: Points that 2+ reviewers agree on (cite which reviewers).
4. **Disputed points**: Issues where reviewers disagree — state both sides.
5. **Overall recommendation**: Accept / Weak Accept / Borderline / Weak Reject / Reject.
   - Weigh the reviews equally but prioritize consensus weaknesses.
   - For plans (not papers): be more tolerant of missing results, less tolerant \
   of unclear methodology or missing baselines.
6. **Top 3 critical improvements**: The single most impactful changes the authors \
should make, in priority order. Each must be actionable (not "improve writing").

## Calibration
- You are meta-reviewing a RESEARCH PLAN. The bar is: "Would I fund/supervise \
this project?" not "Would I accept this NeurIPS submission?"
- A well-designed, feasible plan with some incremental novelty = Weak Accept (6).
- A plan with a genuinely novel core idea, solid methodology, and comprehensive \
experimental design = Accept (7-8).
- A plan with fundamental flaws in methodology OR novelty = Weak Reject (4-5).

Reply with Markdown content only. No meta-commentary."""


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def generate_research_plan(
    topic_id: str,
    topic_name: str,
    data_dir: str,
    cfg: dict,
    idea: dict,
    source_brainstorm_id: str = "",
    *,
    on_section_done: callable | None = None,
) -> dict:
    """Run the multi-stage research plan generation pipeline.

    Stages:
      1. Introduction & Motivation
      2. Related Work
      3. Methodology (with self-consistency instructions)
      4. Experimental Design (JSON → Markdown, with methodology context)
      5. Expected Results & Timeline
      6. Consistency Review (cross-section check)
      7. Auto-patch (fix missing baselines, metrics, hyperparameter conflicts)
      8. Peer Review Simulation

    Args:
        topic_id: Topic identifier.
        topic_name: Human-readable topic name.
        data_dir: Base data directory.
        cfg: Pipeline configuration dict.
        idea: Dict with at least title, problem, motivation, method, experiment_plan.
        source_brainstorm_id: Optional brainstorm session ID this idea came from.
        on_section_done: Optional callback(section_name, content) called after each section.

    Returns:
        Dict with keys: introduction, related_work, methodology, experimental_design,
        expected_results, timeline, review, full_markdown.
    """
    log.info("=== Research Plan generation started: topic=%s idea=%s ===",
             topic_id, idea.get("title", "?"))

    # Load papers and curate context via LLM
    store = Storage(data_dir, topic_id)
    try:
        papers, total = store.get_all_arxiv(limit=200, offset=0)
    finally:
        store.close()
    log.info("Loaded %d papers from library (total %d)", len(papers), total)

    title = idea.get("title", "Untitled")
    problem = idea.get("problem", "")
    motivation = idea.get("motivation", "")
    method = idea.get("method", "")
    experiment_plan = idea.get("experiment_plan", "")

    result = {
        "introduction": "",
        "related_work": "",
        "methodology": "",
        "experimental_design": "",
        "expected_results": "",
        "timeline": "",
        "review": "",
        "full_markdown": "",
    }

    # Token budget: ~4 chars per token for English text
    MAX_PROMPT_CHARS = 950_000 * 4  # ~0.95M tokens

    def _call(prompt: str, stage: str, **kwargs) -> str | None:
        """Wrapper around call_cli with prompt size logging and budget check."""
        prompt_chars = len(prompt)
        est_tokens = prompt_chars // 4
        log.info("  [%s] prompt: %d chars (~%dk tokens)", stage, prompt_chars, est_tokens // 1000)
        if prompt_chars > MAX_PROMPT_CHARS:
            log.warning("  [%s] OVER BUDGET: %d chars > %d max", stage, prompt_chars, MAX_PROMPT_CHARS)
        return call_cli(prompt, cfg, **kwargs)

    def _notify(section: str, content: str) -> None:
        if on_section_done:
            try:
                on_section_done(section, content)
            except Exception:
                log.warning("on_section_done callback failed for %s", section)

    # --- Stage 0: Paper Curation (LLM-based) ---
    log.info("Stage 0/8: Paper Curation")
    paper_context = curate_papers_for_idea(papers, idea, cfg)
    n_papers = len(papers)
    _notify("_curation", f"Curated {len(paper_context)} chars from {n_papers} papers")

    # Build prior art context if available
    prior_art_ctx = _build_prior_art_context(idea)
    if prior_art_ctx:
        log.info("Prior art context: %d chars (maturity=%s)",
                 len(prior_art_ctx), idea.get("prior_art", {}).get("maturity_level", "?"))

    # Build brainstorm review context if available
    review_ctx = _build_review_context(idea)
    if review_ctx:
        log.info("Brainstorm review context: %d chars", len(review_ctx))
        prior_art_ctx = f"{prior_art_ctx}\n\n{review_ctx}" if prior_art_ctx else review_ctx

    # --- Stages 1-3: Introduction + Related Work + Methodology (PARALLEL) ---
    log.info("Stages 1-3/8: Introduction + Related Work + Methodology (parallel)")

    intro_prompt = _INTRODUCTION_PROMPT.format(
        title=title, problem=problem, motivation=motivation, method=method,
        n_papers=n_papers, paper_summaries=paper_context,
        prior_art_context=prior_art_ctx,
    )
    rw_prompt = _RELATED_WORK_PROMPT.format(
        title=title, problem=problem, method=method,
        n_papers=n_papers, paper_summaries=paper_context,
        prior_art_context=prior_art_ctx,
    )
    meth_prompt = _METHODOLOGY_PROMPT.format(
        title=title, problem=problem, method=method,
        experiment_plan=experiment_plan,
        paper_summaries=_truncate(paper_context, 4000),
    )

    def _run_stage(name: str, prompt: str, timeout: int) -> tuple[str, str]:
        raw = _call(prompt, name, timeout=timeout)
        return name, _strip_leading_heading(raw or "")

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(_run_stage, "intro", intro_prompt, 300): "introduction",
            pool.submit(_run_stage, "related_work", rw_prompt, 300): "related_work",
            pool.submit(_run_stage, "methodology", meth_prompt, 480): "methodology",
        }
        for fut in as_completed(futures):
            section_key = futures[fut]
            try:
                name, content = fut.result()
                result[section_key] = content
                _notify(section_key, content)
                log.info("  [%s] done: %d chars", name, len(content))
            except Exception as e:
                log.exception("  [%s] failed: %s", section_key, e)

    log.info("Stages 1-3/8 done: intro=%d, rw=%d, meth=%d",
             len(result["introduction"]), len(result["related_work"]), len(result["methodology"]))

    # --- Stage 4: Experimental Design (JSON → Markdown) ---
    log.info("Stage 4/8: Experimental Design")
    # Pass methodology as context so hyperparameters stay consistent
    methodology_summary = _truncate(result["methodology"], 3000)

    exp_json_prompt = _EXPERIMENT_JSON_PROMPT.format(
        title=title,
        problem=problem,
        method=method,
        experiment_plan=experiment_plan,
        methodology_summary=methodology_summary,
        paper_summaries=_truncate(paper_context, 3000),
    )
    exp_json_raw = _call(exp_json_prompt, "exp_json", timeout=300)

    if exp_json_raw:
        # Convert JSON to Markdown
        exp_md_prompt = _EXPERIMENT_MARKDOWN_PROMPT.format(json_data=exp_json_raw)
        exp_md = _call(exp_md_prompt, "exp_md", timeout=240)
        result["experimental_design"] = _strip_leading_heading(exp_md or exp_json_raw)
    else:
        result["experimental_design"] = ""
    _notify("experimental_design", result["experimental_design"])
    log.info("Stage 4/7 done: %d chars", len(result["experimental_design"]))

    # --- Stage 5: Expected Results & Timeline ---
    log.info("Stage 5/8: Expected Results & Timeline")
    experiment_summary = _truncate(result["experimental_design"], 2000)

    rt_prompt = _RESULTS_TIMELINE_PROMPT.format(
        title=title,
        introduction=result["introduction"],
        methodology_summary=_truncate(result["methodology"], 1500),
        experiment_summary=experiment_summary,
    )
    rt_raw = _call(rt_prompt, "results_timeline", timeout=300)

    if rt_raw:
        _split_results_timeline(rt_raw, result)
    _notify("expected_results", result["expected_results"])
    _notify("timeline", result["timeline"])
    log.info("Stage 5/7 done: results=%d chars, timeline=%d chars",
             len(result["expected_results"]), len(result["timeline"]))

    # --- Stage 6: Consistency Review + Auto-patch ---
    log.info("Stage 6/8: Consistency Review")
    consistency = _run_consistency_review(result, cfg)
    if consistency:
        result["expected_results"] += f"\n\n---\n\n### Internal Consistency Notes\n\n{consistency}"
        _notify("expected_results", result["expected_results"])
    log.info("Stage 6/8 done: %d chars of consistency notes",
             len(consistency) if consistency else 0)

    # --- Stage 7: Auto-patch critical issues ---
    log.info("Stage 7/8: Auto-patching critical issues")
    if consistency:
        patched = _auto_patch_sections(result, consistency, cfg)
        if patched:
            for key, value in patched.items():
                if value and key in result:
                    result[key] = value
                    _notify(key, value)
            log.info("Stage 7/8 done: patched %d sections", len(patched))
        else:
            log.info("Stage 7/8 done: no patches needed")
    else:
        log.info("Stage 7/8 skipped: no consistency issues found")

    # --- Stage 8: Peer Review Simulation (multi-reviewer) ---
    log.info("Stage 8/8: Peer Review Simulation (independent reviewers)")
    review = _run_peer_review(title, result, idea, cfg, _call,
                              review_history="(First review round — no prior history)")
    result["review"] = review
    _notify("review", result["review"])
    log.info("Stage 8/8 done: %d chars", len(result["review"]))

    # Track review history (round 1 = initial generation)
    result["review_history"] = [{"round": 1, "review": review}]

    # --- Stage 9: Auto-refine based on peer review ---
    log.info("Stage 9: Auto-refine based on peer review")
    _notify("_status", "auto_refine")
    try:
        refined = _auto_refine_from_review(
            topic_id, topic_name, data_dir, cfg, result, idea,
            title, paper_context, _notify,
        )
        if refined:
            for key in ["introduction", "related_work", "methodology",
                         "experimental_design", "expected_results", "timeline"]:
                if refined.get(key):
                    result[key] = refined[key]
            result["review"] = refined.get("review", result["review"])
            result["review_history"].append({"round": 2, "review": result["review"]})
            log.info("Stage 9 done: auto-refine completed")
        else:
            log.info("Stage 9 done: auto-refine skipped (no actionable issues)")
    except Exception as e:
        log.exception("Stage 9: Auto-refine failed (non-fatal): %s", e)

    # --- Assemble full Markdown ---
    result["full_markdown"] = _assemble_full_markdown(title, result)

    log.info("=== Research Plan generation finished: %d total chars ===",
             len(result["full_markdown"]))
    return result


def _auto_refine_from_review(
    topic_id: str,
    topic_name: str,
    data_dir: str,
    cfg: dict,
    current_result: dict,
    idea: dict,
    title: str,
    paper_context: str,
    _notify: callable,
) -> dict | None:
    """Auto-refine plan based on peer review. Returns refined sections or None."""
    review = current_result.get("review", "")
    if not review or len(review) < 200:
        return None

    # Call the existing refine pipeline with empty user feedback
    # (it will use the peer review as the primary guidance)
    existing_plan = {
        **current_result,
        "idea_title": title,
        "idea_json": idea,
    }
    return refine_research_plan(
        topic_id=topic_id,
        topic_name=topic_name,
        data_dir=data_dir,
        cfg=cfg,
        existing_plan=existing_plan,
        user_feedback="",  # no user feedback — refine based on peer review only
        on_section_done=_notify,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, max_chars: int) -> str:
    """Truncate text with ellipsis marker."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[... truncated for brevity ...]"


def _split_results_timeline(raw: str, result: dict) -> None:
    """Split the combined results+timeline output into two sections.

    Tries multiple separator patterns to handle LLM output variation.
    Strips any leading section headings from each part.
    """
    # Try explicit separator first
    for sep in ["---SECTION_BREAK---", "---section_break---", "---Section_Break---"]:
        if sep in raw:
            parts = raw.split(sep, 1)
            result["expected_results"] = _strip_leading_heading(parts[0].strip())
            result["timeline"] = _strip_leading_heading(parts[1].strip()) if len(parts) > 1 else ""
            return

    # Fallback: split on ## heading patterns for timeline
    for pattern in [
        r"##\s*6\.?\s*Timeline",
        r"##\s*Timeline\s*(?:&|and)\s*Milestones",
        r"##\s*PART\s*2",
        r"###?\s*Timeline",
    ]:
        match = re.search(pattern, raw, re.IGNORECASE)
        if match:
            result["expected_results"] = _strip_leading_heading(raw[:match.start()].strip())
            result["timeline"] = _strip_leading_heading(raw[match.end():].strip())
            return

    # Last resort: everything goes to expected_results
    result["expected_results"] = _strip_leading_heading(raw.strip())
    result["timeline"] = ""


_AUTOPATCH_PROMPT = """\
You are a technical editor applying corrections to a research plan.

## Consistency Issues Found
{consistency_notes}

## Current Experimental Design Section
{experimental_design}

## Current Introduction Section (for reference)
{introduction}

## Task
Apply ONLY the following types of fixes to the Experimental Design section:
1. Add any missing baselines that were flagged (add them to the baselines table)
2. Add any missing metrics that were flagged (add them to the metrics section)
3. Fix any hyperparameter inconsistencies (align values with methodology)

Do NOT rewrite the entire section. Only add or modify the specific parts that address \
the flagged issues. Keep all existing content intact.

Output the COMPLETE updated Experimental Design section (with all original content + fixes).
Do NOT include a section heading (no "## 4. Experimental Design").

Reply with Markdown content only. No meta-commentary."""


def _auto_patch_sections(sections: dict, consistency: str, cfg: dict) -> dict:
    """Apply critical fixes from consistency review to relevant sections."""
    # Only patch if there are actionable issues
    has_missing_baselines = "missing baseline" in consistency.lower()
    has_missing_metrics = "metric" in consistency.lower() and "mismatch" in consistency.lower()
    has_hyperparam_conflict = "hyperparameter" in consistency.lower() or "conflict" in consistency.lower()

    if not (has_missing_baselines or has_missing_metrics or has_hyperparam_conflict):
        return {}

    prompt = _AUTOPATCH_PROMPT.format(
        consistency_notes=_truncate(consistency, 3000),
        experimental_design=sections.get("experimental_design", ""),
        introduction=_truncate(sections.get("introduction", ""), 1500),
    )
    patched_exp = call_cli(prompt, cfg, timeout=300)
    if patched_exp and len(patched_exp) > len(sections.get("experimental_design", "")) * 0.5:
        return {"experimental_design": _strip_leading_heading(patched_exp)}
    return {}


def _run_consistency_review(sections: dict, cfg: dict) -> str:
    """Run cross-section consistency check."""
    prompt = _CONSISTENCY_REVIEW_PROMPT.format(
        introduction=_truncate(sections.get("introduction", ""), 2000),
        methodology=_truncate(sections.get("methodology", ""), 4000),
        experimental_design=_truncate(sections.get("experimental_design", ""), 3000),
        expected_results=_truncate(sections.get("expected_results", ""), 2000),
    )
    raw = call_cli(prompt, cfg, timeout=240)
    return raw or ""


# ---------------------------------------------------------------------------
# Multi-Reviewer Peer Review Pipeline
# ---------------------------------------------------------------------------

_REVIEWER_ROLES = [
    ("Technical Rigor", _REVIEWER_TECHNICAL_PROMPT),
    ("Experimental Design", _REVIEWER_EMPIRICAL_PROMPT),
    ("Novelty & Impact", _REVIEWER_NOVELTY_PROMPT),
]


def _parse_reviewer_json(raw: str) -> dict | None:
    """Extract JSON review object from reviewer output (after THOUGHT section)."""
    if not raw:
        return None
    # Find the JSON object — it comes after the THOUGHT section
    # Try to find a JSON block
    text = raw.strip()
    start = text.rfind("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        # Walk backwards from end to find the matching opening brace
        depth = 0
        for i in range(end, -1, -1):
            if text[i] == "}":
                depth += 1
            elif text[i] == "{":
                depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[i : end + 1])
                except json.JSONDecodeError:
                    break
    return None


_VALID_SECTION_REFS = {
    "introduction", "related_work", "methodology",
    "experimental_design", "expected_results", "timeline",
}


def _normalize_review_json(parsed: dict) -> dict:
    """Normalize a review JSON to the new structured format.

    Handles backward compatibility: if strengths/weaknesses/suggestions are
    plain string lists (old format), convert them to the new object format.
    """
    if not parsed:
        return parsed

    # Normalize strengths
    strengths = parsed.get("strengths", [])
    if strengths and isinstance(strengths[0], str):
        parsed["strengths"] = [
            {"id": f"S{i+1}", "point": s, "section_refs": ["general"]}
            for i, s in enumerate(strengths)
        ]

    # Normalize weaknesses
    weaknesses = parsed.get("weaknesses", [])
    if weaknesses and isinstance(weaknesses[0], str):
        parsed["weaknesses"] = [
            {"id": f"W{i+1}", "point": w, "section_refs": ["general"], "severity": "major"}
            for i, w in enumerate(weaknesses)
        ]

    # Normalize suggestions
    suggestions = parsed.get("suggestions", [])
    if suggestions and isinstance(suggestions[0], str):
        parsed["suggestions"] = [
            {"id": f"A{i+1}", "action": s, "target_sections": ["general"], "linked_weakness": ""}
            for i, s in enumerate(suggestions)
        ]

    return parsed


def _format_review_markdown(role: str, idx: int, review: dict, thought: str = "") -> str:
    """Format a single parsed review into Markdown with section tags and severity."""
    lines = [f"### Reviewer {idx}: {role}"]

    lines.append(f"\n**Scores** (ICLR scale):")
    lines.append(f"- Soundness: {review.get('soundness', '?')} / 4")
    lines.append(f"- Presentation: {review.get('presentation', '?')} / 4")
    lines.append(f"- Contribution: {review.get('contribution', '?')} / 4")
    lines.append(f"- Overall: {review.get('overall', '?')} / 10")
    lines.append(f"- Confidence: {review.get('confidence', '?')} / 5")

    summary = review.get("summary", "")
    if summary:
        lines.append(f"\n**Summary:** {summary}")

    # Strengths (structured)
    strengths = review.get("strengths", [])
    if strengths:
        lines.append("\n**Strengths:**")
        for i, item in enumerate(strengths, 1):
            if isinstance(item, dict):
                sid = item.get("id", f"S{i}")
                point = item.get("point", "")
                refs = item.get("section_refs", [])
                ref_tag = f" [Section: {', '.join(refs)}]" if refs else ""
                lines.append(f"{i}. **{sid}**{ref_tag} {point}")
            else:
                lines.append(f"{i}. {item}")

    # Weaknesses (structured with severity)
    weaknesses = review.get("weaknesses", [])
    if weaknesses:
        lines.append("\n**Weaknesses:**")
        for i, item in enumerate(weaknesses, 1):
            if isinstance(item, dict):
                wid = item.get("id", f"W{i}")
                point = item.get("point", "")
                refs = item.get("section_refs", [])
                severity = item.get("severity", "major")
                ref_tag = f" [Section: {', '.join(refs)}]" if refs else ""
                sev_tag = f" `{severity}`" if severity else ""
                lines.append(f"{i}. **{wid}**{ref_tag}{sev_tag} {point}")
            else:
                lines.append(f"{i}. {item}")

    # Questions (plain list)
    questions = review.get("questions", [])
    if questions:
        lines.append("\n**Questions for Authors:**")
        for i, item in enumerate(questions, 1):
            lines.append(f"{i}. {item}")

    # Suggestions (structured)
    suggestions = review.get("suggestions", [])
    if suggestions:
        lines.append("\n**Suggestions for Improvement:**")
        for i, item in enumerate(suggestions, 1):
            if isinstance(item, dict):
                aid = item.get("id", f"A{i}")
                action = item.get("action", "")
                targets = item.get("target_sections", [])
                linked = item.get("linked_weakness", "")
                ref_tag = f" [Section: {', '.join(targets)}]" if targets else ""
                link_tag = f" (→{linked})" if linked else ""
                lines.append(f"{i}. **{aid}**{ref_tag}{link_tag} {action}")
            else:
                lines.append(f"{i}. {item}")

    return "\n".join(lines)


def _summarize_review_for_history(review_md: str) -> str:
    """Extract a compact review history summary from review markdown.

    Pure parsing, zero LLM calls. Extracts:
    - Each reviewer's scores + top 3 weaknesses
    - Meta-review's top 3 improvements

    Returns compact markdown (~500-800 chars).
    """
    if not review_md:
        return ""

    lines = review_md.split("\n")
    parts = []

    current_reviewer = ""
    in_weaknesses = False
    in_improvements = False
    weakness_count = 0
    improvement_count = 0
    scores_line = ""

    for line in lines:
        # Detect reviewer headers
        if line.startswith("### Reviewer"):
            current_reviewer = line.replace("###", "").strip()
            in_weaknesses = False
            in_improvements = False
            weakness_count = 0
            scores_line = ""
            parts.append(f"\n**{current_reviewer}**")

        # Capture overall score
        if "- Overall:" in line:
            scores_line = line.strip()
            parts.append(f"  {scores_line}")

        # Detect weaknesses section
        if "**Weaknesses:**" in line or "**Weaknesses**" in line:
            in_weaknesses = True
            in_improvements = False
            weakness_count = 0
            continue

        # Detect other sections (end weaknesses)
        if in_weaknesses and line.startswith("**") and "Weakness" not in line:
            in_weaknesses = False

        # Capture top 3 weaknesses
        if in_weaknesses and re.match(r"^\d+\.", line.strip()) and weakness_count < 3:
            parts.append(f"  - {line.strip()}")
            weakness_count += 1

        # Detect meta-review improvements section
        if "critical improvement" in line.lower() or "top 3" in line.lower():
            in_improvements = True
            in_weaknesses = False
            improvement_count = 0
            continue

        # Capture top 3 improvements
        if in_improvements and re.match(r"^\d+\.", line.strip()) and improvement_count < 3:
            parts.append(f"  - {line.strip()}")
            improvement_count += 1

    summary = "\n".join(parts).strip()
    # Cap at ~1000 chars
    if len(summary) > 1000:
        summary = summary[:1000] + "\n  [... truncated ...]"
    return summary


def _run_peer_review(
    title: str,
    sections: dict,
    idea: dict,
    cfg: dict,
    _call: callable,
    review_history: str = "",
) -> str:
    """Run the multi-reviewer peer review pipeline.

    Architecture (informed by AI Scientist + AgentReview + MARG):
      1. 3 independent, specialized reviewers in parallel
      2. 1 reflection round per reviewer (parallel)
      3. 1 meta-reviewer (Area Chair) aggregation

    All reviewers are fully independent — they never see each other's reviews
    until the meta-review step.

    Args:
        review_history: Summary of previous review rounds. Empty string or
            "(First review round — no prior history)" for first round.
    """
    # Build shared plan context
    timeline_section = (
        f"## 6. Timeline & Milestones\n{sections['timeline']}"
        if sections.get("timeline") else ""
    )
    prior_art_ctx = _build_prior_art_context(idea)
    prior_art_block = (
        f"## Prior Art Analysis\n{prior_art_ctx}" if prior_art_ctx else ""
    )

    plan_context = _PLAN_CONTEXT_BLOCK.format(
        title=title,
        introduction=sections.get("introduction", ""),
        related_work=sections.get("related_work", ""),
        methodology=sections.get("methodology", ""),
        experimental_design=sections.get("experimental_design", ""),
        expected_results=sections.get("expected_results", ""),
        timeline=timeline_section,
        prior_art_block=prior_art_block,
    )

    calibration = _PLAN_CALIBRATION

    # Build review history block for reviewer prompts
    if review_history:
        review_history_block = (
            f"## Review History (previous rounds)\n{review_history}\n\n"
            "If review history is provided, check whether previous weaknesses have been addressed. "
            "Flag any unresolved issues. Do not repeat criticisms that have been fixed."
        )
    else:
        review_history_block = ""

    # --- Phase 1: Independent reviews (parallel) ---
    log.info("  [peer_review] Phase 1: 3 independent reviewers (parallel)")

    def _run_reviewer(role: str, prompt_template: str) -> tuple[str, str, dict | None]:
        """Run one reviewer. Returns (role, raw_output, parsed_json)."""
        prompt = prompt_template.format(
            plan_context=plan_context,
            calibration=calibration,
            review_history_block=review_history_block,
        )
        raw = _call(prompt, f"reviewer_{role[:8]}", timeout=360)
        parsed = _parse_reviewer_json(raw) if raw else None
        if parsed:
            parsed = _normalize_review_json(parsed)
        return role, raw or "", parsed

    reviews: list[tuple[str, str, dict | None]] = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [
            pool.submit(_run_reviewer, role, tmpl)
            for role, tmpl in _REVIEWER_ROLES
        ]
        for fut in as_completed(futures):
            try:
                reviews.append(fut.result())
            except Exception as e:
                log.exception("Reviewer failed: %s", e)

    log.info("  [peer_review] Phase 1 done: %d/%d reviewers succeeded",
             sum(1 for _, _, p in reviews if p), len(_REVIEWER_ROLES))

    # --- Phase 2: Reflection round (parallel) ---
    log.info("  [peer_review] Phase 2: Reflection round (parallel)")

    def _run_reflection(role: str, raw_review: str, parsed: dict | None) -> tuple[str, dict | None]:
        """Run one reflection. Returns (role, refined_parsed)."""
        if not parsed:
            return role, None
        prompt = _REVIEWER_REFLECTION_PROMPT.format(
            previous_review_json=json.dumps(parsed, indent=2),
            plan_context=_truncate(plan_context, 80000),
        )
        raw = _call(prompt, f"reflect_{role[:8]}", timeout=240)
        refined = _parse_reviewer_json(raw) if raw else None
        if refined:
            refined = _normalize_review_json(refined)
        return role, refined or parsed  # fallback to original if reflection fails

    reflected: list[tuple[str, dict]] = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [
            pool.submit(_run_reflection, role, raw, parsed)
            for role, raw, parsed in reviews
        ]
        for fut in as_completed(futures):
            try:
                role, refined = fut.result()
                if refined:
                    reflected.append((role, refined))
            except Exception as e:
                log.exception("Reflection failed: %s", e)

    log.info("  [peer_review] Phase 2 done: %d reflected reviews", len(reflected))

    # --- Phase 3: Format individual reviews ---
    review_sections = []
    for idx, (role, parsed) in enumerate(reflected, 1):
        review_sections.append(_format_review_markdown(role, idx, parsed))

    if not review_sections:
        return "(Peer review failed — no reviewers produced valid output)"

    # --- Phase 4: Meta-review (Area Chair) ---
    log.info("  [peer_review] Phase 3: Meta-review (Area Chair)")
    reviews_text = "\n\n---\n\n".join(review_sections)
    meta_prompt = _META_REVIEW_PROMPT.format(
        n_reviews=len(reflected),
        title=title,
        reviews_text=reviews_text,
    )
    meta_raw = _call(meta_prompt, "meta_review", timeout=300)

    # --- Assemble final review document ---
    parts = [
        f"## Peer Review: {title}\n",
        "---\n",
    ]
    parts.extend(review_sections)
    parts.append("\n---\n")
    if meta_raw:
        parts.append(meta_raw)
    else:
        parts.append("(Meta-review generation failed)")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Refinement pipeline (user-directed improvement of an existing plan)
# ---------------------------------------------------------------------------

_REFINE_PLANNER_PROMPT = """\
You are a research plan revision architect. Analyze the peer review (and user \
feedback if provided) to create a precise, globally-consistent modification plan.

## Current Plan Sections (summaries)
{section_summaries}

## User Feedback
{user_feedback}
(If empty, focus entirely on addressing peer review critiques.)

## Peer Review (section-tagged weaknesses and suggestions)
{review_tagged}

## Current Number Table (authoritative values)
{number_table_text}

## Task
Create a revision plan as a JSON object:
```json
{{
  "global_constraints": [
    "constraint that ALL sections must obey (e.g., never claim lossless — use approximate)",
    "key parameter values that must be consistent across sections"
  ],
  "number_updates": {{
    "compute.num_nodes": 2,
    "training.batch_size": 512
  }},
  "revision_plan": [
    {{
      "section": "methodology",
      "priority": 1,
      "weaknesses_to_address": ["W1", "W3"],
      "instructions": "Specific, detailed instructions for revising this section",
      "invariants": ["Things in this section that must NOT change"]
    }}
  ]
}}
```

Rules:
- `priority` determines revision order (1 = first). Revise high-impact sections first \
(usually methodology → experimental_design → introduction → rest).
- `weaknesses_to_address` references weakness IDs from the review.
- `instructions` must be specific enough that another AI can execute them without ambiguity.
- `invariants` protect correct parts from unnecessary changes.
- `global_constraints` are binding for ALL sections — use for numbers, claims, baselines \
that must be identical everywhere.
- `number_updates` (optional): If revision requires changing any numerical values, specify \
them as dot-notation paths into the number table. The updated table will be injected into \
global constraints for all subsequent section revisions. Omit if no numbers need changing.
- Only include sections that actually need changes. Skip sections with no issues.

Reply ONLY with valid JSON. No markdown, no explanation."""


_REFINE_SECTION_TARGETED_PROMPT = """\
You are revising one section of a research plan according to a precise revision plan.

## GLOBAL CONSTRAINTS (binding — all sections must follow these)
{global_constraints}

## Revision Instructions for This Section
{revision_instructions}

## Current Section Content
{current_content}

## Already-Revised Sections (for cross-reference consistency)
{already_revised}

## Paper Context (curated from library)
{paper_context}

## Rules
1. Follow the revision instructions precisely.
2. Follow ALL global constraints — every number, claim, and parameter must match.
3. Check consistency with already-revised sections shown above.
4. Preserve structure and content not mentioned in the instructions.
5. Do NOT include a section heading — just the revised content.

Reply with Markdown content only. No meta-commentary."""


_SECTION_SELF_VERIFY_PROMPT = """\
You are a verification agent checking a just-revised section for correctness.

## Global Constraints
{global_constraints}

## Invariants for This Section
{invariants}

## Revised Section Content
{revised_content}

## Already-Revised Sections (for cross-reference)
{already_revised}

## Verification Checklist
Check these 4 categories:
1. **Global constraint violations**: Does the section violate any global constraint?
2. **Invariant violations**: Were any invariants changed that should have been preserved?
3. **Overclaims**: Does the section claim something stronger than what's justified? \
(e.g., "lossless" without proof, "guaranteed" without bound)
4. **Cross-section contradictions**: Does this section contradict anything in the \
already-revised sections? (numbers, claims, method descriptions, baselines)

## Output
Reply ONLY with a JSON object:
- If all checks pass: {{"passed": true}}
- If any check fails: {{"passed": false, "issues": ["issue 1", "issue 2"], \
"fixed_content": "the complete corrected section content"}}

The `fixed_content` must be the COMPLETE section with all issues fixed. \
Do not output partial content.

Reply ONLY with valid JSON. No markdown, no explanation."""


_NUMBER_TABLE_EXTRACT_PROMPT = """\
You are a meticulous technical editor extracting ALL quantitative values from a \
research plan into a structured table.

## Research Plan Sections

### Methodology
{methodology}

### Experimental Design
{experimental_design}

### Expected Results
{expected_results}

### Timeline
{timeline}

## Task
Extract EVERY numerical value, specification, and quantitative claim into the \
structured JSON below. If a field is not mentioned, omit it entirely. \
If a value appears in multiple sections with DIFFERENT values, record BOTH in \
the "discrepancies" array.

Reply ONLY with a JSON object (no markdown, no explanation):
{{
  "compute": {{
    "gpu_type": "e.g. H800",
    "gpus_per_node": 8,
    "num_nodes": 3,
    "total_gpus": 24,
    "gpu_memory_gb": 80,
    "total_gpu_memory_gb": 1920,
    "training_duration": "e.g. 1-2 weeks",
    "total_experiment_hours": 2016
  }},
  "model": {{
    "parameters": "e.g. 1.3B",
    "architecture": "e.g. Transformer"
  }},
  "training": {{
    "batch_size": 256,
    "learning_rate": "1e-4",
    "training_steps": 100000
  }},
  "performance_claims": [
    {{"claim": "2.1x speedup", "metric": "wall-clock time", "conditions": "batch=16"}}
  ],
  "datasets": [
    {{"name": "WebVid-10M", "size": "10M videos", "split": "80/10/10"}}
  ],
  "key_hyperparameters": [
    {{"name": "window_size", "value": "8", "section_defined": "methodology"}}
  ],
  "discrepancies": [
    {{"field": "batch_size", "values": {{"methodology": 256, "experimental_design": 128}}, "recommendation": "use 256"}}
  ]
}}

Include ONLY fields that have concrete values in the plan. Omit empty objects/arrays."""


_CROSS_SECTION_VERIFY_PROMPT = """\
You are a meticulous cross-section consistency verifier for a research plan.

## Authoritative Number Table
{number_table_json}

## All Revised Sections

### Introduction
{introduction}

### Related Work
{related_work}

### Methodology
{methodology}

### Experimental Design
{experimental_design}

### Expected Results
{expected_results}

### Timeline
{timeline}

## Verification Focus
Check for these specific types of cross-section inconsistencies:

1. **Number table violations**: Compare every number in each section against the \
Authoritative Number Table above. Any section using a different value than the table \
is a violation.
2. **Arithmetic consistency**: Verify derived quantities. For example: \
total_gpus = gpus_per_node × num_nodes, total_gpu_memory = total_gpus × gpu_memory_gb, \
total_hours = nodes × gpus_per_node × hours_per_gpu, FLOPs derivations. \
Flag any arithmetic errors.
3. **Claim alignment**: Contributions listed in Introduction must match what \
Methodology actually describes and what Experiments actually test.
4. **Baseline consistency**: Every baseline in Experimental Design should be \
mentioned in Related Work. Every baseline in Related Work that's relevant \
should appear in Experimental Design.
5. **Method description consistency**: The method name, algorithm steps, and \
core mechanism should be described identically across sections.

## Output
Reply ONLY with a JSON object:
- If consistent: {{"consistent": true, "number_table_violations": [], "arithmetic_errors": []}}
- If inconsistent: {{"consistent": false, \
"number_table_violations": [{{"section": "...", "field": "compute.total_gpus", "table_value": "24", "section_value": "32"}}], \
"arithmetic_errors": [{{"section": "...", "expression": "16×80=1280GB", "claimed_value": "2TB", "correct_value": "1280GB"}}], \
"fixes": [{{"section": "methodology", "old_text": "exact text to replace", "new_text": "corrected text"}}], \
"verified_number_table": {{}} \
}}

Each fix must use exact text matching (old_text must appear verbatim in the section).
Only include fixes for genuine inconsistencies, not style preferences.
If the number table is "(none)", skip the number table checks (items 1-2).

Reply ONLY with valid JSON. No markdown, no explanation."""


def _extract_number_table(sections: dict, cfg: dict) -> dict | None:
    """Extract structured number table from all sections via 1 LLM call.

    Returns parsed dict or None on failure (pipeline degrades gracefully).
    """
    prompt = _NUMBER_TABLE_EXTRACT_PROMPT.format(
        methodology=_truncate(sections.get("methodology", ""), 4000),
        experimental_design=_truncate(sections.get("experimental_design", ""), 3000),
        expected_results=_truncate(sections.get("expected_results", ""), 2000),
        timeline=_truncate(sections.get("timeline", ""), 1000),
    )
    raw = call_cli(prompt, cfg, timeout=120)
    if not raw:
        log.warning("Number table extraction failed (no LLM response)")
        return None

    parsed = _parse_reviewer_json(raw)
    if not isinstance(parsed, dict):
        log.warning("Number table extraction failed (unparseable JSON)")
        return None

    # Auto-correct arithmetic: total_gpus, total_gpu_memory
    compute = parsed.get("compute", {})
    if isinstance(compute, dict):
        gpus_per_node = compute.get("gpus_per_node")
        num_nodes = compute.get("num_nodes")
        if gpus_per_node and num_nodes:
            correct_total = int(gpus_per_node) * int(num_nodes)
            if compute.get("total_gpus") and int(compute["total_gpus"]) != correct_total:
                log.info("Number table auto-fix: total_gpus %s → %d",
                         compute["total_gpus"], correct_total)
            compute["total_gpus"] = correct_total

        total_gpus = compute.get("total_gpus")
        gpu_mem = compute.get("gpu_memory_gb")
        if total_gpus and gpu_mem:
            correct_mem = int(total_gpus) * int(gpu_mem)
            if compute.get("total_gpu_memory_gb") and int(compute["total_gpu_memory_gb"]) != correct_mem:
                log.info("Number table auto-fix: total_gpu_memory_gb %s → %d",
                         compute["total_gpu_memory_gb"], correct_mem)
            compute["total_gpu_memory_gb"] = correct_mem

    log.info("Number table extracted: %d top-level keys", len(parsed))
    return parsed


def _format_number_table_for_prompt(number_table: dict | None) -> str:
    """Format number table as Markdown text for injection into global_constraints."""
    if not number_table:
        return ""

    lines = ["**Authoritative Number Table** (all sections MUST use these exact values):"]

    compute = number_table.get("compute", {})
    if compute:
        lines.append("\nCompute Budget:")
        for k, v in compute.items():
            lines.append(f"  - {k}: {v}")

    model = number_table.get("model", {})
    if model:
        lines.append("\nModel:")
        for k, v in model.items():
            lines.append(f"  - {k}: {v}")

    training = number_table.get("training", {})
    if training:
        lines.append("\nTraining:")
        for k, v in training.items():
            lines.append(f"  - {k}: {v}")

    claims = number_table.get("performance_claims", [])
    if claims:
        lines.append("\nPerformance Claims:")
        if isinstance(claims, dict):
            for k, v in claims.items():
                lines.append(f"  - {k}: {v}")
        else:
            for c in claims:
                if isinstance(c, dict):
                    lines.append(f"  - {c.get('claim', '?')} ({c.get('metric', '?')}, {c.get('conditions', '')})")
                else:
                    lines.append(f"  - {c}")

    datasets = number_table.get("datasets", [])
    if datasets:
        lines.append("\nDatasets:")
        for d in datasets:
            if isinstance(d, dict):
                lines.append(f"  - {d.get('name', '?')}: {d.get('size', '?')} (split: {d.get('split', '?')})")
            else:
                lines.append(f"  - {d}")

    hyperparams = number_table.get("key_hyperparameters", [])
    if hyperparams:
        lines.append("\nKey Hyperparameters:")
        for h in hyperparams:
            if isinstance(h, dict):
                lines.append(f"  - {h.get('name', '?')} = {h.get('value', '?')} (from {h.get('section_defined', '?')})")
            else:
                lines.append(f"  - {h}")

    discrepancies = number_table.get("discrepancies", [])
    if discrepancies:
        lines.append("\nDiscrepancies Found (MUST be resolved):")
        for d in discrepancies:
            if isinstance(d, dict):
                lines.append(f"  - {d.get('field', '?')}: {d.get('values', {})} → {d.get('recommendation', '?')}")
            else:
                lines.append(f"  - {d}")

    return "\n".join(lines)


def _apply_planner_number_updates(number_table: dict, updates: dict) -> dict:
    """Apply dot-notation path updates to the number table.

    Example: updates = {"compute.num_nodes": 2, "training.batch_size": 512}
    """
    if not updates or not number_table:
        return number_table

    for path, value in updates.items():
        parts = path.split(".")
        obj = number_table
        for part in parts[:-1]:
            if part not in obj or not isinstance(obj[part], dict):
                obj[part] = {}
            obj = obj[part]
        old_val = obj.get(parts[-1])
        obj[parts[-1]] = value
        log.info("Number table update: %s: %s → %s", path, old_val, value)

    # Re-run arithmetic corrections after updates
    compute = number_table.get("compute", {})
    if isinstance(compute, dict):
        gpus_per_node = compute.get("gpus_per_node")
        num_nodes = compute.get("num_nodes")
        if gpus_per_node and num_nodes:
            compute["total_gpus"] = int(gpus_per_node) * int(num_nodes)
        total_gpus = compute.get("total_gpus")
        gpu_mem = compute.get("gpu_memory_gb")
        if total_gpus and gpu_mem:
            compute["total_gpu_memory_gb"] = int(total_gpus) * int(gpu_mem)

    return number_table


def _apply_cross_section_fixes(result: dict, fixes: list[dict]) -> int:
    """Apply cross-section verification fixes via string replacement.

    Returns the number of fixes successfully applied.
    """
    applied = 0
    for fix in fixes:
        section = fix.get("section", "")
        old_text = fix.get("old_text", "")
        new_text = fix.get("new_text", "")
        if section in result and old_text and new_text and old_text in result[section]:
            result[section] = result[section].replace(old_text, new_text, 1)
            applied += 1
            log.info("  Cross-section fix applied to '%s': %d chars replaced",
                     section, len(old_text))
        elif section in result and old_text:
            log.warning("  Cross-section fix for '%s' could not be applied (text not found)",
                        section)
    return applied


def refine_research_plan(
    topic_id: str,
    topic_name: str,
    data_dir: str,
    cfg: dict,
    existing_plan: dict,
    user_feedback: str,
    sections_to_refine: list[str] | None = None,
    *,
    on_section_done: callable | None = None,
) -> dict:
    """Refine an existing research plan using structured revision pipeline.

    New pipeline:
      Step 0: Build review history (pure parsing, zero LLM)
      Step 1: Refine Planner — generate global modification plan (1 LLM call)
      Step 2: Sequential section revision + self-verify
      Step 3: Cross-section verification (1 LLM call)
      Step 4: Peer review with history (7 LLM calls, parallel)
      Step 5: Assemble markdown

    Args:
        existing_plan: Dict with all current section content.
        user_feedback: Free-text user instructions for improvement.
        sections_to_refine: List of section keys to revise. If None, refines all.
        on_section_done: Optional callback(section_name, content).

    Returns:
        Updated dict with revised sections + new review.
    """
    log.info("=== Research Plan refinement started ===")

    result = {k: existing_plan.get(k, "") for k in [
        "introduction", "related_work", "methodology",
        "experimental_design", "expected_results", "timeline", "review",
    ]}

    title = existing_plan.get("idea_title", "")
    if isinstance(existing_plan.get("idea_json"), dict):
        title = title or existing_plan["idea_json"].get("title", "Untitled")

    # Get idea for paper curation
    idea = existing_plan.get("idea_json", {})
    if isinstance(idea, str):
        try:
            idea = json.loads(idea)
        except Exception:
            idea = {}

    review = existing_plan.get("review", "")

    # Load and curate paper context for refinement
    store = Storage(data_dir, topic_id)
    try:
        papers, _ = store.get_all_arxiv(limit=200, offset=0)
    finally:
        store.close()

    paper_context = ""
    if papers and idea:
        paper_context = curate_papers_for_idea(papers, idea, cfg)
    paper_context_truncated = _truncate(paper_context, 3000)

    all_sections = ["introduction", "related_work", "methodology",
                    "experimental_design", "expected_results", "timeline"]

    def _notify(section: str, content: str) -> None:
        if on_section_done:
            try:
                on_section_done(section, content)
            except Exception:
                pass

    # Token budget for _call wrapper
    MAX_PROMPT_CHARS = 950_000 * 4

    def _call(prompt: str, stage: str, **kwargs) -> str | None:
        prompt_chars = len(prompt)
        est_tokens = prompt_chars // 4
        log.info("  [%s] prompt: %d chars (~%dk tokens)", stage, prompt_chars, est_tokens // 1000)
        if prompt_chars > MAX_PROMPT_CHARS:
            log.warning("  [%s] OVER BUDGET: %d chars > %d max", stage, prompt_chars, MAX_PROMPT_CHARS)
        return call_cli(prompt, cfg, **kwargs)

    # --- Step 0: Build review history (pure parsing, zero LLM) ---
    log.info("Step 0: Building review history from previous round")
    review_history = _summarize_review_for_history(review)
    if review_history:
        log.info("Review history: %d chars", len(review_history))
    else:
        review_history = "(First review round — no prior history)"

    # --- Step 1: Refine Planner (1 LLM call) ---
    log.info("Step 1: Generating global revision plan")
    section_summaries = "\n\n".join(
        f"### {k.replace('_', ' ').title()}\n{_truncate(result[k], 800)}"
        for k in all_sections if result.get(k)
    )

    # --- Step 1.5: Extract Number Table (1 LLM call) ---
    log.info("Step 1.5: Extracting authoritative number table")
    number_table = _extract_number_table(result, cfg)
    number_table_text = _format_number_table_for_prompt(number_table) or "(none)"
    if number_table:
        log.info("Number table: %d top-level keys", len(number_table))
    else:
        log.info("Number table extraction failed — proceeding without it")

    planner_prompt = _REFINE_PLANNER_PROMPT.format(
        section_summaries=section_summaries,
        user_feedback=user_feedback,
        review_tagged=_truncate(review, 4000),
        number_table_text=number_table_text,
    )
    planner_raw = _call(planner_prompt, "refine_planner", timeout=180)

    # Parse planner output
    revision_plan = []
    global_constraints = []
    if planner_raw:
        planner_json = _parse_reviewer_json(planner_raw)  # reuse JSON extractor
        if planner_json:
            global_constraints = planner_json.get("global_constraints", [])
            revision_plan = planner_json.get("revision_plan", [])

            # --- Step 1.6: Apply planner number updates (pure code) ---
            number_updates = planner_json.get("number_updates", {})
            if number_updates and number_table:
                log.info("Applying %d number updates from planner", len(number_updates))
                number_table = _apply_planner_number_updates(number_table, number_updates)
                number_table_text = _format_number_table_for_prompt(number_table)

            log.info("Revision plan: %d global constraints, %d section plans",
                     len(global_constraints), len(revision_plan))
            _notify("_planner", json.dumps(planner_json, indent=2))

    if not revision_plan:
        log.warning("Planner failed or returned empty plan, falling back to all sections")
        # Fallback: revise all sections with basic instructions
        to_refine = sections_to_refine if sections_to_refine else all_sections
        fallback_instr = "Address review critiques for this section."
        if user_feedback:
            fallback_instr += f" User feedback: {user_feedback}"
        revision_plan = [
            {
                "section": s, "priority": i + 1,
                "weaknesses_to_address": [],
                "instructions": fallback_instr,
                "invariants": [],
            }
            for i, s in enumerate(to_refine) if result.get(s)
        ]

    # Sort by priority
    revision_plan.sort(key=lambda x: x.get("priority", 99))

    # --- Step 1.7: Inject number table into global constraints (pure code) ---
    global_constraints_text = "\n".join(f"- {c}" for c in global_constraints) or "(none)"
    if number_table_text and number_table_text != "(none)":
        global_constraints_text += f"\n\n{number_table_text}"

    # --- Step 2: Sequential section revision + self-verify ---
    log.info("Step 2: Sequential section revision (%d sections)", len(revision_plan))
    revised_sections = {}  # section → revised content (for cross-reference)

    for plan_item in revision_plan:
        section = plan_item.get("section", "")
        if section not in result or not result[section]:
            log.warning("Skipping unknown/empty section: %s", section)
            continue

        instructions = plan_item.get("instructions", "")
        invariants = plan_item.get("invariants", [])
        weaknesses = plan_item.get("weaknesses_to_address", [])

        log.info("  Revising section: %s (priority=%d, weaknesses=%s)",
                 section, plan_item.get("priority", 0), weaknesses)

        # Build already-revised context
        already_revised_text = ""
        if revised_sections:
            parts = []
            for s, content in revised_sections.items():
                parts.append(f"### {s.replace('_', ' ').title()}\n{_truncate(content, 1500)}")
            already_revised_text = "\n\n".join(parts)

        # --- Step 2a: Revise section ---
        revise_prompt = _REFINE_SECTION_TARGETED_PROMPT.format(
            global_constraints=global_constraints_text,
            revision_instructions=instructions,
            current_content=result[section],
            already_revised=already_revised_text or "(No other sections revised yet)",
            paper_context=paper_context_truncated,
        )
        revised = _call(revise_prompt, f"revise_{section}", timeout=360)

        if not revised or len(revised) < 100:
            log.warning("  Section revision failed for %s, keeping original", section)
            revised_sections[section] = result[section]
            continue

        revised = _strip_leading_heading(revised)

        # --- Step 2b: Self-verify ---
        log.info("  Self-verifying section: %s", section)
        verify_prompt = _SECTION_SELF_VERIFY_PROMPT.format(
            global_constraints=global_constraints_text,
            invariants="\n".join(f"- {inv}" for inv in invariants) or "(none)",
            revised_content=revised,
            already_revised=already_revised_text or "(No other sections revised yet)",
        )
        verify_raw = _call(verify_prompt, f"verify_{section}", timeout=120)

        if verify_raw:
            verify_json = _parse_reviewer_json(verify_raw)
            if verify_json and not verify_json.get("passed", True):
                issues = verify_json.get("issues", [])
                fixed = verify_json.get("fixed_content", "")
                log.info("  Self-verify FAILED for %s: %d issues", section, len(issues))
                for issue in issues:
                    log.info("    Issue: %s", issue[:100])
                if fixed and len(fixed) > 100:
                    revised = _strip_leading_heading(fixed)
                    log.info("  Applied self-verify fix: %d chars", len(revised))
            else:
                log.info("  Self-verify PASSED for %s", section)

        # --- Step 2c: Update result ---
        result[section] = revised
        revised_sections[section] = revised
        _notify(section, result[section])

    log.info("Step 2 done: revised %d sections", len(revised_sections))

    # --- Step 3: Cross-section verification (1 LLM call) ---
    log.info("Step 3: Cross-section verification (with number table)")
    number_table_json_str = json.dumps(number_table, indent=2) if number_table else "(none)"
    cross_prompt = _CROSS_SECTION_VERIFY_PROMPT.format(
        number_table_json=number_table_json_str,
        introduction=_truncate(result.get("introduction", ""), 2000),
        related_work=_truncate(result.get("related_work", ""), 2000),
        methodology=_truncate(result.get("methodology", ""), 3000),
        experimental_design=_truncate(result.get("experimental_design", ""), 2500),
        expected_results=_truncate(result.get("expected_results", ""), 1500),
        timeline=_truncate(result.get("timeline", ""), 1000),
    )
    cross_raw = _call(cross_prompt, "cross_section_verify", timeout=180)

    if cross_raw:
        cross_json = _parse_reviewer_json(cross_raw)
        if cross_json and not cross_json.get("consistent", True):
            # --- Step 3.5: Log violations + update verified number table ---
            violations = cross_json.get("number_table_violations", [])
            arith_errors = cross_json.get("arithmetic_errors", [])
            if violations:
                log.info("  Number table violations: %d", len(violations))
                for v in violations:
                    log.info("    %s.%s: table=%s, section=%s",
                             v.get("section", "?"), v.get("field", "?"),
                             v.get("table_value", "?"), v.get("section_value", "?"))
            if arith_errors:
                log.info("  Arithmetic errors: %d", len(arith_errors))
                for e in arith_errors:
                    log.info("    %s: %s (claimed %s, correct %s)",
                             e.get("section", "?"), e.get("expression", "?"),
                             e.get("claimed_value", "?"), e.get("correct_value", "?"))

            # Update number table if verifier provided a corrected version
            verified_table = cross_json.get("verified_number_table")
            if verified_table and isinstance(verified_table, dict) and verified_table:
                number_table = verified_table
                log.info("  Updated number table from verifier")

            fixes = cross_json.get("fixes", [])
            applied = _apply_cross_section_fixes(result, fixes)
            log.info("Step 3 done: %d/%d cross-section fixes applied", applied, len(fixes))
            if applied > 0:
                for section in revised_sections:
                    _notify(section, result[section])
        else:
            log.info("Step 3 done: cross-section check passed")
    else:
        log.info("Step 3: cross-section verification failed, skipping")

    # --- Step 4: Peer review with history (7 LLM calls, parallel) ---
    log.info("Step 4: Peer review with history")
    new_review = _run_peer_review(title, result, idea, cfg, _call,
                                   review_history=review_history)
    result["review"] = new_review
    _notify("review", result["review"])

    # --- Step 5: Assemble markdown ---
    result["full_markdown"] = _assemble_full_markdown(title, result)

    log.info("=== Research Plan refinement finished ===")
    return result


def _extract_review_for_section(review: str, section: str) -> str:
    """Extract review comments relevant to a particular section."""
    section_keywords = {
        "introduction": ["introduction", "motivation", "claim", "contribution"],
        "related_work": ["related work", "prior work", "literature", "baseline"],
        "methodology": ["method", "algorithm", "proof", "soundness", "mathematical", "theoretical"],
        "experimental_design": ["experiment", "baseline", "dataset", "metric", "ablation", "reproducib"],
        "expected_results": ["result", "expected", "risk", "limitation", "failure"],
        "timeline": ["timeline", "milestone", "schedule"],
    }
    keywords = section_keywords.get(section, [])
    if not keywords or not review:
        return ""

    relevant_lines = []
    lines = review.split("\n")
    for i, line in enumerate(lines):
        if any(kw in line.lower() for kw in keywords):
            # Include context: 1 line before, the match, 2 lines after
            start = max(0, i - 1)
            end = min(len(lines), i + 3)
            relevant_lines.extend(lines[start:end])
            relevant_lines.append("")

    return "\n".join(relevant_lines[:60])  # Cap at 60 lines


def _assemble_full_markdown(title: str, sections: dict) -> str:
    """Assemble all sections into a single Markdown document."""
    parts = [f"# Research Plan: {title}"]

    if sections.get("introduction"):
        parts.append(f"\n## 1. Introduction & Motivation\n\n{sections['introduction']}")

    if sections.get("related_work"):
        parts.append(f"\n## 2. Related Work\n\n{sections['related_work']}")

    if sections.get("methodology"):
        parts.append(f"\n## 3. Methodology\n\n{sections['methodology']}")

    if sections.get("experimental_design"):
        parts.append(f"\n## 4. Experimental Design\n\n{sections['experimental_design']}")

    if sections.get("expected_results"):
        parts.append(f"\n## 5. Expected Results & Risk Analysis\n\n{sections['expected_results']}")

    if sections.get("timeline"):
        parts.append(f"\n## 6. Timeline & Milestones\n\n{sections['timeline']}")

    if sections.get("review"):
        parts.append(f"\n---\n\n## Appendix: Peer Review Simulation\n\n{sections['review']}")

    return "\n".join(parts)
