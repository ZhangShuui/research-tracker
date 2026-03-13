"""Brainstorm pipeline: idea generation + multi-stage verification.

Design informed by:
- AI Scientist v2 (Sakana AI): agentic tree search for idea generation
- Stanford "Can LLMs Generate Novel Research Ideas?": structured idea template
- Google Co-Scientist: generate-debate-evolve pattern
- Chain of Ideas (CoI): progressive literature chain for trend extrapolation
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from paper_tracker.llm import call_cli, call_codex, call_copilot
from paper_tracker.sources.arxiv import search_by_query
from paper_tracker.sources.web import gather_perspectives
from paper_tracker.storage import Storage

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stage 1: Idea Generation Prompts
# ---------------------------------------------------------------------------

_AUTO_IDEAS_PROMPT = """\
You are a senior research scientist brainstorming novel research ideas.

## Research Topic
"{topic_name}"

## Current Paper Library ({n_papers} papers)
Below are structured summaries of papers already in the library. Study the \
progression of ideas — where the field started, where it is now, and what gaps \
remain.

{paper_summaries}

{discovery_context}

{extra_context}

## Your Task
Generate 3-5 **novel, concrete research ideas** that could lead to publishable \
work. For EACH idea, provide:

1. "title": A concise, specific paper title (not generic)
2. "problem": The specific problem or limitation in current work this addresses \
(1-2 sentences)
3. "motivation": Why this problem matters and why NOW is the right time to solve it \
(2-3 sentences). Reference specific papers from the library to ground your argument.
4. "method": A detailed technical approach (3-5 sentences). Name specific architectures, \
loss functions, training strategies, or algorithms. Be concrete enough that a PhD \
student could start implementing.
5. "experiment_plan": How to validate this idea (2-4 sentences). Specify datasets, \
baselines, metrics, and what a successful result looks like.
6. "novelty_score": 1-10 (10 = completely new direction, 1 = incremental improvement)
7. "feasibility_score": 1-10 (10 = can be done in 1 week with existing tools, \
1 = requires fundamental breakthroughs)

## Strategies for Novel Ideas
- **Cross-pollination**: Combine techniques from different papers that haven't been \
combined before
- **Failure analysis**: What assumptions in existing work might be wrong?
- **Scale shift**: What works at one scale but hasn't been tried at another?
- **Modality transfer**: What works in one domain but could be adapted to another?
- **Simplification**: Is there an unnecessarily complex step that could be replaced?

## CRITICAL: Diversity Requirement
Each idea MUST tackle a DIFFERENT core problem or use a fundamentally DIFFERENT \
approach. Do NOT generate multiple ideas that address the same problem with minor \
variations (e.g., different loss functions for the same task). If two ideas share \
>50% of their method, merge them into one and generate a new distinct idea instead.

Reply ONLY with a JSON array of idea objects. No markdown, no explanation."""


# ---------------------------------------------------------------------------
# Stage 0: Parallel Multi-CLI Research Prompts
# ---------------------------------------------------------------------------

_RESEARCH_GAP_ANALYSIS_PROMPT = """\
You are a senior research strategist analyzing gaps in a body of literature.

## Research Topic
"{topic_name}"

## Paper Library ({n_papers} papers)
{paper_summaries}

{discovery_context}

## Your Task
Perform a systematic gap analysis across 5 dimensions:

1. **Methodological Gaps**: What techniques are underexplored? Are there standard \
methods from adjacent fields that haven't been applied here?
2. **Evaluation Gaps**: What metrics, datasets, or benchmarks are missing? Are \
current evaluations insufficient to prove the claims being made?
3. **Scale Gaps**: What has only been shown at small scale? What breaks when you \
scale up/down? What settings are never tested (e.g., low-resource, edge devices)?
4. **Theoretical Gaps**: What lacks formal analysis? Are there empirical results \
without theoretical explanation? Are there assumptions that haven't been validated?
5. **Application Gaps**: What real-world use cases are unaddressed? Where is the \
gap between research demos and production systems widest?

For EACH gap, provide:
- "dimension": one of the 5 categories above
- "description": 2-3 sentences describing the gap precisely
- "evidence": which papers (by title) reveal this gap
- "opportunity_level": HIGH / MEDIUM / LOW — how promising is this for new research?
- "why_now": why is this gap addressable NOW but wasn't before?

Reply ONLY with a JSON object:
{{
  "gaps": [list of gap objects],
  "meta_observation": "2-3 sentences about the overall state of the field — \
what is the biggest unexplored frontier?"
}}
No markdown fences."""

_RESEARCH_CROSS_DOMAIN_PROMPT = """\
You are a cross-disciplinary research consultant identifying techniques from \
OTHER fields that could be transformative when applied to this domain.

## Research Topic
"{topic_name}"

## Techniques Currently Used in This Field
{technique_summary}

## Your Task
Identify 5-8 techniques from OUTSIDE this specific field that could be impactful. \
Consider techniques from:
- Control theory & dynamical systems
- Information theory & coding theory
- Spectral methods & harmonic analysis
- Causal inference & counterfactual reasoning
- Game theory & mechanism design
- Online learning & bandit algorithms
- Topological data analysis
- Compiler optimization & program synthesis
- Neuroscience-inspired computation
- Statistical physics & thermodynamics

For EACH technique, provide:
- "technique_name": name of the technique
- "source_field": where it comes from
- "description": 2-3 sentences on how it works
- "application_angle": 2-3 sentences on HOW it could be applied to this topic
- "novelty_potential": HIGH / MEDIUM / LOW
- "closest_existing_work": any papers in AI/ML that have started using this \
(if you know of any)

Reply ONLY with a JSON object:
{{
  "cross_domain_techniques": [list of technique objects],
  "synthesis": "2-3 sentences on which combination of techniques could yield \
the most novel contribution"
}}
No markdown fences."""

_RESEARCH_PRIOR_ART_LANDSCAPE_PROMPT = """\
You are a research landscape analyst mapping the maturity of sub-areas within \
a research topic.

## Research Topic
"{topic_name}"

## Paper Library ({n_papers} papers)
{paper_summaries}

## Your Task
Decompose this topic into 4-6 distinct sub-areas/threads and assess each:

For EACH sub-area:
- "sub_area": descriptive name
- "representative_papers": list of 2-3 paper titles from the library
- "maturity": NASCENT / GROWING / MATURE / SATURATED
- "key_techniques": list of main methods used
- "open_questions": 2-3 unresolved questions in this sub-area
- "trend_direction": ACCELERATING / STEADY / DECLINING / PIVOTING
- "competition_density": HIGH / MEDIUM / LOW — how many groups are working on this?

Then provide:
- "white_spaces": 2-3 areas between or beyond current sub-areas that are \
completely unexplored
- "contrarian_opportunities": 1-2 directions that go against the current trend \
but might be valuable

Reply ONLY with a JSON object:
{{
  "sub_areas": [list of sub-area objects],
  "white_spaces": [list of white-space descriptions],
  "contrarian_opportunities": [list of contrarian descriptions],
  "landscape_summary": "3-4 sentences summarizing where the most promising \
unexplored territory lies"
}}
No markdown fences."""


# ---------------------------------------------------------------------------
# Stage 1 (enhanced): Research-Informed Idea Generation
# ---------------------------------------------------------------------------

_RESEARCH_INFORMED_IDEAS_PROMPT = """\
You are a senior research scientist brainstorming novel research ideas, informed \
by deep research analysis.

## Research Topic
"{topic_name}"

## Current Paper Library ({n_papers} papers)
{paper_summaries}

{discovery_context}

{extra_context}

## Deep Research Context
The following research was conducted by specialized research agents:

{research_context}

## Your Task
Generate 3-5 **genuinely novel research ideas** that could lead to top-venue \
publications. Use the research context above to ground your ideas in REAL gaps, \
cross-domain opportunities, and unexplored territory.

For EACH idea, provide:

1. "title": A concise, specific paper title (not generic)
2. "problem": The specific problem or limitation this addresses (1-2 sentences)
3. "motivation": Why this matters and why NOW (2-3 sentences). Reference specific \
papers AND research gaps identified above.
4. "method": A detailed technical approach (3-5 sentences). Name specific \
architectures, loss functions, training strategies, or algorithms.
5. "experiment_plan": Validation plan (2-4 sentences). Datasets, baselines, \
metrics, success criteria.
6. "novelty_score": 1-10 with calibration:
   - 1-3: Incremental — straightforward extension of existing work
   - 4-5: Modest — new combination but each piece is known
   - 6-7: Significant — non-obvious insight connecting distant ideas
   - 8-9: Major — new paradigm or framework
   - 10: Revolutionary — fundamentally new way of thinking
7. "feasibility_score": 1-10
8. "novelty_reasoning": 3-5 sentences justifying the novelty score. What exactly \
is new? NOT "we combine X and Y" — explain the NON-OBVIOUS insight.
9. "closest_prior_work": The single most similar existing paper/approach and how \
this differs.
10. "what_is_genuinely_new": One sentence capturing the core novel contribution \
that doesn't exist anywhere.
11. "research_gap_addressed": Which gap from the research context this idea targets.

## CRITICAL: Novelty Self-Check
Before finalizing each idea, apply the **X+Y Test**:
- Can this idea be summarized as "apply technique X to domain Y"?
- If YES → the novelty score MUST be ≤ 5 unless there is a genuinely non-obvious \
insight in HOW X is adapted.
- The insight must be in the CONNECTION, not just the combination.

## Strategies for Novel Ideas
- **Gap exploitation**: Directly address gaps from the research analysis
- **Cross-domain transfer**: Import techniques from the cross-domain research, \
but with a non-trivial adaptation
- **White space exploration**: Target the unexplored territory identified
- **Contrarian bets**: Go against current trends with good reason
- **Failure analysis**: What assumptions in existing work might be wrong?

## CRITICAL: Diversity Requirement
Each idea MUST tackle a DIFFERENT core problem or use a fundamentally DIFFERENT \
approach. Do NOT generate multiple ideas that address the same problem with minor \
variations. If two ideas share >50% of their method, merge them into one and \
generate a new distinct idea instead.

Reply ONLY with a JSON array of idea objects. No markdown, no explanation."""


_USER_IDEA_PROMPT = """\
You are a senior research advisor evaluating and refining a research idea.

## Research Topic
"{topic_name}"

## Existing Paper Library ({n_papers} papers)
{paper_summaries}

{extra_context}

## User's Research Idea
{user_idea}

## Your Task
Evaluate this idea rigorously, then produce a refined and strengthened version.

Provide your response as a JSON object with these fields:

1. "title": A polished paper title for this idea
2. "problem": The problem statement, refined for clarity (1-2 sentences)
3. "motivation": Strengthened motivation grounded in the existing literature \
(2-3 sentences). Reference specific papers.
4. "method": A detailed, concrete technical approach (3-5 sentences). Fill in \
any gaps the user left. Name architectures, algorithms, loss functions.
5. "experiment_plan": A rigorous validation plan (2-4 sentences). Specify \
datasets, baselines, metrics, expected outcomes.
6. "novelty_score": 1-10 with justification
7. "feasibility_score": 1-10 with justification
8. "strengths": List of 2-3 strengths of the original idea
9. "weaknesses": List of 2-3 weaknesses or risks
10. "suggestions": List of 2-3 concrete improvements

Reply ONLY with a JSON object (not array). No markdown, no explanation."""


# ---------------------------------------------------------------------------
# Stage 2: Literature Verification Prompt
# ---------------------------------------------------------------------------

_LITERATURE_CHECK_PROMPT = """\
You are a thorough literature reviewer checking for novelty.

## Research Ideas to Verify
{ideas_text}

## Existing Paper Library
{paper_summaries}

## Your Task
For EACH idea, determine if similar work already exists in the paper library. \
Check for:
- Identical or near-identical approaches
- Same problem addressed with similar methods
- Work that would make this idea redundant

For each idea, provide:
- "title": The idea title
- "verdict": "NOVEL" | "PARTIALLY_NOVEL" | "ALREADY_EXISTS"
- "evidence": Specific papers that support your verdict (reference titles)
- "differentiation": If PARTIALLY_NOVEL, what specifically is new vs. existing
- "prior_art_gap": What the existing papers DON'T do that this idea DOES

Reply ONLY with a JSON array of objects. No markdown."""


# ---------------------------------------------------------------------------
# Stage 3: Logic Verification Prompt
# ---------------------------------------------------------------------------

_LOGIC_CHECK_PROMPT = """\
You are a rigorous peer reviewer and research critic. Your job is to stress-test \
research ideas before they waste anyone's time.

## Ideas to Critique
{ideas_text}

## Multi-Angle Challenge
For each idea, analyze from these perspectives:

### 1. Assumption Analysis
- What are the implicit assumptions? Which are most likely to be wrong?
- Is there empirical evidence supporting the core hypothesis?

### 2. Scalability & Practicality
- Will this work scale? What happens with 10x more data/parameters?
- What are the computational requirements? Is it practical?

### 3. Edge Cases & Failure Modes
- Where will this approach break? What inputs will cause failure?
- What happens when the data distribution shifts?

### 4. Alternative Approaches
- Is there a simpler way to achieve the same goal?
- Why would this approach be better than the obvious baseline?

### 5. Experimental Rigor
- Is the proposed evaluation sufficient to prove the claims?
- What confounding variables should be controlled for?
- What ablation studies are needed?

### 6. Suggested Next Steps
- What is the single most important experiment to run first?
- What would convince a skeptical reviewer?

Provide your analysis as a Markdown document with clear headings for each idea. \
Be critical but constructive — point out problems AND suggest fixes."""


# ---------------------------------------------------------------------------
# Stage 4: Code Verification Prompt (optional)
# ---------------------------------------------------------------------------

_CODE_CHECK_PROMPT = """\
You are a research engineer. Write a minimal Python proof-of-concept to test \
the feasibility of this research idea.

## Idea
{idea_text}

## Requirements
- Write self-contained Python code (use numpy, scipy, torch if needed)
- The PoC should test the CORE hypothesis, not implement the full system
- Include synthetic data generation if real data isn't available
- Print clear output showing whether the hypothesis holds
- Add comments explaining what each section tests
- Keep it under 100 lines
- End with a clear PASS/FAIL assessment

Reply ONLY with Python code. No markdown fences, no explanation outside the code."""


# ---------------------------------------------------------------------------
# Stage 1b/1c: Idea Review & Refinement Prompts
# ---------------------------------------------------------------------------

_IDEA_REVIEW_PROMPT = """\
You are a rigorous research review committee evaluating brainstorm ideas for \
their potential as publishable research contributions.

## Research Topic
"{topic_name}"

## Paper Library ({n_papers} papers)
{paper_summaries}

## Ideas to Review
{ideas_text}

## Your Task
For EACH idea, evaluate 4 dimensions on a 1-10 scale:

- **novelty**: Is the core idea genuinely new, or an incremental combination of existing \
techniques? Apply the "X + Y" test: could the contribution be summarized as combining two \
known things (X + Y) without a non-obvious insight? Check whether the idea clearly \
differentiates from the closest prior work in the library. Do the components form a \
coherent, unified contribution or a bag of tricks? \
(1 = well-known combination, 5 = incremental twist on existing work, 10 = paradigm shift)

- **feasibility**: How practical to implement and validate? Consider: can this be done \
with 2-3 nodes of 8×H800 GPUs? Does it require fundamental breakthroughs or just \
engineering effort? Are the proposed datasets and baselines available? \
(1 = needs breakthroughs, 5 = moderate engineering, 10 = straightforward with existing tools)

- **clarity**: How well-defined is the problem, method, and evaluation plan? \
Is the problem well-motivated — does it convince you this NEEDS solving right now? \
Is the method concrete enough that a PhD student could start implementing? \
(1 = vague hand-waving, 5 = reasonable but gaps remain, 10 = implementation-ready spec)

- **impact**: If successful, would the results matter to the community? Would \
practitioners adopt this? Would it open new research directions or is it a marginal \
improvement on a niche problem? \
(1 = marginal improvement, 5 = useful to specialists, 10 = field-changing)

Then assign a verdict:
- **ACCEPT**: Idea is strong enough to proceed directly (overall >= 7, no dimension below 5)
- **REVISE**: Idea has potential but needs specific improvements (overall 5-7)
- **DROP**: Idea has fundamental issues that revision cannot fix (overall < 5 or any \
dimension <= 2)

For REVISE verdicts, provide specific, actionable revision_instructions. Focus especially \
on novelty weaknesses — these are hardest to fix at the research plan stage.

Reply ONLY with a JSON array (no markdown):
[{{
  "idea_title": "...",
  "novelty": 7, "feasibility": 8, "clarity": 6, "impact": 7,
  "overall": 7.0,
  "weaknesses": ["weakness 1", "weakness 2"],
  "strengths": ["strength 1"],
  "verdict": "REVISE",
  "revision_instructions": ["Replace X with Y", "Add concrete baseline Z"]
}}]"""

_REVIEW_FEASIBILITY_PROMPT = """\
You are Reviewer A — an experienced research engineer evaluating the **practical \
feasibility and specification clarity** of brainstorm ideas. You do NOT judge \
novelty or impact (a separate reviewer handles those).

## Research Topic
"{topic_name}"

## Paper Library ({n_papers} papers)
{paper_summaries}

## Ideas to Review
{ideas_text}

## Your Task
For EACH idea, evaluate exactly 2 dimensions on a 1-10 scale:

- **feasibility**: How practical is this to implement and validate?
  Consider: GPU resource requirements (assume 2-3 nodes of 8×H800), dataset \
availability, implementation complexity, whether proposed baselines exist, \
engineering effort vs. fundamental research breakthroughs needed.
  (1 = needs fundamental breakthroughs, 5 = moderate engineering, \
10 = straightforward with existing tools)

- **clarity**: How well-defined is the problem, method, and evaluation plan?
  Is the problem well-motivated — does it convince you this NEEDS solving? \
Is the method concrete enough that a PhD student could start implementing \
within a week? Is the experiment plan specific (datasets, metrics, baselines)?
  (1 = vague hand-waving, 5 = reasonable but gaps remain, \
10 = implementation-ready spec)

For each dimension, also provide a list of specific weaknesses.

Reply ONLY with a JSON array (no markdown):
[{{
  "idea_title": "...",
  "feasibility": 8, "clarity": 6,
  "f_weaknesses": ["weakness about feasibility 1"],
  "c_weaknesses": ["weakness about clarity 1"]
}}]"""

_REVIEW_NOVELTY_PROMPT = """\
You are Reviewer B — a senior academic evaluating the **novelty and potential \
impact** of brainstorm ideas. You do NOT judge feasibility or clarity (a \
separate reviewer handles those). Your job is to be a discerning but FAIR \
judge of intellectual contribution.

## Research Topic
"{topic_name}"

## Paper Library ({n_papers} papers)
{paper_summaries}

## Ideas to Review
{ideas_text}

## Your Task
For EACH idea, evaluate exactly 2 dimensions on a 1-10 scale:

- **novelty**: Is the core idea genuinely new?
  Apply the "X + Y" test: can the contribution be summarized as "apply X to Y" \
without a non-obvious insight? If yes, novelty ≤ 5 unless the adaptation is \
truly non-trivial. Check differentiation from closest prior work in the library.
  IMPORTANT: Do NOT conflate novelty with feasibility. An idea can be highly \
novel AND hard to implement, or incremental AND easy. Judge novelty INDEPENDENTLY.
  (1 = well-known combination, 5 = incremental twist, 10 = paradigm shift)

- **impact**: If successful, would the results matter?
  Would practitioners adopt this? Would it open new research directions? \
Is the problem important enough that a solution would be widely cited?
  (1 = marginal niche improvement, 5 = useful to specialists, \
10 = field-changing)

Additionally, for EACH idea provide:
- **novelty_diagnosis**: 2-3 sentences analyzing whether this is a genuine \
insight or an X+Y combination. Name X and Y if applicable. Explain what (if \
anything) makes the connection non-obvious.
- **novelty_boost_hint**: If novelty < 6, provide ONE concrete sentence \
suggesting how to break out of the X+Y pattern (e.g., "Instead of applying X \
to Y, consider what happens if you invert the relationship..."). If novelty \
>= 6, set to null.

Reply ONLY with a JSON array (no markdown):
[{{
  "idea_title": "...",
  "novelty": 7, "impact": 8,
  "novelty_diagnosis": "This idea connects A and B through insight C, which is ...",
  "novelty_boost_hint": null,
  "n_weaknesses": ["novelty weakness 1"],
  "i_weaknesses": ["impact weakness 1"]
}}]"""


# ---------------------------------------------------------------------------
# Stage 1.5: Novelty Challenge Prompts
# ---------------------------------------------------------------------------

_CHALLENGE_ASSUMPTION_FLIP_PROMPT = """\
You are an **Assumption Flipper** — a contrarian research thinker who finds \
hidden assumptions and inverts them to reveal non-obvious research directions.

## Research Context
Topic: "{topic_name}"

{research_context}

## Idea to Challenge
Title: {idea_title}
Problem: {idea_problem}
Method: {idea_method}
Experiment: {idea_experiment}

## Your Task
1. Identify 3 **implicit assumptions** that this idea takes for granted. These \
are things the idea never questions — e.g., "data is i.i.d.", "larger models are \
better", "supervised learning is needed", "the loss function should be minimized".

2. For each assumption, **flip it** (negate or invert) and generate a \
**mutation** — an alternative research direction that follows from the flipped \
assumption.

Each mutation should be concrete enough to be a research idea on its own, not \
just a vague direction.

Reply ONLY with a JSON object (no markdown):
{{
  "assumptions": [
    {{
      "assumption": "The original implicit assumption",
      "flipped": "The inverted/negated version",
      "mutation_title": "A concrete research idea title based on the flip",
      "mutation_method": "2-3 sentences describing the technical approach"
    }}
  ]
}}"""

_CHALLENGE_ANALOGICAL_LEAP_PROMPT = """\
You are an **Analogical Leaper** — a cross-disciplinary thinker who finds deep \
structural analogies between distant fields and maps them into novel technical \
approaches.

## Research Context
Topic: "{topic_name}"

{research_context}

## Idea to Challenge
Title: {idea_title}
Problem: {idea_problem}
Method: {idea_method}
Experiment: {idea_experiment}

## Your Task
Find 2 analogies from **completely different fields** (physics, biology, \
economics, ecology, linguistics, material science, game theory, control theory, \
sociology, etc.) that share a deep structural similarity with the problem or \
method of this idea.

For each analogy:
- Explain the structural mapping (what corresponds to what)
- Generate a NEW technical angle that imports the framework from the source \
domain — not just the metaphor, but the actual mathematical/algorithmic structure

Reply ONLY with a JSON object (no markdown):
{{
  "analogies": [
    {{
      "source_domain": "e.g., statistical mechanics",
      "analogy": "2-3 sentences explaining the structural similarity",
      "new_angle_title": "A concrete research idea title",
      "new_angle_method": "2-3 sentences describing the technical approach imported from the source domain"
    }}
  ]
}}"""

_CHALLENGE_CONTRADICTION_PROMPT = """\
You are a **Contradiction Finder** — a rigorous critic who finds internal \
contradictions in research ideas and contradictions with existing literature, \
then converts them into research opportunities.

## Research Context
Topic: "{topic_name}"

## Paper Library
{paper_summaries}

## Idea to Challenge
Title: {idea_title}
Problem: {idea_problem}
Method: {idea_method}
Experiment: {idea_experiment}

## Your Task
Find 2-3 contradictions of these types:

1. **Internal contradiction**: Where does the idea's method contradict its own \
stated goals or assumptions? (e.g., claims to be efficient but uses O(n²) \
attention, claims robustness but assumes clean data)

2. **Literature contradiction**: Which papers in the library present evidence \
that contradicts a key claim or assumption of this idea? What empirical results \
would this idea need to explain away?

For each contradiction, turn it into a research **opportunity** — how could \
resolving the contradiction lead to a stronger, more novel idea?

Reply ONLY with a JSON object (no markdown):
{{
  "contradictions": [
    {{
      "type": "internal" or "literature",
      "description": "2-3 sentences describing the contradiction",
      "opportunity_title": "A concrete research idea title that resolves the contradiction",
      "opportunity_method": "2-3 sentences describing the approach"
    }}
  ]
}}"""

_CHALLENGE_WILD_PERSPECTIVE_PROMPT = """\
You are a **Wild Perspective Injector** — a research provocateur who finds \
insights from OUTSIDE academia: practitioner forums, industry blogs, open-source \
communities, Chinese tech platforms, and real-world deployment stories.

Your job is to find angles that academics MISS because they live inside the \
paper-publishing bubble.

## Research Context
Topic: "{topic_name}"

## Idea to Challenge
Title: {idea_title}
Problem: {idea_problem}
Method: {idea_method}
Experiment: {idea_experiment}

## Real-World Perspectives Found Online
Below are snippets from Reddit, HackerNews, Zhihu, Xiaohongshu, and other \
platforms. These represent how PRACTITIONERS think about related problems:

{wild_perspectives}

## Your Task
Extract 3 **non-academic insights** from these perspectives that could \
fundamentally change how we think about this research idea:

1. **Practitioner Pain Point**: What real-world problem do practitioners face \
that this idea ignores? What would make practitioners actually ADOPT (or REJECT) \
this approach? What failure modes do they see in production that papers never discuss?

2. **Unconventional Angle**: What approach do practitioners use that the academic \
community hasn't formalized? Is there a "folk wisdom" or engineering hack that \
points to a deeper theoretical insight? What would a systems engineer or startup \
CTO suggest differently?

3. **Reality Check**: What assumptions does this idea make that practitioners \
know are false in production? What "works in the lab but fails in deployment" \
pattern does this fall into?

For each insight, generate a **mutation** — a concrete research direction that \
would emerge from taking the practitioner perspective seriously.

Reply ONLY with a JSON object (no markdown):
{{
  "wild_insights": [
    {{
      "type": "pain_point" | "unconventional_angle" | "reality_check",
      "source_perspective": "1-2 sentences summarizing the real-world perspective",
      "academic_blind_spot": "What academics miss about this",
      "mutation_title": "A concrete research idea title",
      "mutation_method": "2-3 sentences describing the approach"
    }}
  ]
}}"""


_NOVELTY_DEEPEN_PROMPT = """\
You are a **Novelty Synthesizer** — your job is to COLLIDE challenge results \
from 3 different angles to forge a **genuinely novel idea (C)** that cannot be \
decomposed back into its ingredients.

## Research Context
Topic: "{topic_name}"

## Original Idea (the "A+B" starting point — you must TRANSCEND this)
Title: {idea_title}
Problem: {idea_problem}
Method: {idea_method}
Motivation: {idea_motivation}
Experiment: {idea_experiment}
Novelty Score: {idea_novelty}
{idea_extras}

## Challenge Results

### Assumption Flips
{assumption_flips}

### Analogical Leaps
{analogical_leaps}

### Contradictions Found
{contradictions}

### Wild Perspectives (from practitioners, forums, industry)
{wild_perspectives}

## Your Task — Collision Synthesis (NOT cherry-picking)

### Step 1: Find the TENSION
Look across ALL challenge results (including wild perspectives). Identify a \
**tension** — a place where two mutations CONTRADICT each other, or where an \
assumption flip CONFLICTS with an analogical leap, or where a practitioner \
insight UNDERMINES an academic assumption. The tension is the raw material for \
genuine novelty. Wild perspectives are especially valuable because they reveal \
what WORKS IN PRACTICE but has no theoretical explanation — formalizing that \
gap IS a novel contribution.

Example: If a flip says "remove the encoder" but an analogy suggests "use a \
richer representation," the tension is: how do you get rich representations \
WITHOUT an encoder? Resolving THAT is the novel insight.

### Step 2: Resolve the tension into C
The resolution of the tension IS your new idea C. C should:
- NOT be the original idea with a mutation bolted on (that's still A + mutation)
- NOT be any single mutation taken as-is (that's just picking, not synthesizing)
- BE something that only exists BECAUSE you confronted the tension

### Step 3: Verify C is irreducible
Apply the **decomposition test**: try to describe C as "apply X to Y" for any \
X and Y. If you can, C is not novel enough — go back to Step 1 with a different \
tension.

### ANTI-PATTERNS (your output will be rejected if it matches these)
- "We combine mutation_1 with mutation_2" ← this is A+B with extra steps
- "We take the analogical approach and apply it to the original problem" ← X+Y
- "Building on the assumption flip, we modify the original method to..." ← incremental
- The method section reads like "original method + one new component" ← bolting

{retry_context}

Reply ONLY with a JSON object (no markdown):
{{
  "title": "The novel idea C (not a modification of the original title)",
  "problem": "1-2 sentences — reframed through the lens of the tension",
  "motivation": "2-3 sentences explaining the tension and why resolving it matters",
  "method": "3-5 sentences with specific technical details. Must NOT read as \
'original method + tweak'. Should feel like a different approach to the same problem domain.",
  "experiment_plan": "2-4 sentences with datasets, baselines, metrics",
  "novelty_score": 8,
  "feasibility_score": 7,
  "novelty_reasoning": "3-5 sentences justifying the novelty score",
  "closest_prior_work": "The single most similar existing approach",
  "what_is_genuinely_new": "One sentence: the insight that emerges from the tension resolution",
  "tension_used": "1-2 sentences describing the specific tension between challenge results",
  "three_sentence_novelty_test": "3 sentences proving C cannot be decomposed into apply-X-to-Y",
  "challenge_sources_used": ["which challenge results were involved in the tension"],
  "deepening_note": "1-2 sentences on how C differs structurally (not just incrementally) from the original"
}}"""

_NOVELTY_VERIFY_PROMPT = """\
You are a **Novelty Auditor** — a ruthless but fair judge who determines whether \
a research idea is genuinely novel (C) or secretly just an A+B combination.

## Original Idea (before deepening)
Title: {original_title}
Method: {original_method}

## Deepened Idea (claims to be novel)
Title: {deepened_title}
Problem: {deepened_problem}
Method: {deepened_method}
What Is Genuinely New: {deepened_new}
Tension Used: {tension_used}
3-Sentence Novelty Test: {novelty_test}

## Your Task
Perform 3 tests:

### Test 1: Decomposition Test
Can this idea be restated as "apply X to Y" for specific X and Y? Try hard to \
find such a decomposition. If you find one, the idea FAILS.

### Test 2: Subtraction Test
Remove the "novel" component from the method. Does the remaining method still \
make sense as a coherent (if less interesting) approach? If yes, the novel part \
is just bolted on — FAILS.

### Test 3: Origin Test
Could someone arrive at this idea by simply reading the original idea + one of \
the challenge results? Or does it require the specific TENSION between multiple \
results to make sense? If a single challenge result suffices, it FAILS.

Reply ONLY with a JSON object (no markdown):
{{
  "passes_decomposition": true or false,
  "decomposition_attempt": "The best X+Y decomposition you found, or 'none found'",
  "passes_subtraction": true or false,
  "subtraction_analysis": "What happens when you remove the novel component",
  "passes_origin": true or false,
  "origin_analysis": "Could this come from a single challenge result alone?",
  "overall_verdict": "NOVEL" or "STILL_AB",
  "diagnosis": "2-3 sentences explaining the verdict",
  "improvement_hint": "If STILL_AB: one concrete sentence on what structural change \
would make it genuinely novel. If NOVEL: null"
}}"""


_IDEA_REFINE_PROMPT = """\
You are a senior researcher refining brainstorm ideas based on review feedback.

## Research Topic
"{topic_name}"

## Paper Library ({n_papers} papers)
{paper_summaries}

## Ideas with Review Feedback
{ideas_with_reviews}

## Your Task
For each idea marked REVISE, generate an improved version that addresses the \
weaknesses and follows the revision instructions. For ACCEPT ideas, return them \
unchanged. Omit DROP ideas entirely.

For each refined idea, include all original fields (title, problem, motivation, \
method, experiment_plan, novelty_score, feasibility_score) PLUS:
- "revision_note": Brief summary of what changed and why (1-2 sentences)

Preserve the core insight of each idea — improve execution, not replace the concept.

Reply ONLY with a JSON array of refined idea objects. No markdown."""

_NOVELTY_AWARE_REFINE_PROMPT = """\
You are a senior researcher refining brainstorm ideas with a **special focus on \
improving novelty**. The review identified novelty as the primary weakness — \
surface-level fixes to clarity or feasibility will NOT help.

## Research Topic
"{topic_name}"

## Paper Library ({n_papers} papers)
{paper_summaries}

## Ideas with Review Feedback
{ideas_with_reviews}

## Your Task
For each idea marked REVISE, generate an improved version. Your PRIMARY goal is \
to **boost novelty** — the other dimensions are secondary this round.

### Novelty-Specific Guidance
For each idea, the reviewer provided:
- A **novelty diagnosis** explaining whether the idea is genuine insight or X+Y
- A **novelty boost hint** suggesting how to break out of the X+Y pattern

Use these to guide your revision. Specifically:
1. Read the novelty_diagnosis carefully — if it says "X+Y combination", you MUST \
change the fundamental approach, not just reword it.
2. Follow the novelty_boost_hint as a starting point, but go FURTHER.
3. After revising, apply the **3-sentence novelty test**: can you explain in 3 \
sentences why this idea cannot be reduced to "apply X to Y"? If not, revise again.

For each refined idea, include all original fields PLUS:
- "revision_note": What changed and why, specifically addressing the novelty issue
- "novelty_reasoning": 3-5 sentences justifying the new novelty score
- "what_is_genuinely_new": One sentence capturing the core novel contribution

For ACCEPT ideas, return them unchanged. Omit DROP ideas entirely.

Reply ONLY with a JSON array of refined idea objects. No markdown."""


_TARGETED_REFINE_PROMPT = """\
You are a senior researcher performing **surgical revisions** of brainstorm ideas.
Each idea has a specific bottleneck dimension identified below. You MUST fix ONLY \
that bottleneck while keeping everything else intact.

CRITICAL RULES:
1. Do NOT rewrite the idea from scratch. Keep the same title, core method, and \
insight. Only modify the parts related to the bottleneck.
2. If the bottleneck is feasibility: make the method more concrete, add specific \
architectures/datasets/compute requirements. Do NOT change the core theoretical \
framework or simplify the novelty.
3. If the bottleneck is novelty: follow the novelty_boost_hint. But do NOT make \
the method more abstract or add unimplementable math. Keep the same feasibility level.
4. If the bottleneck is impact: strengthen the motivation and broaden applications. \
Do NOT change the method.
5. If the bottleneck is clarity: rewrite the method description more precisely. \
Do NOT change what the method does.

## Research Topic
"{topic_name}"

## Paper Library ({n_papers} papers)
{paper_summaries}

## Ideas with Targeted Revision Instructions
{ideas_with_reviews}

For each idea, include all original fields PLUS:
- "revision_note": 1 sentence on what you changed (must reference the bottleneck)

Reply ONLY with a JSON array of refined idea objects. No markdown."""


_IDEA_RESCUE_PROMPT = """\
You are a senior researcher performing a **deep revision** of research ideas that \
have failed to reach ACCEPT quality after {n_rounds} consecutive review rounds. \
Surface-level tweaks are NOT working — you need to fundamentally rethink the novelty.

## Research Topic
"{topic_name}"

## Paper Library ({n_papers} papers)
{paper_summaries}

## Additional Research Context
The following prior art searches and cross-topic insights may reveal differentiation \
angles that previous refinement rounds missed.

{extra_context}

## Reference: Ideas That Passed Review
These ideas were ACCEPTED by the same reviewer. Study what makes them strong — \
clear differentiation, concrete novelty, tight scope. Use them as calibration.

{accepted_ideas}

## Stubborn Ideas (REVISE × {n_rounds} rounds)
{ideas_with_history}

## Your Task
For each idea:

1. **Root cause analysis**: Why has this idea been REVISE for {n_rounds} rounds? \
Identify the STRUCTURAL weakness (usually: novelty is just X+Y combination, or the \
core assumption is already disproven, or the problem doesn't need solving).

2. **Check the prior art**: Use the additional context above to see if a different \
angle exists. Can you reposition against the closest prior work?

3. **Learn from accepted ideas**: What differentiation strategies worked for the \
accepted ideas? Can you apply a similar strategy?

4. **Deep revision OR drop**:
   - If a genuinely new angle exists: rewrite with a fundamentally different novelty \
claim. Change the technical approach if needed — preserve only the problem domain.
   - If no viable angle exists: return the idea with "verdict": "DROP" and explain why.

For each idea, include all fields (title, problem, motivation, method, \
experiment_plan, novelty_score, feasibility_score) PLUS:
- "revision_note": What fundamentally changed and why (2-3 sentences)
- "verdict": "REVISED" or "DROP"

Reply ONLY with a JSON array. No markdown."""


_IDEA_RESCUE_PIVOT_PROMPT = """\
You are a senior researcher making a FINAL attempt to salvage research ideas. \
These ideas have failed {n_rounds} review rounds AND an initial rescue attempt. \
Incremental fixes cannot save them — only a fundamental pivot can.

## Research Topic
"{topic_name}"

## Paper Library ({n_papers} papers)
{paper_summaries}

## Additional Research Context
{extra_context}

## Successfully Accepted Ideas (for cross-pollination)
These ideas passed review. You MUST cross-pollinate — borrow a technique, \
theoretical lens, or evaluation methodology from an accepted idea and apply it \
to the stubborn idea's problem domain in a non-obvious way.

{accepted_ideas}

## Ideas to Pivot (with reviewer feedback from rescue attempt)
{ideas_with_feedback}

## Your Task
For each idea, generate a **completely new idea** that:
1. Stays in the same **problem domain** (e.g., if the original is about video DiT \
inference speed, keep that domain)
2. Uses a **fundamentally different technical approach** — not a tweak, a pivot
3. Cross-pollinates with at least one accepted idea's technique or insight
4. Is concrete enough to implement (specific architecture, loss, algorithm)

Think of techniques from adjacent fields: control theory, information theory, \
spectral analysis, causal inference, game theory, online learning. The best ideas \
import a framework from another field and apply it to this problem.

For each idea, include all fields (title, problem, motivation, method, \
experiment_plan, novelty_score, feasibility_score) PLUS:
- "revision_note": What was pivoted and which accepted idea inspired the cross-pollination
- "cross_pollination_source": Title of the accepted idea that inspired this pivot

Reply ONLY with a JSON array. No markdown."""


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _build_paper_summaries(papers: list[dict], max_papers: int = 30) -> str:
    """Build compact paper summaries for brainstorm context."""
    lines = []
    for p in papers[:max_papers]:
        parts = [f"**{p.get('title', 'Untitled')}**"]
        if p.get("key_insight"):
            parts.append(f"  Insight: {p['key_insight']}")
        if p.get("method"):
            parts.append(f"  Method: {p['method']}")
        if p.get("contribution"):
            parts.append(f"  Contribution: {p['contribution']}")
        elif p.get("summary"):
            parts.append(f"  Summary: {p['summary']}")
        if p.get("math_concepts"):
            concepts = p["math_concepts"]
            if isinstance(concepts, str):
                try:
                    concepts = json.loads(concepts)
                except (json.JSONDecodeError, TypeError):
                    concepts = []
            if concepts:
                parts.append(f"  Math: {', '.join(concepts)}")
        lines.append("\n".join(parts))
    return "\n\n".join(lines) or "(No papers in library yet)"


def _format_ideas_for_prompt(ideas: list[dict]) -> str:
    """Format ideas for verification prompts, including novelty metadata."""
    lines = []
    for i, idea in enumerate(ideas, 1):
        lines.append(f"### Idea {i}: {idea.get('title', 'Untitled')}")
        lines.append(f"Problem: {idea.get('problem', '')}")
        lines.append(f"Method: {idea.get('method', '')}")
        lines.append(f"Experiment: {idea.get('experiment_plan', '')}")
        # Enhanced novelty fields
        if idea.get("novelty_reasoning"):
            lines.append(f"Novelty Reasoning: {idea['novelty_reasoning']}")
        if idea.get("closest_prior_work"):
            lines.append(f"Closest Prior Work: {idea['closest_prior_work']}")
        if idea.get("what_is_genuinely_new"):
            lines.append(f"Genuinely New: {idea['what_is_genuinely_new']}")
        # Prescreen results
        ps = idea.get("_prescreen", {})
        if ps:
            level = ps.get("overlap_level", "")
            if level:
                lines.append(f"arXiv Prescreen: {level} (max_overlap={ps.get('max_overlap', 0)})")
                for m in ps.get("top_matches", [])[:3]:
                    lines.append(f"  - [{m.get('overlap_score', 0):.2f}] {m.get('title', '?')}")
        lines.append("")
    return "\n".join(lines)


def _load_discovery_context(registry) -> str:
    """Load latest trending + math discovery reports as brainstorm context."""
    if registry is None:
        return ""
    parts = []
    try:
        trending = registry.get_latest_discovery_report("trending")
        if trending and trending.get("content"):
            parts.append(
                "## Recent Trending Themes (from cross-topic discovery)\n"
                + trending["content"][:3000]
            )
    except Exception:
        pass
    try:
        math = registry.get_latest_discovery_report("math")
        if math and math.get("content"):
            parts.append(
                "## Recent Math Insights (from cross-topic discovery)\n"
                + math["content"][:3000]
            )
    except Exception:
        pass
    if parts:
        return (
            "## Discovery Context\n"
            "Below are recent cross-topic discovery insights. Use these to inform "
            "your brainstorming — look for connections between trending themes, "
            "mathematical techniques, and this topic's paper library.\n\n"
            + "\n\n".join(parts)
        )
    return ""


# ---------------------------------------------------------------------------
# Enhanced Context Loaders (Phase 1)
# ---------------------------------------------------------------------------

def _load_topic_insights(data_dir: str, topic_id: str, registry) -> str:
    """Load the latest insights.md for this topic. Returns formatted string."""
    if registry is None:
        return ""
    try:
        latest = registry.get_latest_session(topic_id)
        if not latest:
            return ""
        insights_path = latest.get("insights_path", "")
        if not insights_path:
            return ""
        from pathlib import Path
        p = Path(insights_path)
        if not p.exists():
            # Try relative to data_dir
            p = Path(data_dir) / topic_id / "sessions" / latest["id"] / "insights.md"
        if not p.exists():
            return ""
        content = p.read_text(encoding="utf-8")
        if not content.strip():
            return ""
        # Truncate to keep prompt manageable
        return (
            "## Topic Insights (from latest paper analysis)\n"
            "These insights were synthesized from your paper library. Pay special "
            "attention to Research Gaps & Opportunities — these are prime idea seeds.\n\n"
            + content[:4000]
        )
    except Exception as e:
        log.debug("Failed to load topic insights: %s", e)
        return ""


def _load_session_reports(data_dir: str, topic_id: str, registry) -> str:
    """Load the latest session report (Executive Summary + Thematic Analysis)."""
    if registry is None:
        return ""
    try:
        latest = registry.get_latest_session(topic_id)
        if not latest:
            return ""
        report_path = latest.get("report_path", "")
        if not report_path:
            return ""
        from pathlib import Path
        p = Path(report_path)
        if not p.exists():
            p = Path(data_dir) / topic_id / "sessions" / latest["id"] / "report.md"
        if not p.exists():
            return ""
        content = p.read_text(encoding="utf-8")
        if not content.strip():
            return ""
        # Extract Executive Summary + Thematic Analysis (skip Paper Details)
        parts = []
        for section in ("## Executive Summary", "## Thematic Analysis"):
            idx = content.find(section)
            if idx >= 0:
                # Find end: next ## or end of content
                end = content.find("\n## ", idx + len(section))
                if end < 0:
                    end = len(content)
                parts.append(content[idx:end].strip())
        if not parts:
            return ""
        return (
            "## Session Report Context\n"
            "Executive summary and thematic analysis from the latest research session.\n\n"
            + "\n\n".join(parts)[:3000]
        )
    except Exception as e:
        log.debug("Failed to load session reports: %s", e)
        return ""


def _load_github_repos(data_dir: str, topic_id: str) -> str:
    """Load GitHub repos from storage for feasibility/tooling signals."""
    try:
        store = Storage(data_dir, topic_id)
        try:
            repos, total = store.get_all_github(limit=30, offset=0)
        finally:
            store.close()
        if not repos:
            return ""
        lines = ["## GitHub Ecosystem Context",
                 "Related repositories — use for feasibility assessment, tooling gaps, "
                 "and engineering pain points.\n"]
        for r in repos[:20]:
            name = r.get("repo_full_name", "?")
            stars = r.get("stars", 0)
            desc = r.get("description", "") or ""
            summary = r.get("summary", "") or ""
            lines.append(f"- **{name}** ({stars} stars): {desc[:100]}")
            if summary:
                lines.append(f"  Implementation: {summary[:150]}")
        return "\n".join(lines)
    except Exception as e:
        log.debug("Failed to load GitHub repos: %s", e)
        return ""


def _load_brainstorm_history(topic_id: str, registry) -> str:
    """Load past brainstorm ideas + review verdicts (researcher memory).

    This prevents re-generating the same ideas and enables building on past failures.
    """
    if registry is None:
        return ""
    try:
        sessions = registry.list_brainstorm_sessions(topic_id)
        if not sessions:
            return ""
        # Only completed sessions
        completed = [s for s in sessions if s.get("status") == "completed"]
        if not completed:
            return ""
        lines = ["## Brainstorm History (Researcher Memory)",
                 "Past ideas generated for this topic. AVOID regenerating similar ideas. "
                 "Instead, build on failures, recombine successful ideas, or explore "
                 "entirely new directions.\n"]
        idea_count = 0
        for sess in completed[:5]:  # Last 5 sessions
            ideas = sess.get("ideas_json") or []
            if not ideas:
                continue
            date = (sess.get("started_at") or "?")[:10]
            mode = sess.get("mode", "?")
            lines.append(f"### Session {date} ({mode} mode, {len(ideas)} ideas)")
            for idea in ideas[:6]:
                title = idea.get("title", "Untitled")
                status = idea.get("status", "active")
                review = idea.get("review", {})
                novelty = review.get("novelty", "?")
                overall = review.get("overall", "?")
                verdict = review.get("verdict", "?")
                weaknesses = review.get("weaknesses", [])
                lines.append(f"- **{title}** [{status}] — novelty={novelty}, overall={overall}, verdict={verdict}")
                if status == "dropped" and weaknesses:
                    lines.append(f"  Drop reasons: {'; '.join(weaknesses[:2])}")
                elif weaknesses:
                    lines.append(f"  Weaknesses: {'; '.join(weaknesses[:2])}")
                idea_count += 1
        if idea_count == 0:
            return ""
        lines.append(f"\n*{idea_count} past ideas total. Generate NEW directions, not variations of these.*")
        return "\n".join(lines)
    except Exception as e:
        log.debug("Failed to load brainstorm history: %s", e)
        return ""


def _load_research_plans(topic_id: str, registry) -> str:
    """Load existing research plans to know what directions are already being explored."""
    if registry is None:
        return ""
    try:
        plans_data = registry.list_research_plans(topic_id)
        plans = plans_data.get("plans", []) if isinstance(plans_data, dict) else plans_data
        if not plans:
            return ""
        completed = [p for p in plans if p.get("status") == "completed"]
        if not completed:
            return ""
        lines = ["## Existing Research Plans",
                 "These ideas have already been developed into full research plans. "
                 "Consider complementary or orthogonal directions.\n"]
        for plan in completed[:5]:
            title = plan.get("idea_title", "Untitled")
            review = plan.get("review", "")
            # Extract key review points
            review_snippet = review[:200] + "..." if len(review) > 200 else review
            lines.append(f"- **{title}** (completed)")
            if review_snippet:
                lines.append(f"  Review: {review_snippet}")
        return "\n".join(lines)
    except Exception as e:
        log.debug("Failed to load research plans: %s", e)
        return ""


def _build_citation_weighted_summaries(papers: list[dict], max_papers: int = 30) -> str:
    """Build paper summaries weighted by citation count + quality score.

    High-citation and high-quality papers get more context;
    low-quality papers are de-emphasized.
    """
    if not papers:
        return "(No papers in library yet)"
    # Score each paper: citation_count * 0.3 + quality_score * 0.7 (normalized)
    scored = []
    for p in papers:
        cite = p.get("citation_count", 0) or 0
        quality = p.get("quality_score", 3) or 3
        # Composite score: quality dominates, citations boost
        score = quality * 10 + min(cite, 100) * 0.5
        scored.append((score, p))
    scored.sort(key=lambda x: x[0], reverse=True)

    lines = []
    for rank, (score, p) in enumerate(scored[:max_papers], 1):
        cite = p.get("citation_count", 0) or 0
        quality = p.get("quality_score", 0) or 0
        venue = p.get("venue", "") or ""
        parts = [f"**{p.get('title', 'Untitled')}**"]
        # Add citation/venue info for high-impact papers
        meta = []
        if cite > 0:
            meta.append(f"citations={cite}")
        if quality > 0:
            meta.append(f"quality={quality}/5")
        if venue:
            meta.append(venue)
        if meta:
            parts.append(f"  [{', '.join(meta)}]")
        if p.get("key_insight"):
            parts.append(f"  Insight: {p['key_insight']}")
        if p.get("method"):
            parts.append(f"  Method: {p['method']}")
        if p.get("contribution"):
            parts.append(f"  Contribution: {p['contribution']}")
        elif p.get("summary"):
            parts.append(f"  Summary: {p['summary']}")
        if p.get("math_concepts"):
            concepts = p["math_concepts"]
            if isinstance(concepts, str):
                try:
                    concepts = json.loads(concepts)
                except (json.JSONDecodeError, TypeError):
                    concepts = []
            if concepts:
                parts.append(f"  Math: {', '.join(concepts)}")
        lines.append("\n".join(parts))
    return "\n\n".join(lines)


_QUESTION_GENERATION_PROMPT = """\
You are a senior research strategist generating sharp, unanswered research questions.

## Research Topic
"{topic_name}"

## Paper Library ({n_papers} papers)
{paper_summaries}

{insights_context}

## Your Task
Generate 5-8 **specific, answerable research questions** that are NOT yet addressed by \
the papers above. These questions should:

1. Be precise enough to drive a single paper (not a PhD thesis)
2. Be grounded in gaps or tensions visible in the literature
3. Span different types: empirical questions, theoretical questions, methodological \
questions, evaluation questions
4. Include at least one "contrarian" question that challenges a common assumption

For each question, provide:
- "question": The research question (1-2 sentences, ending with ?)
- "type": EMPIRICAL | THEORETICAL | METHODOLOGICAL | EVALUATION | CONTRARIAN
- "grounding": Which papers or gaps motivate this question (1 sentence)
- "potential_impact": HIGH | MEDIUM | LOW
- "why_unanswered": Why hasn't this been answered yet? (1 sentence)

Reply ONLY with a JSON array. No markdown fences."""


def _questions_agree(q1: dict, q2: dict, threshold: float = 0.35) -> bool:
    """Check if two research questions are similar enough (word overlap on question text)."""
    words1 = set(q1.get("question", "").lower().split())
    words2 = set(q2.get("question", "").lower().split())
    if not words1 or not words2:
        return False
    overlap = len(words1 & words2) / len(words1 | words2)
    return overlap >= threshold


def _generate_research_questions(
    topic_name: str,
    paper_summaries: str,
    insights_text: str,
    cfg: dict,
) -> str:
    """Generate research questions using Claude Opus + Codex in parallel.

    Only questions where both models agree (produce similar questions) are kept.
    """
    n_papers = paper_summaries.count("**")  # rough count
    prompt = _QUESTION_GENERATION_PROMPT.format(
        topic_name=topic_name,
        n_papers=n_papers,
        paper_summaries=paper_summaries[:6000],
        insights_context=insights_text[:2000] if insights_text else "",
    )

    claude_result: list = []
    codex_result: list = []

    def _run_claude():
        nonlocal claude_result
        raw = call_cli(prompt, cfg, model="opus", timeout=120)
        if raw:
            parsed = _parse_json_with_repair(raw, cfg, label="questions_claude")
            if isinstance(parsed, list):
                claude_result = parsed

    def _run_codex():
        nonlocal codex_result
        raw = call_codex(prompt, cfg, timeout=120)
        if raw:
            parsed = _parse_json_with_repair(raw, cfg, label="questions_codex")
            if isinstance(parsed, list):
                codex_result = parsed

    # Run both in parallel
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_run_claude), pool.submit(_run_codex)]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                log.warning("Question generation thread failed: %s", e)

    log.info("Research questions: Claude produced %d, Codex produced %d",
             len(claude_result), len(codex_result))

    # Agreement: keep Claude questions that have a matching Codex question
    if claude_result and codex_result:
        agreed: list[dict] = []
        for cq in claude_result:
            for xq in codex_result:
                if _questions_agree(cq, xq):
                    # Merge: use Claude's text but note agreement
                    cq["_agreed"] = True
                    agreed.append(cq)
                    break
        log.info("Research questions agreement: %d/%d Claude questions confirmed by Codex",
                 len(agreed), len(claude_result))
        final = agreed if agreed else claude_result[:4]  # fallback to top Claude if no agreement
    elif claude_result:
        log.info("Research questions: Codex returned nothing, using Claude results only")
        final = claude_result
    elif codex_result:
        log.info("Research questions: Claude returned nothing, using Codex results only")
        final = codex_result
    else:
        return ""

    if not final:
        return ""

    # Format output
    n_agreed = sum(1 for q in final if q.get("_agreed"))
    agreement_note = f" ({n_agreed}/{len(final)} confirmed by dual-model agreement)" if n_agreed else ""
    lines = [f"## Research Questions (Question-First, Opus+Codex){agreement_note}",
             "These unanswered questions were identified from gaps in the literature. "
             "Use them as seeds — each question could become a research idea.\n"]
    for q in final[:8]:
        question = q.get("question", "?")
        qtype = q.get("type", "?")
        impact = q.get("potential_impact", "?")
        grounding = q.get("grounding", "")
        agreed_marker = " [agreed]" if q.get("_agreed") else ""
        lines.append(f"- [{qtype}, {impact}]{agreed_marker} **{question}**")
        if grounding:
            lines.append(f"  Grounding: {grounding}")
    return "\n".join(lines)


_NOVELTY_MAP_PROMPT = """\
You are a novelty analysis specialist performing multi-axis novelty assessment.

## Ideas to Analyze
{ideas_text}

## Past Ideas (from previous brainstorm sessions)
{history_ideas}

## Known Research Gaps
{insights_gaps}

## Your Task
For EACH idea, assess novelty on 6 independent axes:

1. **problem_novelty**: Is the PROBLEM itself new? (1-10)
   1 = well-studied problem, 10 = problem nobody has articulated
2. **method_novelty**: Is the METHOD/approach new? (1-10)
   1 = standard technique, 10 = entirely new algorithm/framework
3. **data_novelty**: Is the data/supervision/setting new? (1-10)
   1 = standard benchmarks, 10 = novel data source or training paradigm
4. **eval_novelty**: Is the evaluation approach new? (1-10)
   1 = standard metrics, 10 = new way to measure success
5. **insight_novelty**: Is there a non-obvious intellectual insight? (1-10)
   1 = straightforward combination, 10 = paradigm-shifting insight
6. **domain_novelty**: Is this a novel application domain or cross-domain transfer? (1-10)
   1 = standard domain, 10 = entirely new application area

Also provide:
- "nearest_past_idea": The most similar past idea (title) or "none"
- "differentiation": How this differs from the nearest past idea (1 sentence)
- "strongest_axis": Which axis is most novel
- "weakest_axis": Which axis needs improvement
- "salvage_path": If weakest < 4, suggest how to boost it (1 sentence)

Reply ONLY with a JSON array. No markdown."""


_NOVELTY_AXES = ("problem_novelty", "method_novelty", "data_novelty",
                  "eval_novelty", "insight_novelty", "domain_novelty")


def _merge_novelty_assessments(claude_item: dict, codex_item: dict) -> dict:
    """Merge two novelty assessments by averaging scores and flagging disagreements.

    Only updates scores when both models agree within 3 points.
    Large disagreements are flagged for transparency.
    """
    merged = dict(claude_item)  # start from Claude's richer output
    disagreements: list[str] = []

    for axis in _NOVELTY_AXES:
        c_score = claude_item.get(axis)
        x_score = codex_item.get(axis)
        if c_score is not None and x_score is not None:
            try:
                c_val = int(c_score)
                x_val = int(x_score)
                diff = abs(c_val - x_val)
                if diff <= 3:
                    # Agreement: average
                    merged[axis] = round((c_val + x_val) / 2)
                else:
                    # Disagreement: take conservative (lower) score, flag it
                    merged[axis] = min(c_val, x_val)
                    disagreements.append(
                        f"{axis}: Claude={c_val} vs Codex={x_val} (using {merged[axis]})"
                    )
            except (TypeError, ValueError):
                pass

    merged["_dual_model"] = True
    if disagreements:
        merged["_disagreements"] = disagreements
    # Prefer Codex's salvage_path if Claude didn't provide one
    if not merged.get("salvage_path") and codex_item.get("salvage_path"):
        merged["salvage_path"] = codex_item["salvage_path"]
    return merged


def _build_novelty_map(
    ideas: list[dict],
    history_ideas: str,
    insights_gaps: str,
    cfg: dict,
) -> list[dict]:
    """Multi-axis novelty decomposition using Claude Opus + Codex in parallel.

    Both models score each idea on 6 axes. Results are merged:
    - Scores within 3 points: averaged (agreement)
    - Scores >3 apart: conservative (lower) score used, flagged as disagreement

    Returns list of novelty assessment dicts, one per idea.
    """
    ideas_text = _format_ideas_for_prompt(ideas)
    prompt = _NOVELTY_MAP_PROMPT.format(
        ideas_text=ideas_text,
        history_ideas=history_ideas[:3000] if history_ideas else "(No past ideas)",
        insights_gaps=insights_gaps[:2000] if insights_gaps else "(No known gaps)",
    )

    claude_result: list = []
    codex_result: list = []

    def _run_claude():
        nonlocal claude_result
        raw = call_cli(prompt, cfg, model="opus", timeout=120)
        if raw:
            parsed = _parse_json_with_repair(raw, cfg, label="novelty_map_claude")
            if isinstance(parsed, list):
                claude_result = parsed

    def _run_codex():
        nonlocal codex_result
        raw = call_codex(prompt, cfg, timeout=120)
        if raw:
            parsed = _parse_json_with_repair(raw, cfg, label="novelty_map_codex")
            if isinstance(parsed, list):
                codex_result = parsed

    # Run both in parallel
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_run_claude), pool.submit(_run_codex)]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                log.warning("Novelty map thread failed: %s", e)

    log.info("Novelty map: Claude produced %d, Codex produced %d assessments",
             len(claude_result), len(codex_result))

    # Merge: pair up by index (both should have one entry per idea)
    if claude_result and codex_result:
        merged: list[dict] = []
        for i in range(len(ideas)):
            c_item = claude_result[i] if i < len(claude_result) else {}
            x_item = codex_result[i] if i < len(codex_result) else {}
            if c_item and x_item:
                merged.append(_merge_novelty_assessments(c_item, x_item))
            elif c_item:
                merged.append(c_item)
            elif x_item:
                merged.append(x_item)
            else:
                merged.append({})
        # Log agreement stats
        n_dual = sum(1 for m in merged if m.get("_dual_model"))
        n_disagree = sum(1 for m in merged if m.get("_disagreements"))
        log.info("Novelty map merged: %d dual-model, %d with disagreements", n_dual, n_disagree)
        return merged
    elif claude_result:
        log.info("Novelty map: Codex returned nothing, using Claude results only")
        return claude_result
    elif codex_result:
        log.info("Novelty map: Claude returned nothing, using Codex results only")
        return codex_result
    return []


def _clamp_score(val, lo: int = 1, hi: int = 10) -> int:
    """Clamp a score value to [lo, hi] range."""
    try:
        return max(lo, min(hi, int(val)))
    except (TypeError, ValueError):
        return lo


def _identify_bottleneck(review: dict) -> tuple[str, str]:
    """Identify the primary bottleneck dimension and build a targeted instruction.

    Returns (bottleneck_name, instruction_text).
    """
    n = review.get("novelty", 5)
    i = review.get("impact", 5)
    f = review.get("feasibility", 5)
    c = review.get("clarity", 5)

    dims = {"novelty": n, "impact": i, "feasibility": f, "clarity": c}
    bottleneck = min(dims, key=dims.get)
    score = dims[bottleneck]

    if bottleneck == "feasibility":
        hint = (f"BOTTLENECK: feasibility ({score}/10). "
                "Make the method more concrete: specify architectures, datasets, "
                "compute budget, and implementation steps. "
                "Do NOT simplify the core idea or reduce novelty.")
    elif bottleneck == "novelty":
        boost = review.get("novelty_boost_hint", "")
        diag = review.get("novelty_diagnosis", "")
        hint = (f"BOTTLENECK: novelty ({score}/10). "
                f"Diagnosis: {diag[:200]} "
                f"Hint: {boost[:200]} "
                "Do NOT make feasibility worse — keep the method implementable.")
    elif bottleneck == "impact":
        hint = (f"BOTTLENECK: impact ({score}/10). "
                "Strengthen the motivation: who benefits, what problem size, "
                "what downstream applications. Do NOT change the method.")
    else:  # clarity
        hint = (f"BOTTLENECK: clarity ({score}/10). "
                "Rewrite the method description more precisely. "
                "Add concrete notation, pseudocode, or algorithmic steps. "
                "Do NOT change what the method does.")

    return bottleneck, hint


def _format_ideas_with_reviews(
    ideas: list[dict], reviews: list[dict], *, targeted: bool = False,
) -> str:
    """Format ideas alongside their review feedback for the refine prompt.

    If targeted=True, adds per-idea bottleneck analysis and explicit
    instructions to fix ONLY the weakest dimension.
    """
    lines = []
    for i, idea in enumerate(ideas):
        review = reviews[i] if i < len(reviews) else {}
        verdict = review.get("verdict", "ACCEPT")
        lines.append(f"### Idea {i+1}: {idea.get('title', 'Untitled')} [{verdict}]")
        lines.append(f"Problem: {idea.get('problem', '')}")
        lines.append(f"Method: {idea.get('method', '')}")
        lines.append(f"Experiment: {idea.get('experiment_plan', '')}")
        if review:
            lines.append(f"\n**Review Scores**: novelty={review.get('novelty', '?')}, "
                         f"feasibility={review.get('feasibility', '?')}, "
                         f"clarity={review.get('clarity', '?')}, "
                         f"impact={review.get('impact', '?')}, "
                         f"overall={review.get('overall', '?')}")

            if targeted and verdict == "REVISE":
                bottleneck, instruction = _identify_bottleneck(review)
                lines.append(f"\n**>>> TARGETED FIX: {instruction}**")
                # Show which dimensions are GOOD and must be preserved
                n = review.get("novelty", 5)
                imp = review.get("impact", 5)
                f = review.get("feasibility", 5)
                c = review.get("clarity", 5)
                preserve = [
                    f"{dim}={s}" for dim, s in
                    [("novelty", n), ("impact", imp), ("feasibility", f), ("clarity", c)]
                    if dim != bottleneck and s >= 6
                ]
                if preserve:
                    lines.append(f"**PRESERVE (do not lower)**: {', '.join(preserve)}")
            else:
                weaknesses = review.get("weaknesses", [])
                if weaknesses:
                    lines.append("**Weaknesses**:")
                    for w in weaknesses:
                        lines.append(f"  - {w}")
                instructions = review.get("revision_instructions", [])
                if instructions:
                    lines.append("**Revision Instructions**:")
                    for inst in instructions:
                        lines.append(f"  - {inst}")
                # Novelty-specific fields from split reviewer
                diag = review.get("novelty_diagnosis", "")
                if diag:
                    lines.append(f"**Novelty Diagnosis**: {diag}")
                hint = review.get("novelty_boost_hint")
                if hint:
                    lines.append(f"**Novelty Boost Hint**: {hint}")
        lines.append("")
    return "\n".join(lines)


def _merge_reviews(reviews_a: list[dict], reviews_b: list[dict]) -> list[dict]:
    """Merge Reviewer A (feasibility/clarity) + Reviewer B (novelty/impact) results.

    Matches by idea_title, computes weighted overall, assigns verdict.
    Weights: novelty 0.35, impact 0.25, feasibility 0.25, clarity 0.15.
    """
    # Index Reviewer B by title for matching
    b_by_title: dict[str, dict] = {}
    for rb in reviews_b:
        title = rb.get("idea_title", "")
        if title:
            b_by_title[title] = rb

    merged: list[dict] = []
    for i, ra in enumerate(reviews_a):
        title = ra.get("idea_title", "")
        rb = b_by_title.get(title) or (reviews_b[i] if i < len(reviews_b) else {})

        feasibility = _clamp_score(ra.get("feasibility", 5))
        clarity = _clamp_score(ra.get("clarity", 5))
        novelty = _clamp_score(rb.get("novelty", 5))
        impact = _clamp_score(rb.get("impact", 5))

        overall = round(
            novelty * 0.35 + impact * 0.25 + feasibility * 0.25 + clarity * 0.15,
            1,
        )

        # Collect weaknesses from both reviewers
        weaknesses = (
            ra.get("f_weaknesses", [])
            + ra.get("c_weaknesses", [])
            + rb.get("n_weaknesses", [])
            + rb.get("i_weaknesses", [])
        )

        # Verdict logic
        scores = [novelty, impact, feasibility, clarity]
        if overall >= 7.0 and all(s >= 5 for s in scores):
            verdict = "ACCEPT"
        elif overall < 5.0 or any(s <= 2 for s in scores):
            verdict = "DROP"
        elif novelty >= 8:
            # High novelty is the hardest to achieve — graduate now,
            # feasibility can be improved when writing the research plan
            verdict = "CONDITIONAL_ACCEPT"
        else:
            verdict = "REVISE"

        # Build revision instructions from weaknesses
        revision_instructions = []
        if verdict == "REVISE":
            if novelty < 6:
                revision_instructions.append(
                    f"Novelty is weak ({novelty}/10). "
                    + (rb.get("novelty_boost_hint") or "Find a non-obvious insight beyond X+Y combination.")
                )
            if feasibility < 6:
                revision_instructions.append(
                    f"Feasibility needs work ({feasibility}/10). Make the implementation plan more concrete."
                )
            if clarity < 6:
                revision_instructions.append(
                    f"Clarity is low ({clarity}/10). Specify the method, datasets, and baselines precisely."
                )
            if impact < 6:
                revision_instructions.append(
                    f"Impact is limited ({impact}/10). Explain why this problem matters broadly."
                )

        merged.append({
            "idea_title": title,
            "novelty": novelty,
            "feasibility": feasibility,
            "clarity": clarity,
            "impact": impact,
            "overall": overall,
            "weaknesses": weaknesses,
            "strengths": [],  # split reviewers don't produce strengths
            "verdict": verdict,
            "revision_instructions": revision_instructions,
            "novelty_diagnosis": rb.get("novelty_diagnosis", ""),
            "novelty_boost_hint": rb.get("novelty_boost_hint"),
        })

    return merged


def _review_ideas_split(
    ideas: list[dict],
    topic_name: str,
    paper_summaries: str,
    n_papers: int,
    cfg: dict,
) -> list[dict]:
    """Review ideas via 2 parallel reviewers: A (feasibility/clarity) + B (novelty/impact).

    Reviewer A: Codex (fallback to Claude opus)
    Reviewer B: Claude opus

    Returns merged list of review dicts with all 4 dimensions + novelty_diagnosis.
    """
    ideas_text = _format_ideas_for_prompt(ideas)

    prompt_a = _REVIEW_FEASIBILITY_PROMPT.format(
        topic_name=topic_name,
        n_papers=n_papers,
        paper_summaries=paper_summaries,
        ideas_text=ideas_text,
    )
    prompt_b = _REVIEW_NOVELTY_PROMPT.format(
        topic_name=topic_name,
        n_papers=n_papers,
        paper_summaries=paper_summaries,
        ideas_text=ideas_text,
    )

    reviews_a: list[dict] = []
    reviews_b: list[dict] = []

    def _run_reviewer_a() -> list[dict]:
        """Reviewer A: Codex → Claude opus fallback."""
        raw = call_codex(prompt_a, cfg, timeout=600)
        if not raw:
            log.info("Reviewer A: Codex failed, falling back to Claude opus")
            raw = call_cli(prompt_a, cfg, model="opus", timeout=600)
        if not raw:
            log.warning("Reviewer A: all backends failed")
            return []
        parsed = _parse_json_with_repair(raw, cfg, label="review_feasibility")
        if not isinstance(parsed, list):
            log.warning("Reviewer A: returned non-list JSON")
            return []
        # Clamp scores
        for r in parsed:
            for dim in ("feasibility", "clarity"):
                if dim in r:
                    r[dim] = _clamp_score(r[dim])
        return parsed

    def _run_reviewer_b() -> list[dict]:
        """Reviewer B: Claude opus."""
        raw = call_cli(prompt_b, cfg, model="opus", timeout=600)
        if not raw:
            log.warning("Reviewer B: Claude opus failed")
            return []
        parsed = _parse_json_with_repair(raw, cfg, label="review_novelty")
        if not isinstance(parsed, list):
            log.warning("Reviewer B: returned non-list JSON")
            return []
        # Clamp scores
        for r in parsed:
            for dim in ("novelty", "impact"):
                if dim in r:
                    r[dim] = _clamp_score(r[dim])
        return parsed

    with ThreadPoolExecutor(max_workers=2) as executor:
        fa = executor.submit(_run_reviewer_a)
        fb = executor.submit(_run_reviewer_b)
        try:
            reviews_a = fa.result()
            log.info("Reviewer A (feasibility/clarity): %d reviews", len(reviews_a))
        except Exception as exc:
            log.warning("Reviewer A failed: %s", exc)
        try:
            reviews_b = fb.result()
            log.info("Reviewer B (novelty/impact): %d reviews", len(reviews_b))
        except Exception as exc:
            log.warning("Reviewer B failed: %s", exc)

    # If one reviewer failed completely, fall back to single-reviewer mode
    if not reviews_a and not reviews_b:
        log.warning("Both reviewers failed, falling back to single-reviewer mode")
        return _review_ideas_single(ideas, topic_name, paper_summaries, n_papers, cfg)
    if not reviews_a:
        # Synthesize feasibility/clarity defaults
        reviews_a = [{"idea_title": rb.get("idea_title", ""), "feasibility": 5, "clarity": 5,
                       "f_weaknesses": [], "c_weaknesses": []} for rb in reviews_b]
    if not reviews_b:
        # Synthesize novelty/impact defaults
        reviews_b = [{"idea_title": ra.get("idea_title", ""), "novelty": 5, "impact": 5,
                       "novelty_diagnosis": "", "novelty_boost_hint": None,
                       "n_weaknesses": [], "i_weaknesses": []} for ra in reviews_a]

    return _merge_reviews(reviews_a, reviews_b)


def _review_ideas_single(
    ideas: list[dict],
    topic_name: str,
    paper_summaries: str,
    n_papers: int,
    cfg: dict,
) -> list[dict]:
    """Fallback: review ideas via the original single LLM call."""
    ideas_text = _format_ideas_for_prompt(ideas)
    prompt = _IDEA_REVIEW_PROMPT.format(
        topic_name=topic_name,
        n_papers=n_papers,
        paper_summaries=paper_summaries,
        ideas_text=ideas_text,
    )
    raw = call_cli(prompt, cfg)
    if not raw:
        log.warning("Idea review LLM call failed")
        return []

    parsed = _parse_json_with_repair(raw, cfg, label="idea_review")
    if not isinstance(parsed, list):
        log.warning("Idea review returned non-list JSON")
        return []

    # Clamp scores
    for r in parsed:
        for dim in ("novelty", "feasibility", "clarity", "impact"):
            if dim in r:
                r[dim] = _clamp_score(r[dim])
        if "overall" in r:
            try:
                r["overall"] = round(max(1.0, min(10.0, float(r["overall"]))), 1)
            except (TypeError, ValueError):
                r["overall"] = 5.0

    return parsed


def _review_ideas(
    ideas: list[dict],
    topic_name: str,
    paper_summaries: str,
    n_papers: int,
    cfg: dict,
) -> list[dict]:
    """Review ideas via parallel split reviewers. Returns list of review dicts.

    Uses 2 parallel reviewers (feasibility/clarity + novelty/impact) for better
    calibration. Falls back to single-reviewer mode if both fail.
    """
    return _review_ideas_split(ideas, topic_name, paper_summaries, n_papers, cfg)


def _refine_ideas(
    ideas: list[dict],
    reviews: list[dict],
    topic_name: str,
    paper_summaries: str,
    n_papers: int,
    cfg: dict,
    *,
    round_num: int = 1,
) -> list[dict]:
    """Refine ideas based on review feedback. 0-1 LLM calls.

    Returns refined idea list (ACCEPT unchanged, REVISE improved, DROP removed).
    On round >= 2, uses targeted refine (fix bottleneck only, preserve strengths).
    """
    # Check if any need revision
    needs_revision = any(
        r.get("verdict", "ACCEPT") == "REVISE"
        for r in reviews
    )
    if not needs_revision:
        log.info("All ideas ACCEPT — skipping refine LLM call")
        # Still filter out DROP ideas
        result = []
        for i, idea in enumerate(ideas):
            verdict = reviews[i].get("verdict", "ACCEPT") if i < len(reviews) else "ACCEPT"
            if verdict != "DROP":
                idea["review"] = reviews[i] if i < len(reviews) else {}
                result.append(idea)
        return result

    # Round 2+: use targeted refine (fix bottleneck only, preserve strengths)
    use_targeted = round_num >= 2
    if use_targeted:
        ideas_with_reviews = _format_ideas_with_reviews(
            ideas, reviews, targeted=True,
        )
        log.info("Using targeted refine prompt (round %d, fix bottleneck only)",
                 round_num)
        refine_prompt_template = _TARGETED_REFINE_PROMPT
    else:
        ideas_with_reviews = _format_ideas_with_reviews(ideas, reviews)
        # Decide which refine prompt to use: novelty-aware if any REVISE idea has
        # low novelty + a boost hint from the split reviewer
        use_novelty_aware = False
        for r in reviews:
            if (r.get("verdict") == "REVISE"
                    and r.get("novelty_boost_hint")
                    and r.get("novelty", 10) < 6):
                use_novelty_aware = True
                break

        if use_novelty_aware:
            log.info("Using novelty-aware refine prompt (low-novelty REVISE ideas detected)")
            refine_prompt_template = _NOVELTY_AWARE_REFINE_PROMPT
        else:
            refine_prompt_template = _IDEA_REFINE_PROMPT

    prompt = refine_prompt_template.format(
        topic_name=topic_name,
        n_papers=n_papers,
        paper_summaries=paper_summaries,
        ideas_with_reviews=ideas_with_reviews,
    )
    raw = call_cli(prompt, cfg)
    if not raw:
        log.warning("Idea refine LLM call failed, keeping originals")
        return ideas

    parsed = _parse_json_with_repair(raw, cfg, label="idea_refine")
    if not isinstance(parsed, list):
        log.warning("Idea refine returned non-list, keeping originals")
        return ideas

    # Attach review info to refined ideas
    for i, refined_idea in enumerate(parsed):
        # Try to match by title to find original review
        rtitle = refined_idea.get("title", "")
        matched_review = None
        for j, review in enumerate(reviews):
            if review.get("idea_title", "") == rtitle or (j < len(ideas) and ideas[j].get("title", "") == rtitle):
                matched_review = review
                break
        if matched_review:
            refined_idea["review"] = matched_review
        elif i < len(reviews):
            refined_idea["review"] = reviews[i]

    return parsed


def _gather_rescue_context(ideas: list[dict], registry) -> str:
    """Gather extra context for stubborn ideas: arXiv prior art + discovery insights.

    Does NOT make LLM calls — only arXiv API searches and registry reads.
    """
    parts: list[str] = []

    # Prior art search per idea
    for idea in ideas:
        title = idea.get("title", "Untitled")
        queries = _build_prior_art_queries(idea)
        papers: list[dict] = []
        seen: set[str] = set()
        for q in queries:
            try:
                results = search_by_query(q, max_results=10)
                for p in results:
                    aid = p.get("arxiv_id", "")
                    if aid and aid not in seen:
                        seen.add(aid)
                        papers.append(p)
            except Exception:
                continue

        if papers:
            lines = [f"### Prior Art for: {title}"]
            for p in papers[:15]:
                abstract = (p.get("abstract") or "")[:200]
                lines.append(f"- [{p.get('arxiv_id', '?')}] {p.get('title', '?')}: {abstract}...")
            parts.append("\n".join(lines))

    # Discovery context (trending + math insights)
    disc = _load_discovery_context(registry)
    if disc:
        parts.append(disc)

    return "\n\n".join(parts) or "(No additional context available)"


def _format_ideas_with_weakness_history(
    ideas: list[dict],
    review_history: list[dict],
) -> str:
    """Format stubborn ideas with accumulated weaknesses from all review rounds."""
    lines: list[str] = []
    for i, idea in enumerate(ideas):
        lines.append(f"### Idea {i+1}: {idea.get('title', 'Untitled')}")
        lines.append(f"Problem: {idea.get('problem', '')}")
        lines.append(f"Method: {idea.get('method', '')}")
        lines.append(f"Experiment: {idea.get('experiment_plan', '')}")

        # Collect weaknesses from all rounds
        all_weaknesses: list[str] = []
        for rh in review_history:
            round_num = rh.get("round", "?")
            for review in rh.get("reviews", []):
                # Match by title (approximate — titles change across rounds)
                rtitle = review.get("idea_title", "")
                # Also try matching the current idea's title
                if (rtitle and rtitle == idea.get("title", "")) or (not all_weaknesses):
                    for w in review.get("weaknesses", []):
                        tagged = f"[Round {round_num}] {w}"
                        if tagged not in all_weaknesses:
                            all_weaknesses.append(tagged)

        if all_weaknesses:
            lines.append("\n**Accumulated Weaknesses:**")
            for w in all_weaknesses:
                lines.append(f"  - {w}")
        lines.append("")
    return "\n".join(lines)


def _format_accepted_ideas(ideas: list[dict]) -> str:
    """Format accepted ideas as reference examples for rescue prompts."""
    if not ideas:
        return "(No accepted ideas available)"
    lines: list[str] = []
    for i, idea in enumerate(ideas, 1):
        lines.append(f"### Accepted Idea {i}: {idea.get('title', 'Untitled')}")
        lines.append(f"Problem: {idea.get('problem', '')}")
        lines.append(f"Method: {idea.get('method', '')}")
        review = idea.get("review", {})
        if review:
            lines.append(f"Review: novelty={review.get('novelty', '?')}, "
                         f"overall={review.get('overall', '?')}")
            strengths = review.get("strengths", [])
            if strengths:
                lines.append("Strengths: " + "; ".join(strengths))
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stage 0: Parallel Multi-CLI Research
# ---------------------------------------------------------------------------

def _build_technique_summary(papers: list[dict]) -> str:
    """Extract and deduplicate methods from papers for cross-domain prompt."""
    methods: list[str] = []
    seen: set[str] = set()
    for p in papers:
        m = p.get("method", "")
        if m:
            key = m.strip().lower()[:80]
            if key not in seen:
                seen.add(key)
                methods.append(f"- {m.strip()}")
    return "\n".join(methods[:20]) or "(No methods extracted from papers)"


def _format_research_context(research: dict) -> str:
    """Format the 3 research results into Markdown for injection into the idea prompt."""
    parts: list[str] = []

    gap = research.get("gap_analysis")
    if gap:
        parts.append("### Gap Analysis")
        if isinstance(gap, dict):
            gaps = gap.get("gaps", [])
            for g in gaps[:8]:
                dim = g.get("dimension", "?")
                desc = g.get("description", "")
                opp = g.get("opportunity_level", "?")
                parts.append(f"- [{opp}] **{dim}**: {desc}")
            meta = gap.get("meta_observation", "")
            if meta:
                parts.append(f"\n*Meta*: {meta}")
        else:
            parts.append(str(gap)[:2000])

    cross = research.get("cross_domain")
    if cross:
        parts.append("\n### Cross-Domain Techniques")
        if isinstance(cross, dict):
            techniques = cross.get("cross_domain_techniques", [])
            for t in techniques[:6]:
                name = t.get("technique_name", "?")
                angle = t.get("application_angle", "")
                nov = t.get("novelty_potential", "?")
                parts.append(f"- [{nov}] **{name}**: {angle}")
            synthesis = cross.get("synthesis", "")
            if synthesis:
                parts.append(f"\n*Synthesis*: {synthesis}")
        else:
            parts.append(str(cross)[:2000])

    landscape = research.get("prior_art_landscape")
    if landscape:
        parts.append("\n### Prior Art Landscape")
        if isinstance(landscape, dict):
            for sa in landscape.get("sub_areas", [])[:6]:
                name = sa.get("sub_area", "?")
                maturity = sa.get("maturity", "?")
                trend = sa.get("trend_direction", "?")
                comp = sa.get("competition_density", "?")
                open_qs = sa.get("open_questions", [])
                parts.append(f"- **{name}** [{maturity}, {trend}, competition={comp}]")
                for oq in open_qs[:2]:
                    parts.append(f"  - Open: {oq}")
            ws = landscape.get("white_spaces", [])
            if ws:
                parts.append("\n**White Spaces:**")
                for w in ws:
                    parts.append(f"- {w}")
            contrarian = landscape.get("contrarian_opportunities", [])
            if contrarian:
                parts.append("\n**Contrarian Opportunities:**")
                for c in contrarian:
                    parts.append(f"- {c}")
            summary = landscape.get("landscape_summary", "")
            if summary:
                parts.append(f"\n*Summary*: {summary}")
        else:
            parts.append(str(landscape)[:2000])

    return "\n".join(parts) or ""


def _run_parallel_research(
    topic_name: str,
    papers: list[dict],
    paper_summaries: str,
    discovery_context: str,
    cfg: dict,
) -> dict:
    """Run 3 parallel research threads using Claude/Codex/Copilot CLIs.

    Returns dict with keys: gap_analysis, cross_domain, prior_art_landscape.
    Each value is a parsed dict or None.
    """
    n_papers = len(papers)
    technique_summary = _build_technique_summary(papers)

    gap_prompt = _RESEARCH_GAP_ANALYSIS_PROMPT.format(
        topic_name=topic_name,
        n_papers=n_papers,
        paper_summaries=paper_summaries,
        discovery_context=discovery_context,
    )
    cross_prompt = _RESEARCH_CROSS_DOMAIN_PROMPT.format(
        topic_name=topic_name,
        technique_summary=technique_summary,
    )
    landscape_prompt = _RESEARCH_PRIOR_ART_LANDSCAPE_PROMPT.format(
        topic_name=topic_name,
        n_papers=n_papers,
        paper_summaries=paper_summaries,
    )

    def _run_gap(prompt: str) -> dict | None:
        """Thread 1: Claude (opus) for gap analysis."""
        raw = call_cli(prompt, cfg, timeout=600)
        if raw:
            parsed = _parse_json_with_repair(raw, cfg, label="gap_analysis")
            if isinstance(parsed, dict):
                return parsed
        return None

    def _run_cross(prompt: str) -> dict | None:
        """Thread 2: Codex first, fallback to Claude opus."""
        raw = call_codex(prompt, cfg, timeout=600)
        if not raw:
            log.info("Codex failed for cross-domain, falling back to Claude opus")
            raw = call_cli(prompt, cfg, model="opus", timeout=600)
        if raw:
            parsed = _parse_json_with_repair(raw, cfg, label="cross_domain")
            if isinstance(parsed, dict):
                return parsed
        return None

    def _run_landscape(prompt: str) -> dict | None:
        """Thread 3: Copilot first, fallback to Claude opus."""
        raw = call_copilot(prompt, cfg, timeout=600)
        if not raw:
            log.info("Copilot failed for landscape, falling back to Claude opus")
            raw = call_cli(prompt, cfg, model="opus", timeout=600)
        if raw:
            parsed = _parse_json_with_repair(raw, cfg, label="prior_art_landscape")
            if isinstance(parsed, dict):
                return parsed
        return None

    result = {"gap_analysis": None, "cross_domain": None, "prior_art_landscape": None}

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(_run_gap, gap_prompt): "gap_analysis",
            executor.submit(_run_cross, cross_prompt): "cross_domain",
            executor.submit(_run_landscape, landscape_prompt): "prior_art_landscape",
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                result[key] = future.result()
                if result[key]:
                    log.info("Stage 0: %s research completed successfully", key)
                else:
                    log.warning("Stage 0: %s research returned no usable result", key)
            except Exception as exc:
                log.warning("Stage 0: %s research failed: %s", key, exc)

    return result


# ---------------------------------------------------------------------------
# Stage 1a: Novelty Prescreen (arXiv API only, 0 LLM calls)
# ---------------------------------------------------------------------------

def _prescreen_novelty(ideas: list[dict], cfg: dict) -> list[dict]:
    """Prescreen ideas for novelty using arXiv title overlap heuristic.

    For each idea, searches arXiv with 3 queries × 10 results, then computes
    word overlap between idea title and retrieved paper titles.

    Mutates ideas in-place (adds _prescreen dict) and returns filtered list.
    Ideas with novelty_score ≤ 2 + HIGH_OVERLAP are dropped.
    Ideas with HIGH_OVERLAP get novelty_score clamped to ≤ 4.
    """

    def _title_words(title: str) -> set[str]:
        """Extract meaningful words from a title (lowercase, len >= 3)."""
        stop = {"the", "and", "for", "with", "from", "that", "this", "are",
                "was", "were", "been", "being", "have", "has", "had", "its",
                "into", "through", "during", "before", "after", "above",
                "below", "between", "under", "over", "can", "via", "using",
                "based", "towards", "toward", "about", "our", "their", "new",
                "novel", "efficient", "effective", "improved", "learning",
                "approach", "method", "framework", "model", "models"}
        words = re.sub(r"[^\w\s]", " ", title.lower()).split()
        return {w for w in words if len(w) >= 3 and w not in stop}

    def _compute_overlap(idea_words: set[str], paper_title: str) -> float:
        """Jaccard-like overlap between idea title words and paper title words."""
        pw = _title_words(paper_title)
        if not idea_words or not pw:
            return 0.0
        intersection = idea_words & pw
        union = idea_words | pw
        return len(intersection) / len(union) if union else 0.0

    def _search_one_idea(idea: dict) -> dict:
        """Search arXiv for one idea, return prescreen result."""
        queries = _build_prior_art_queries(idea)[:3]
        all_papers: list[dict] = []
        seen_ids: set[str] = set()
        for q in queries:
            try:
                results = search_by_query(q, max_results=10)
                for p in results:
                    aid = p.get("arxiv_id", "")
                    if aid and aid not in seen_ids:
                        seen_ids.add(aid)
                        all_papers.append(p)
            except Exception:
                continue

        idea_words = _title_words(idea.get("title", ""))
        overlaps: list[dict] = []
        max_overlap = 0.0
        for p in all_papers:
            score = _compute_overlap(idea_words, p.get("title", ""))
            if score > 0.15:
                overlaps.append({
                    "arxiv_id": p.get("arxiv_id", ""),
                    "title": p.get("title", ""),
                    "overlap_score": round(score, 3),
                })
            max_overlap = max(max_overlap, score)

        overlaps.sort(key=lambda x: x["overlap_score"], reverse=True)

        if max_overlap >= 0.4:
            level = "HIGH_OVERLAP"
        elif max_overlap >= 0.25:
            level = "MODERATE_OVERLAP"
        else:
            level = "LOW_OVERLAP"

        return {
            "overlap_level": level,
            "max_overlap": round(max_overlap, 3),
            "top_matches": overlaps[:5],
            "total_retrieved": len(all_papers),
        }

    # Parallel arXiv search
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(_search_one_idea, idea): i
                   for i, idea in enumerate(ideas)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                ideas[idx]["_prescreen"] = future.result()
            except Exception as exc:
                log.warning("Prescreen failed for idea %d: %s", idx, exc)
                ideas[idx]["_prescreen"] = {
                    "overlap_level": "LOW_OVERLAP",
                    "max_overlap": 0.0,
                    "top_matches": [],
                    "total_retrieved": 0,
                }

    # Apply clamping and filtering
    kept: list[dict] = []
    for idea in ideas:
        ps = idea.get("_prescreen", {})
        level = ps.get("overlap_level", "LOW_OVERLAP")
        novelty = idea.get("novelty_score", 5)

        if level == "HIGH_OVERLAP":
            old_score = novelty
            novelty = min(novelty, 4)
            idea["novelty_score"] = novelty
            if old_score != novelty:
                log.info("  Prescreen: clamped novelty %d→%d for '%s' (HIGH_OVERLAP)",
                         old_score, novelty, idea.get("title", "?")[:60])

        if novelty <= 2 and level == "HIGH_OVERLAP":
            log.info("  Prescreen DROP: '%s' (novelty=%d, %s)",
                     idea.get("title", "?")[:60], novelty, level)
            continue

        kept.append(idea)

    dropped = len(ideas) - len(kept)
    if dropped:
        log.info("Prescreen: dropped %d ideas, %d remaining", dropped, len(kept))

    return kept


# ---------------------------------------------------------------------------
# Stage 1.5: Novelty Challenge Pipeline
# ---------------------------------------------------------------------------

def _run_novelty_challenge(
    idea: dict,
    research_context: str,
    paper_summaries: str,
    cfg: dict,
    *,
    wild_perspectives: str = "",
) -> dict:
    """Run parallel challenge on a single idea.

    3 core challengers always run. If wild_perspectives is provided (round 2+),
    a 4th challenger (Wild Perspective) runs in parallel.

    Returns dict with keys: assumptions, analogies, contradictions, wild_insights.
    """
    title = idea.get("title", "Untitled")
    problem = idea.get("problem", "")
    method = idea.get("method", "")
    experiment = idea.get("experiment_plan", "")
    topic = idea.get("_topic_name", "")

    prompt_flip = _CHALLENGE_ASSUMPTION_FLIP_PROMPT.format(
        topic_name=topic, research_context=research_context,
        idea_title=title, idea_problem=problem,
        idea_method=method, idea_experiment=experiment,
    )
    prompt_analogy = _CHALLENGE_ANALOGICAL_LEAP_PROMPT.format(
        topic_name=topic, research_context=research_context,
        idea_title=title, idea_problem=problem,
        idea_method=method, idea_experiment=experiment,
    )
    prompt_contra = _CHALLENGE_CONTRADICTION_PROMPT.format(
        topic_name=topic, paper_summaries=paper_summaries,
        idea_title=title, idea_problem=problem,
        idea_method=method, idea_experiment=experiment,
    )

    result: dict = {
        "assumptions": [], "analogies": [], "contradictions": [], "wild_insights": [],
    }

    def _run_flip() -> list[dict]:
        """Thread 1: Codex → Claude fallback for assumption flipping."""
        raw = call_codex(prompt_flip, cfg, timeout=600)
        if not raw:
            log.info("Assumption Flipper: Codex failed, falling back to Claude")
            raw = call_cli(prompt_flip, cfg, model="opus", timeout=600)
        if raw:
            parsed = _parse_json_with_repair(raw, cfg, label="challenge_flip")
            if isinstance(parsed, dict):
                return parsed.get("assumptions", [])
        return []

    def _run_analogy() -> list[dict]:
        """Thread 2: Copilot → Claude fallback for analogical leaps."""
        raw = call_copilot(prompt_analogy, cfg, timeout=600)
        if not raw:
            log.info("Analogical Leaper: Copilot failed, falling back to Claude")
            raw = call_cli(prompt_analogy, cfg, model="opus", timeout=600)
        if raw:
            parsed = _parse_json_with_repair(raw, cfg, label="challenge_analogy")
            if isinstance(parsed, dict):
                return parsed.get("analogies", [])
        return []

    def _run_contra() -> list[dict]:
        """Thread 3: Claude opus for contradiction finding."""
        raw = call_cli(prompt_contra, cfg, model="opus", timeout=600)
        if raw:
            parsed = _parse_json_with_repair(raw, cfg, label="challenge_contra")
            if isinstance(parsed, dict):
                return parsed.get("contradictions", [])
        return []

    def _run_wild() -> list[dict]:
        """Thread 4: Copilot → Claude fallback for wild perspective challenge."""
        if not wild_perspectives:
            return []
        prompt = _CHALLENGE_WILD_PERSPECTIVE_PROMPT.format(
            topic_name=topic, idea_title=title, idea_problem=problem,
            idea_method=method, idea_experiment=experiment,
            wild_perspectives=wild_perspectives,
        )
        raw = call_copilot(prompt, cfg, timeout=600)
        if not raw:
            log.info("Wild Perspective: Copilot failed, falling back to Claude")
            raw = call_cli(prompt, cfg, model="opus", timeout=600)
        if raw:
            parsed = _parse_json_with_repair(raw, cfg, label="challenge_wild")
            if isinstance(parsed, dict):
                return parsed.get("wild_insights", [])
        return []

    max_workers = 4 if wild_perspectives else 3
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_flip): "assumptions",
            executor.submit(_run_analogy): "analogies",
            executor.submit(_run_contra): "contradictions",
        }
        if wild_perspectives:
            futures[executor.submit(_run_wild)] = "wild_insights"

        for future in as_completed(futures):
            key = futures[future]
            try:
                result[key] = future.result()
                log.info("  Challenge %s: %d results for '%s'",
                         key, len(result[key]), title[:40])
            except Exception as exc:
                log.warning("  Challenge %s failed for '%s': %s", key, title[:40], exc)

    return result


def _merge_challenge_results(accumulated: dict, new: dict) -> dict:
    """Merge new challenge results into accumulated results, deduplicating by title."""
    merged = {}
    for key in ("assumptions", "analogies", "contradictions", "wild_insights"):
        existing = list(accumulated.get(key, []))
        existing_titles = {
            item.get("mutation_title") or item.get("new_angle_title")
            or item.get("opportunity_title") or item.get("mutation_title", "")
            for item in existing
        }
        for item in new.get(key, []):
            item_title = (
                item.get("mutation_title") or item.get("new_angle_title")
                or item.get("opportunity_title") or ""
            )
            if item_title not in existing_titles:
                existing.append(item)
                existing_titles.add(item_title)
        merged[key] = existing
    return merged


def _format_challenge_texts(challenges: dict) -> tuple[str, str, str, str]:
    """Format challenge results into text blocks for prompts.

    Returns (assumptions_text, analogies_text, contradictions_text, wild_text).
    """
    assumption_lines = []
    for a in challenges.get("assumptions", []):
        assumption_lines.append(
            f"- Assumption: {a.get('assumption', '?')}\n"
            f"  Flipped: {a.get('flipped', '?')}\n"
            f"  Mutation: {a.get('mutation_title', '?')} — {a.get('mutation_method', '')}"
        )
    analogy_lines = []
    for a in challenges.get("analogies", []):
        analogy_lines.append(
            f"- From {a.get('source_domain', '?')}: {a.get('analogy', '')}\n"
            f"  New angle: {a.get('new_angle_title', '?')} — {a.get('new_angle_method', '')}"
        )
    contradiction_lines = []
    for c in challenges.get("contradictions", []):
        contradiction_lines.append(
            f"- [{c.get('type', '?')}] {c.get('description', '')}\n"
            f"  Opportunity: {c.get('opportunity_title', '?')} — {c.get('opportunity_method', '')}"
        )
    wild_lines = []
    for w in challenges.get("wild_insights", []):
        wild_lines.append(
            f"- [{w.get('type', '?')}] Blind spot: {w.get('academic_blind_spot', '?')}\n"
            f"  Practitioner view: {w.get('source_perspective', '?')}\n"
            f"  Mutation: {w.get('mutation_title', '?')} — {w.get('mutation_method', '')}"
        )
    return (
        "\n".join(assumption_lines) or "(none produced)",
        "\n".join(analogy_lines) or "(none produced)",
        "\n".join(contradiction_lines) or "(none produced)",
        "\n".join(wild_lines) or "(none produced)",
    )


def _verify_novelty(
    original: dict,
    deepened: dict,
    cfg: dict,
) -> dict:
    """Cross-verify a deepened idea using a DIFFERENT LLM (Codex → Claude fallback).

    Returns verification dict with overall_verdict and improvement_hint.
    """
    prompt = _NOVELTY_VERIFY_PROMPT.format(
        original_title=original.get("title", ""),
        original_method=original.get("method", ""),
        deepened_title=deepened.get("title", ""),
        deepened_problem=deepened.get("problem", ""),
        deepened_method=deepened.get("method", ""),
        deepened_new=deepened.get("what_is_genuinely_new", ""),
        tension_used=deepened.get("tension_used", "(not specified)"),
        novelty_test=deepened.get("three_sentence_novelty_test", "(not provided)"),
    )

    # Use Codex for verification (different LLM than the opus that generated)
    raw = call_codex(prompt, cfg, timeout=300)
    if not raw:
        log.info("Novelty verify: Codex failed, falling back to Claude sonnet")
        raw = call_cli(prompt, cfg, model="sonnet", timeout=300)
    if not raw:
        log.warning("Novelty verify: all backends failed")
        return {"overall_verdict": "NOVEL", "diagnosis": "Verification unavailable",
                "improvement_hint": None}

    parsed = _parse_json_with_repair(raw, cfg, label="novelty_verify")
    if not isinstance(parsed, dict):
        return {"overall_verdict": "NOVEL", "diagnosis": "Verification parse failed",
                "improvement_hint": None}
    return parsed


def _deepen_novelty(
    idea: dict,
    challenges: dict,
    research_context: str,
    cfg: dict,
    max_attempts: int = 2,
) -> dict:
    """Synthesize challenge results into a deepened idea via collision synthesis.

    Uses a verify-retry loop: after synthesis, a DIFFERENT LLM checks whether
    the result is genuinely novel. If STILL_AB, retries with the diagnosis
    injected as context. Max 2 attempts (1 initial + 1 retry).
    """
    assumption_text, analogy_text, contradiction_text, wild_text = (
        _format_challenge_texts(challenges)
    )

    # Build extras string for optional fields
    extras_parts = []
    if idea.get("novelty_reasoning"):
        extras_parts.append(f"Novelty Reasoning: {idea['novelty_reasoning']}")
    if idea.get("closest_prior_work"):
        extras_parts.append(f"Closest Prior Work: {idea['closest_prior_work']}")
    if idea.get("what_is_genuinely_new"):
        extras_parts.append(f"Genuinely New: {idea['what_is_genuinely_new']}")

    base_kwargs = dict(
        topic_name=idea.get("_topic_name", ""),
        idea_title=idea.get("title", "Untitled"),
        idea_problem=idea.get("problem", ""),
        idea_method=idea.get("method", ""),
        idea_motivation=idea.get("motivation", ""),
        idea_experiment=idea.get("experiment_plan", ""),
        idea_novelty=idea.get("novelty_score", "?"),
        idea_extras="\n".join(extras_parts) if extras_parts else "(none)",
        assumption_flips=assumption_text,
        analogical_leaps=analogy_text,
        contradictions=contradiction_text,
        wild_perspectives=wild_text,
    )

    best_result = idea  # fallback to original
    for attempt in range(max_attempts):
        # Build retry context for attempt > 0
        if attempt == 0:
            retry_ctx = ""
        else:
            retry_ctx = (
                f"\n## RETRY — Previous Attempt Failed Verification\n"
                f"A different reviewer judged your previous attempt as STILL_AB:\n"
                f"- Decomposition: {prev_verification.get('decomposition_attempt', '?')}\n"
                f"- Subtraction: {prev_verification.get('subtraction_analysis', '?')}\n"
                f"- Origin: {prev_verification.get('origin_analysis', '?')}\n"
                f"- Diagnosis: {prev_verification.get('diagnosis', '?')}\n"
                f"- Hint: {prev_verification.get('improvement_hint', '?')}\n\n"
                f"You MUST address this diagnosis. Find a DIFFERENT tension to resolve.\n"
            )

        prompt = _NOVELTY_DEEPEN_PROMPT.format(**base_kwargs, retry_context=retry_ctx)

        raw = call_cli(prompt, cfg, model="opus", timeout=600)
        if not raw:
            log.warning("Novelty deepen attempt %d failed for '%s'",
                        attempt + 1, idea.get("title", "?")[:40])
            break

        parsed = _parse_json_with_repair(raw, cfg, label="novelty_deepen")
        if not isinstance(parsed, dict):
            log.warning("Novelty deepen attempt %d returned non-dict for '%s'",
                        attempt + 1, idea.get("title", "?")[:40])
            break

        best_result = parsed

        # Cross-verify with a different LLM
        log.info("  Verifying deepened idea (attempt %d)...", attempt + 1)
        verification = _verify_novelty(idea, parsed, cfg)
        verdict = verification.get("overall_verdict", "NOVEL")

        if verdict == "NOVEL":
            log.info("  Verification PASSED: genuinely novel")
            parsed["_verification"] = verification
            best_result = parsed
            break
        else:
            log.info("  Verification FAILED (STILL_AB): %s",
                     verification.get("diagnosis", "?")[:100])
            parsed["_verification"] = verification
            best_result = parsed  # keep the attempt even if STILL_AB (better than original)
            prev_verification = verification  # noqa: F841 — used in next iteration's retry_ctx

    # Carry over internal tracking fields
    if best_result is not idea:
        for key in ("_revise_streak", "_prescreen", "_topic_name"):
            if key in idea:
                best_result[key] = idea[key]
        best_result["_deepened"] = True

    return best_result


_MAX_NOVELTY_ROUNDS = 3


def _count_challenges(challenges: dict) -> int:
    """Count total challenge results across all categories."""
    return sum(
        len(challenges.get(k, []))
        for k in ("assumptions", "analogies", "contradictions", "wild_insights")
    )


def _run_novelty_pipeline_for_idea(
    idea: dict,
    research_context: str,
    paper_summaries: str,
    cfg: dict,
) -> dict:
    """Multi-round novelty challenge loop for a SINGLE idea.

    Round 1: Academic challenge (3-way) → deepen → verify
    Round 2: + Web perspectives (Reddit, HN) → 4-way re-challenge → deepen → verify
    Round 3: + Expanded web (Zhihu, Xiaohongshu, StackOverflow, etc.) → 4-way → deepen → verify

    Each round ACCUMULATES challenge results — later rounds see everything
    from previous rounds plus new information. The deepen step always gets
    the full accumulated context.

    Returns the best deepened idea, or the original if all rounds fail.
    """
    title = idea.get("title", "?")[:50]
    accumulated_challenges: dict = {
        "assumptions": [], "analogies": [], "contradictions": [], "wild_insights": [],
    }
    best_result = idea
    best_verdict = "STILL_AB"
    prev_verification: dict = {}

    for round_num in range(1, _MAX_NOVELTY_ROUNDS + 1):
        log.info("  Round %d/%d for '%s'", round_num, _MAX_NOVELTY_ROUNDS, title)

        # --- Phase A: Gather wild perspectives (round 2+) ---
        wild_text = ""
        if round_num >= 2:
            log.info("    Gathering wild perspectives (round %d)...", round_num)
            wild_text = gather_perspectives(
                idea_title=idea.get("title", ""),
                idea_problem=idea.get("problem", ""),
                idea_method=idea.get("method", ""),
                round_num=round_num,
            )
            if wild_text:
                n_lines = wild_text.count("\n") + 1
                log.info("    Wild perspectives: %d lines from web platforms", n_lines)
            else:
                log.info("    No wild perspectives found this round")

        # --- Phase B: Challenge (3-way or 4-way parallel) ---
        log.info("    Challenging (%s)...",
                 "4-way with wild" if wild_text else "3-way academic")
        new_challenges = _run_novelty_challenge(
            idea, research_context, paper_summaries, cfg,
            wild_perspectives=wild_text,
        )

        # Accumulate (merge, dedup)
        accumulated_challenges = _merge_challenge_results(
            accumulated_challenges, new_challenges,
        )
        total = _count_challenges(accumulated_challenges)
        if total == 0:
            log.info("    No challenge results after round %d, skipping deepen", round_num)
            continue

        log.info("    Accumulated: %d challenges total", total)

        # --- Phase C: Deepen (collision synthesis with ALL accumulated) ---
        # _deepen_novelty internally runs verify-retry loop
        deepened = _deepen_novelty(
            idea, accumulated_challenges, research_context, cfg,
            max_attempts=2 if round_num == _MAX_NOVELTY_ROUNDS else 1,
        )

        if deepened is idea:
            log.info("    Deepen produced nothing new, continuing")
            continue

        # _deepen_novelty already verified internally — use its result
        verification = deepened.get("_verification", {})
        verdict = verification.get("overall_verdict", "NOVEL")
        deepened["_challenge_round"] = round_num

        # Track best result across rounds
        if verdict == "NOVEL":
            log.info("    NOVEL after round %d!", round_num)
            best_result = deepened
            best_verdict = "NOVEL"
            break
        else:
            diag = verification.get("diagnosis", "?")[:80]
            log.info("    STILL_AB after round %d: %s", round_num, diag)
            # Still keep it if it's better than what we had
            best_result = deepened
            prev_verification = verification

    if best_verdict != "NOVEL":
        log.info("  Idea '%s' didn't reach NOVEL after %d rounds (keeping best attempt)",
                 title, _MAX_NOVELTY_ROUNDS)

    # Carry over internal tracking fields
    if best_result is not idea:
        for key in ("_revise_streak", "_prescreen", "_topic_name"):
            if key in idea:
                best_result[key] = idea[key]
        best_result["_deepened"] = True

    return best_result


def _run_novelty_pipeline(
    ideas: list[dict],
    research_context: str,
    paper_summaries: str,
    cfg: dict,
) -> list[dict]:
    """Run multi-round novelty challenge + deepening on qualifying ideas.

    Ideas with novelty_score < 3 pass through unchanged (likely prescreen-dropped
    or truly hopeless). All others go through up to 3 rounds of:
      challenge → deepen → verify → (web research) → re-challenge → ...

    Each round brings NEW information (web perspectives from diverse platforms)
    and accumulates challenge results across rounds.
    """
    qualifying = [(i, idea) for i, idea in enumerate(ideas)
                  if idea.get("novelty_score", 0) >= 3]
    if not qualifying:
        log.info("Stage 1.5: No ideas qualify for novelty challenge (all novelty < 3)")
        return ideas

    log.info("Stage 1.5: Running multi-round novelty challenge for %d/%d ideas "
             "(max %d rounds each)", len(qualifying), len(ideas), _MAX_NOVELTY_ROUNDS)

    result = list(ideas)  # copy
    for idx, idea in qualifying:
        title = idea.get("title", "?")[:50]
        log.info("Stage 1.5: Processing idea '%s'", title)

        deepened = _run_novelty_pipeline_for_idea(
            idea, research_context, paper_summaries, cfg,
        )

        if deepened is not idea:
            old_novelty = idea.get("novelty_score", "?")
            new_novelty = deepened.get("novelty_score", "?")
            challenge_round = deepened.get("_challenge_round", "?")
            verification = deepened.get("_verification", {})
            verdict = verification.get("overall_verdict", "?")
            log.info("  Result: novelty %s → %s, verdict=%s, rounds=%s, title='%s'",
                     old_novelty, new_novelty, verdict, challenge_round,
                     deepened.get("title", "?")[:50])
            result[idx] = deepened

    n_deepened = sum(1 for idea in result if idea.get("_deepened"))
    n_novel = sum(
        1 for idea in result
        if idea.get("_verification", {}).get("overall_verdict") == "NOVEL"
    )
    log.info("Stage 1.5: %d ideas deepened (%d verified NOVEL), %d unchanged",
             n_deepened, n_novel, len(ideas) - n_deepened)
    return result


def _rescue_refine_ideas(
    ideas: list[dict],
    review_history: list[dict],
    topic_name: str,
    paper_summaries: str,
    n_papers: int,
    extra_context: str,
    n_rounds: int,
    cfg: dict,
    accepted_ideas: list[dict] | None = None,
) -> list[dict]:
    """Enhanced refine for stubborn ideas with extra research context. 1 LLM call.

    Returns refined ideas. Items with verdict=DROP are removed.
    """
    ideas_with_history = _format_ideas_with_weakness_history(ideas, review_history)
    accepted_text = _format_accepted_ideas(accepted_ideas or [])
    prompt = _IDEA_RESCUE_PROMPT.format(
        topic_name=topic_name,
        n_papers=n_papers,
        paper_summaries=paper_summaries,
        extra_context=extra_context,
        accepted_ideas=accepted_text,
        ideas_with_history=ideas_with_history,
        n_rounds=n_rounds,
    )
    raw = call_cli(prompt, cfg)
    if not raw:
        log.warning("Rescue refine LLM call failed, keeping originals")
        return ideas

    parsed = _parse_json_with_repair(raw, cfg, label="rescue_refine")
    if not isinstance(parsed, list):
        log.warning("Rescue refine returned non-list, keeping originals")
        return ideas

    # Filter out DROP verdicts
    kept: list[dict] = []
    for item in parsed:
        verdict = item.get("verdict", "REVISED")
        if verdict == "DROP":
            log.info("  Rescue DROP: %s", item.get("title", "?"))
        else:
            kept.append(item)

    return kept


def _pivot_refine_ideas(
    ideas: list[dict],
    rescue_reviews: list[dict],
    topic_name: str,
    paper_summaries: str,
    n_papers: int,
    extra_context: str,
    n_rounds: int,
    cfg: dict,
    accepted_ideas: list[dict] | None = None,
) -> list[dict]:
    """Second-attempt rescue via cross-pollination pivot. 1 LLM call.

    Takes ideas that failed the first rescue + their rescue review feedback.
    Returns pivoted ideas.
    """
    accepted_text = _format_accepted_ideas(accepted_ideas or [])

    # Format ideas with rescue review feedback
    lines: list[str] = []
    for i, idea in enumerate(ideas):
        lines.append(f"### Idea {i+1}: {idea.get('title', 'Untitled')}")
        lines.append(f"Problem: {idea.get('problem', '')}")
        lines.append(f"Method: {idea.get('method', '')}")
        if i < len(rescue_reviews):
            rev = rescue_reviews[i]
            lines.append(f"\n**Why it STILL failed (rescue review):**")
            for w in rev.get("weaknesses", []):
                lines.append(f"  - {w}")
            insts = rev.get("revision_instructions", [])
            if insts:
                lines.append("**Reviewer suggestions:**")
                for inst in insts:
                    lines.append(f"  - {inst}")
        lines.append("")

    prompt = _IDEA_RESCUE_PIVOT_PROMPT.format(
        topic_name=topic_name,
        n_papers=n_papers,
        paper_summaries=paper_summaries,
        extra_context=extra_context,
        accepted_ideas=accepted_text,
        ideas_with_feedback="\n".join(lines),
        n_rounds=n_rounds,
    )
    raw = call_cli(prompt, cfg)
    if not raw:
        log.warning("Pivot refine LLM call failed, keeping originals")
        return ideas

    parsed = _parse_json_with_repair(raw, cfg, label="pivot_refine")
    if not isinstance(parsed, list):
        log.warning("Pivot refine returned non-list, keeping originals")
        return ideas

    return parsed


_DEFAULT_CONTEXT_OPTIONS: dict = {
    "use_insights": True,
    "use_reports": True,
    "use_github": True,
    "use_history": True,
    "use_research_plans": True,
    "use_citations": True,
    "use_questions": True,
    "use_novelty_map": True,
}


def run_brainstorm(
    topic_id: str,
    topic_name: str,
    data_dir: str,
    cfg: dict,
    mode: str = "auto",
    user_idea: str = "",
    run_code_verification: bool = False,
    registry=None,
    max_review_rounds: int = 1,
    on_progress: "callable | None" = None,
    context_options: dict | None = None,
) -> dict:
    """Run the full brainstorm pipeline. Returns result dict with ideas + verifications."""
    max_review_rounds = max(0, min(10, max_review_rounds))  # clamp to [0, 10]
    ctx_opts = {**_DEFAULT_CONTEXT_OPTIONS, **(context_options or {})}
    log.info("=== Brainstorm started: topic=%s mode=%s review_rounds=%d ctx=%s ===",
             topic_id, mode, max_review_rounds, ctx_opts)

    def _progress(stage: str, **kw: object) -> None:
        if on_progress:
            on_progress(stage, kw)

    # Load existing papers for context
    store = Storage(data_dir, topic_id)
    try:
        papers, _ = store.get_all_arxiv(limit=200, offset=0)
    finally:
        store.close()

    # Paper summaries: citation-weighted or standard
    if ctx_opts.get("use_citations"):
        paper_summaries = _build_citation_weighted_summaries(papers)
        log.info("Using citation-weighted paper summaries")
    else:
        paper_summaries = _build_paper_summaries(papers)
    n_papers = len(papers)

    discovery_context = _load_discovery_context(registry)

    # --- Load enhanced context sources ---
    _progress("context", message="Loading context sources...")
    extra_parts: list[str] = []

    if ctx_opts.get("use_insights"):
        insights_text = _load_topic_insights(data_dir, topic_id, registry)
        if insights_text:
            extra_parts.append(insights_text)
            log.info("Loaded topic insights (%d chars)", len(insights_text))
    else:
        insights_text = ""

    if ctx_opts.get("use_reports"):
        report_text = _load_session_reports(data_dir, topic_id, registry)
        if report_text:
            extra_parts.append(report_text)
            log.info("Loaded session report (%d chars)", len(report_text))

    if ctx_opts.get("use_github"):
        github_text = _load_github_repos(data_dir, topic_id)
        if github_text:
            extra_parts.append(github_text)
            log.info("Loaded GitHub repos (%d chars)", len(github_text))

    if ctx_opts.get("use_history"):
        history_text = _load_brainstorm_history(topic_id, registry)
        if history_text:
            extra_parts.append(history_text)
            log.info("Loaded brainstorm history (%d chars)", len(history_text))
    else:
        history_text = ""

    if ctx_opts.get("use_research_plans"):
        plans_text = _load_research_plans(topic_id, registry)
        if plans_text:
            extra_parts.append(plans_text)
            log.info("Loaded research plans (%d chars)", len(plans_text))

    extra_context = "\n\n".join(extra_parts) if extra_parts else ""

    # --- Question-First generation ---
    research_questions = ""
    if ctx_opts.get("use_questions") and mode == "auto" and n_papers >= 3:
        _progress("questions", message="Generating research questions...")
        log.info("Stage 0q: Generating research questions (question-first)")
        research_questions = _generate_research_questions(
            topic_name, paper_summaries, insights_text, cfg,
        )
        if research_questions:
            extra_parts.append(research_questions)
            extra_context = "\n\n".join(extra_parts)
            log.info("Generated research questions (%d chars)", len(research_questions))

    result: dict = {
        "ideas": [],
        "literature_result": "",
        "logic_result": "",
        "code_result": "",
        "review_history": [],
        "research_context": {},
        "context_sources": {k: bool(v) for k, v in ctx_opts.items()},
    }

    # --- Stage 0: Parallel multi-CLI research ---
    _progress("research", message="Running parallel research threads...")
    research_context: dict = {}
    if mode == "auto" and n_papers >= 3:
        log.info("Stage 0: Running parallel multi-CLI research (%d papers)", n_papers)
        research_context = _run_parallel_research(
            topic_name, papers, paper_summaries, discovery_context, cfg,
        )
        n_successful = sum(1 for v in research_context.values() if v)
        log.info("Stage 0: %d/3 research threads returned results", n_successful)
        result["research_context"] = research_context
    elif mode == "auto":
        log.info("Stage 0: Skipped (only %d papers, need >= 3)", n_papers)

    # --- Stage 1: Generate ideas ---
    _progress("generating", message="Generating research ideas...")
    log.info("Stage 1: Generating ideas (mode=%s)", mode)
    if mode == "user" and user_idea:
        prompt = _USER_IDEA_PROMPT.format(
            topic_name=topic_name,
            n_papers=n_papers,
            paper_summaries=paper_summaries,
            extra_context=extra_context,
            user_idea=user_idea,
        )
        raw = call_cli(prompt, cfg)
        if raw:
            ideas = _parse_ideas(raw, single=True, cfg=cfg)
        else:
            ideas = []
    else:
        # Use research-informed prompt if we have research context
        research_ctx_text = _format_research_context(research_context)
        if research_ctx_text:
            log.info("Stage 1: Using research-informed idea generation")
            prompt = _RESEARCH_INFORMED_IDEAS_PROMPT.format(
                topic_name=topic_name,
                n_papers=n_papers,
                paper_summaries=paper_summaries,
                discovery_context=discovery_context,
                extra_context=extra_context,
                research_context=research_ctx_text,
            )
        else:
            log.info("Stage 1: Using standard idea generation (no research context)")
            prompt = _AUTO_IDEAS_PROMPT.format(
                topic_name=topic_name,
                n_papers=n_papers,
                paper_summaries=paper_summaries,
                discovery_context=discovery_context,
                extra_context=extra_context,
            )
        raw = call_cli(prompt, cfg)
        if raw:
            ideas = _parse_ideas(raw, single=False, cfg=cfg)
        else:
            ideas = []

    if not ideas:
        log.warning("Idea generation produced no results")
        result["ideas"] = []
        return result

    # Dedup near-identical ideas
    ideas = _dedup_ideas(ideas)

    result["ideas"] = ideas
    log.info("Generated %d ideas", len(ideas))

    # --- Stage 1a: Novelty prescreen ---
    if ctx_opts.get("use_novelty_map") and mode == "auto":
        # Multi-axis novelty map (replaces binary prescreen when enabled)
        _progress("prescreen", message=f"Building novelty map for {len(ideas)} ideas...", ideas_count=len(ideas))
        log.info("Stage 1a: Building multi-axis novelty map (%d ideas)", len(ideas))
        novelty_map = _build_novelty_map(ideas, history_text, insights_text, cfg)
        if novelty_map:
            for i, idea in enumerate(ideas):
                if i < len(novelty_map):
                    idea["_novelty_map"] = novelty_map[i]
                    # Attach strongest/weakest axis info
                    nm = novelty_map[i]
                    idea["_novelty_strongest"] = nm.get("strongest_axis", "")
                    idea["_novelty_weakest"] = nm.get("weakest_axis", "")
            log.info("Stage 1a: Novelty map complete for %d ideas", len(novelty_map))
        # Still run arXiv prescreen as supplementary signal
        ideas = _prescreen_novelty(ideas, cfg)
        result["ideas"] = ideas
        if not ideas:
            log.warning("All ideas dropped by novelty prescreen")
            return result
    else:
        # Standard arXiv-only prescreen
        _progress("prescreen", message=f"Prescreening {len(ideas)} ideas for novelty...", ideas_count=len(ideas))
        if mode == "auto" and ideas:
            log.info("Stage 1a: Novelty prescreen (%d ideas)", len(ideas))
            ideas = _prescreen_novelty(ideas, cfg)
            result["ideas"] = ideas
            if not ideas:
                log.warning("All ideas dropped by novelty prescreen")
                return result

    # --- Stage 1.5: Novelty Challenge Pipeline ---
    _progress("novelty", message=f"Novelty challenge pipeline ({len(ideas)} ideas)...", ideas_count=len(ideas))
    research_ctx_text = _format_research_context(research_context) if research_context else ""
    if mode == "auto" and research_ctx_text and ideas:
        # Tag ideas with topic_name for challenge prompts
        for idea in ideas:
            idea["_topic_name"] = topic_name
        log.info("Stage 1.5: Novelty challenge pipeline (%d ideas)", len(ideas))
        ideas = _run_novelty_pipeline(ideas, research_ctx_text, paper_summaries, cfg)
        result["ideas"] = ideas
        # Clean up topic_name tag
        for idea in ideas:
            idea.pop("_topic_name", None)

    # --- Stage 1b/1c: Review-Refinement Loop ---
    # ACCEPT/DROP ideas are removed from the loop immediately.
    # Only REVISE ideas continue to the next round.
    accepted_pool: list[dict] = []   # graduated ideas (ACCEPT)
    dropped_pool: list[dict] = []    # discarded ideas (DROP)

    if max_review_rounds > 0 and ideas:
        for round_num in range(1, max_review_rounds + 1):
            _progress("reviewing", message=f"Review round {round_num}/{max_review_rounds}...",
                       ideas_count=len(ideas), round=round_num, total_rounds=max_review_rounds,
                       accepted=len(accepted_pool))
            if not ideas:
                log.info("  No ideas left to review — exiting loop")
                break

            log.info("Stage 1b: Review round %d/%d (%d ideas, %d already accepted)",
                     round_num, max_review_rounds, len(ideas), len(accepted_pool))

            reviews = _review_ideas(ideas, topic_name, paper_summaries, n_papers, cfg)
            if not reviews:
                log.warning("Review round %d failed, skipping remaining rounds", round_num)
                break

            # Log review summary
            verdicts = [r.get("verdict", "?") for r in reviews]
            log.info("  Review verdicts: %s", verdicts)

            # Separate ACCEPT / CONDITIONAL_ACCEPT / DROP / REVISE
            revise_ideas: list[dict] = []
            revise_reviews: list[dict] = []
            for i, idea in enumerate(ideas):
                if i >= len(reviews):
                    revise_ideas.append(idea)
                    continue
                v = reviews[i].get("verdict", "ACCEPT")
                if v in ("ACCEPT", "CONDITIONAL_ACCEPT"):
                    idea["review"] = reviews[i]
                    idea["_revise_streak"] = 0
                    idea["_accepted_at_round"] = round_num
                    accepted_pool.append(idea)
                    label = "ACCEPT" if v == "ACCEPT" else "COND_ACCEPT"
                    log.info("  %s: '%s' graduated at round %d (overall=%.1f)",
                             label, idea.get("title", "?")[:50], round_num,
                             reviews[i].get("overall", 0))
                elif v == "DROP":
                    idea["review"] = reviews[i]
                    idea["_status"] = "dropped"
                    idea["_dropped_at_round"] = round_num
                    dropped_pool.append(idea)
                    log.info("  DROP: '%s' at round %d",
                             idea.get("title", "?")[:50], round_num)
                else:
                    idea["_revise_streak"] = idea.get("_revise_streak", 0) + 1
                    revise_ideas.append(idea)
                    revise_reviews.append(reviews[i])

            round_record = {
                "round": round_num,
                "reviews": reviews,
                "verdicts": verdicts,
            }

            # All ideas resolved — exit early
            if not revise_ideas:
                log.info("  All ideas resolved (no REVISE left) — exiting loop")
                result["review_history"].append(round_record)
                break

            # Refine only REVISE ideas
            log.info("Stage 1c: Refining %d REVISE ideas (round %d)",
                     len(revise_ideas), round_num)
            refined = _refine_ideas(
                revise_ideas, revise_reviews, topic_name,
                paper_summaries, n_papers, cfg,
                round_num=round_num,
            )
            if refined:
                # Transfer _revise_streak to refined ideas
                for j, ref in enumerate(refined):
                    if j < len(revise_ideas):
                        ref["_revise_streak"] = revise_ideas[j].get("_revise_streak", 0)
                ideas = refined
            else:
                ideas = revise_ideas

            round_record["refined_count"] = len(ideas)
            result["review_history"].append(round_record)

    # Merge: accepted first, then remaining REVISE, then dropped (kept for display)
    ideas = accepted_pool + ideas + dropped_pool
    result["ideas"] = ideas
    log.info("Review loop done: %d accepted, %d still REVISE, %d dropped",
             len(accepted_pool),
             len(ideas) - len(accepted_pool) - len(dropped_pool),
             len(dropped_pool))

    # --- Stage 1d: Rescue stubborn ideas ---
    # After the loop, ideas = accepted_pool + remaining REVISE.
    # Stubborn = ideas still in REVISE with streak >= 2.
    _progress("rescue", message="Rescuing stubborn ideas...", ideas_count=len(ideas), accepted=len(accepted_pool))
    remaining_revise = [idea for idea in ideas if idea.get("_revise_streak", 0) >= 1]
    if (max_review_rounds >= 2
            and len(result["review_history"]) >= 2
            and remaining_revise):
        stubborn = [idea for idea in remaining_revise if idea.get("_revise_streak", 0) >= 2]
        accepted = accepted_pool  # already graduated
        if stubborn:
            n_rnds = len(result["review_history"])
            log.info("Stage 1d: Rescuing %d stubborn ideas (REVISE >= 2 rounds), "
                     "%d accepted as reference", len(stubborn), len(accepted))
            # Gather extra context: arXiv prior art + discovery/math insights
            extra_ctx = _gather_rescue_context(stubborn, registry)

            # --- Attempt 1: deep revision with prior art + accepted references ---
            rescued = _rescue_refine_ideas(
                stubborn, result["review_history"],
                topic_name, paper_summaries, n_papers, extra_ctx,
                n_rnds, cfg, accepted_ideas=accepted,
            )
            still_revise: list[dict] = []
            still_revise_reviews: list[dict] = []
            if rescued:
                log.info("  Attempt 1: rescued %d ideas, reviewing", len(rescued))
                rescue_reviews = _review_ideas(
                    rescued, topic_name, paper_summaries, n_papers, cfg,
                )
                if rescue_reviews:
                    rescue_verdicts = [r.get("verdict", "?") for r in rescue_reviews]
                    log.info("  Attempt 1 verdicts: %s", rescue_verdicts)
                    kept: list[dict] = []
                    for j, r_idea in enumerate(rescued):
                        if j < len(rescue_reviews):
                            rv = rescue_reviews[j].get("verdict", "REVISE")
                            if rv == "REVISE":
                                still_revise.append(r_idea)
                                still_revise_reviews.append(rescue_reviews[j])
                            else:
                                r_idea["review"] = rescue_reviews[j]
                                r_idea["_revise_streak"] = 0
                                kept.append(r_idea)
                        else:
                            kept.append(r_idea)
                    result["review_history"].append({
                        "round": "rescue-1",
                        "reviews": rescue_reviews,
                        "verdicts": rescue_verdicts,
                    })
                    rescued = kept

            # --- Attempt 2: cross-pollination pivot for remaining failures ---
            if still_revise:
                log.info("  Attempt 2: pivoting %d ideas via cross-pollination",
                         len(still_revise))
                pivoted = _pivot_refine_ideas(
                    still_revise, still_revise_reviews,
                    topic_name, paper_summaries, n_papers, extra_ctx,
                    n_rnds, cfg, accepted_ideas=accepted,
                )
                if pivoted:
                    log.info("  Attempt 2: pivoted %d ideas, reviewing", len(pivoted))
                    pivot_reviews = _review_ideas(
                        pivoted, topic_name, paper_summaries, n_papers, cfg,
                    )
                    if pivot_reviews:
                        pivot_verdicts = [r.get("verdict", "?") for r in pivot_reviews]
                        log.info("  Attempt 2 verdicts: %s", pivot_verdicts)
                        for j, p_idea in enumerate(pivoted):
                            if j < len(pivot_reviews):
                                pv = pivot_reviews[j].get("verdict", "REVISE")
                                if pv != "REVISE":
                                    p_idea["review"] = pivot_reviews[j]
                                    p_idea["_revise_streak"] = 0
                                    rescued.append(p_idea)
                                else:
                                    p_idea["_status"] = "dropped"
                                    p_idea["_dropped_at_round"] = "rescue"
                                    rescued.append(p_idea)
                                    log.info("  Final DROP (failed both rescues): %s",
                                             p_idea.get("title", "?"))
                        result["review_history"].append({
                            "round": "rescue-2",
                            "reviews": pivot_reviews,
                            "verdicts": pivot_verdicts,
                        })
                else:
                    for sr in still_revise:
                        sr["_status"] = "dropped"
                        sr["_dropped_at_round"] = "rescue"
                        rescued.append(sr)
                        log.info("  Final DROP (pivot failed): %s",
                                 sr.get("title", "?"))

            # Replace stubborn ideas with rescued ones in the idea list
            non_stubborn = [idea for idea in ideas
                           if idea.get("_revise_streak", 0) < 2]
            ideas = non_stubborn + rescued
            result["ideas"] = ideas
            log.info("  Final idea count: %d", len(ideas))

    # --- Stage 1e: Final polish for any remaining REVISE ideas ---
    _progress("polish", message="Final polish pass...", ideas_count=len(ideas))
    remaining_revise = [
        idea for idea in ideas
        if idea.get("_accepted_at_round") is None  # not graduated
    ]
    if remaining_revise and accepted_pool:
        log.info("Stage 1e: Final polish for %d remaining REVISE ideas",
                 len(remaining_revise))
        last_reviews_for_polish = [
            idea.get("review", {}) for idea in remaining_revise
        ]
        polished = _refine_ideas(
            remaining_revise, last_reviews_for_polish,
            topic_name, paper_summaries, n_papers, cfg,
        )
        if polished:
            polish_reviews = _review_ideas(
                polished, topic_name, paper_summaries, n_papers, cfg,
            )
            if polish_reviews:
                polish_verdicts = [r.get("verdict", "?") for r in polish_reviews]
                log.info("  Polish verdicts: %s", polish_verdicts)
                for j, p_idea in enumerate(polished):
                    if j < len(polish_reviews):
                        p_idea["review"] = polish_reviews[j]
                result["review_history"].append({
                    "round": "polish",
                    "reviews": polish_reviews,
                    "verdicts": polish_verdicts,
                })
                # Replace remaining REVISE ideas with polished versions
                ideas = list(accepted_pool) + polished
                result["ideas"] = ideas
                log.info("  After polish: %d ideas (%d accepted + %d polished)",
                         len(ideas), len(accepted_pool), len(polished))

    # Clean up internal tracking fields, promote _status to status
    for idea in ideas:
        if idea.get("_status"):
            idea["status"] = idea.pop("_status")
        idea.pop("_revise_streak", None)
        idea.pop("_prescreen", None)
        idea.pop("_deepened", None)
        idea.pop("_verification", None)
        idea.pop("_challenge_round", None)
        idea.pop("_topic_name", None)
        idea.pop("_accepted_at_round", None)
        idea.pop("_dropped_at_round", None)

    # Split active vs dropped for verification (don't waste LLM calls on dropped)
    active_ideas = [i for i in ideas if i.get("status") != "dropped"]
    dropped_ideas = [i for i in ideas if i.get("status") == "dropped"]

    # --- Stage 2: Literature verification ---
    _progress("literature", message=f"Verifying literature ({len(active_ideas)} ideas)...", ideas_count=len(active_ideas))
    log.info("Stage 2: Literature verification")
    ideas_text = _format_ideas_for_prompt(active_ideas)
    lit_prompt = _LITERATURE_CHECK_PROMPT.format(
        ideas_text=ideas_text,
        paper_summaries=paper_summaries,
    )
    lit_raw = call_cli(lit_prompt, cfg)
    if lit_raw:
        result["literature_result"] = lit_raw
        # Try to parse verdicts and attach to active ideas
        lit_verdicts = _parse_json_with_repair(lit_raw, cfg, label="literature_check")
        if isinstance(lit_verdicts, list):
            for i, verdict in enumerate(lit_verdicts):
                if i < len(active_ideas):
                    active_ideas[i]["novelty_verdict"] = verdict.get("verdict", "")
                    active_ideas[i]["literature_evidence"] = verdict.get("evidence", "")

    # --- Stage 3: Logic verification ---
    _progress("logic", message="Checking logical coherence...", ideas_count=len(active_ideas))
    log.info("Stage 3: Logic verification")
    logic_prompt = _LOGIC_CHECK_PROMPT.format(ideas_text=ideas_text)
    logic_raw = call_cli(logic_prompt, cfg)
    if logic_raw:
        result["logic_result"] = logic_raw

    # --- Stage 4: Code verification (optional) ---
    if run_code_verification and active_ideas:
        _progress("code", message="Generating code proof-of-concept...", ideas_count=len(active_ideas))
        log.info("Stage 4: Code verification")
        # Only generate PoC for the highest-scoring active idea
        best = max(active_ideas, key=lambda x: x.get("novelty_score", 0) + x.get("feasibility_score", 0))
        code_prompt = _CODE_CHECK_PROMPT.format(
            idea_text=f"Title: {best.get('title', '')}\n"
                      f"Problem: {best.get('problem', '')}\n"
                      f"Method: {best.get('method', '')}\n"
                      f"Experiment: {best.get('experiment_plan', '')}",
        )
        code_raw = call_cli(code_prompt, cfg)
        if code_raw:
            result["code_result"] = code_raw

    # Final result: active ideas first, dropped ideas at the end
    result["ideas"] = active_ideas + dropped_ideas
    log.info("=== Brainstorm finished: %d active + %d dropped ideas ===",
             len(active_ideas), len(dropped_ideas))
    return result


def _parse_ideas(raw: str, single: bool = False, cfg: dict | None = None) -> list[dict]:
    """Parse idea JSON from LLM output, with optional LLM repair."""
    if cfg:
        parsed = _parse_json_with_repair(raw, cfg, label="idea_generation")
    else:
        parsed = _parse_json_safe(raw)
    if single and isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return parsed
    return []


_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "this", "that",
    "these", "those", "it", "its", "as", "not", "no", "so", "if", "than",
    "into", "about", "between", "through", "during", "before", "after",
    "above", "below", "up", "down", "out", "off", "over", "under",
    "via", "using", "based", "towards", "toward",
})


def _dedup_ideas(ideas: list[dict]) -> list[dict]:
    """Remove near-duplicate ideas by title+problem word overlap (Jaccard > 0.5)."""
    if len(ideas) <= 1:
        return ideas

    import re as _re

    def _words(text: str) -> set[str]:
        return {w for w in _re.findall(r"[a-z0-9]+", text.lower())
                if w not in _STOP_WORDS and len(w) > 2}

    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    # Build word sets from title + problem
    word_sets = []
    for idea in ideas:
        ws = _words(idea.get("title", "") + " " + idea.get("problem", ""))
        word_sets.append(ws)

    kept: list[dict] = []
    kept_ws: list[set[str]] = []
    for i, idea in enumerate(ideas):
        is_dup = False
        for j, kws in enumerate(kept_ws):
            sim = _jaccard(word_sets[i], kws)
            if sim > 0.5:
                log.info("Dedup: dropping '%s' (sim=%.2f with '%s')",
                         idea.get("title", "?")[:50], sim,
                         kept[j].get("title", "?")[:50])
                is_dup = True
                break
        if not is_dup:
            kept.append(idea)
            kept_ws.append(word_sets[i])

    if len(kept) < len(ideas):
        log.info("Dedup: %d → %d ideas", len(ideas), len(kept))
    return kept


def _parse_json_safe(raw: str) -> list | dict | None:
    """Best-effort parse JSON from LLM output."""
    text = raw.strip()
    # Strip markdown fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON structure
        for start_char, end_char in [("[", "]"), ("{", "}")]:
            start = text.find(start_char)
            end = text.rfind(end_char)
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    continue
    log.warning("Failed to parse JSON from brainstorm output")
    return None


_JSON_REPAIR_PROMPT = """\
The following text was supposed to be valid JSON but it has formatting issues \
(missing brackets, trailing commas, unescaped characters, markdown mixed in, etc.).

Extract and repair the JSON. Return ONLY valid JSON — no markdown fences, no \
explanation, no commentary. Preserve all data content exactly.

--- BEGIN BROKEN TEXT ---
{broken_text}
--- END BROKEN TEXT ---"""


def _parse_json_with_repair(
    raw: str,
    cfg: dict,
    *,
    label: str = "",
) -> list | dict | None:
    """Parse JSON from LLM output, with an LLM repair fallback.

    First tries _parse_json_safe(). If that fails and the raw text is non-trivial
    (>50 chars), sends it to Claude for JSON repair (1 extra LLM call, sonnet).
    """
    result = _parse_json_safe(raw)
    if result is not None:
        return result

    # Don't bother repairing tiny/empty outputs
    if len(raw.strip()) < 50:
        return None

    log.info("JSON parse failed for %s, attempting LLM repair (%d chars)",
             label or "output", len(raw))
    repair_prompt = _JSON_REPAIR_PROMPT.format(broken_text=raw[:8000])
    repaired = call_cli(repair_prompt, cfg, model="sonnet", timeout=60)
    if repaired:
        result = _parse_json_safe(repaired)
        if result is not None:
            log.info("JSON repair succeeded for %s", label or "output")
            return result
        log.warning("JSON repair also failed for %s", label or "output")
    else:
        log.warning("JSON repair LLM call failed for %s", label or "output")
    return None


# ---------------------------------------------------------------------------
# Prior Art Check
# ---------------------------------------------------------------------------

def _build_prior_art_queries(idea: dict) -> list[str]:
    """Build 2-3 arXiv search queries from an idea's title/method/problem."""
    queries = []

    title = idea.get("title", "")
    if title:
        # Clean up title: remove quotes, keep core terms
        clean = re.sub(r"[^\w\s-]", " ", title).strip()
        queries.append(clean)

    method = idea.get("method", "")
    if method:
        if isinstance(method, dict):
            method = method.get("description", "") or str(method)
        # Extract key technical terms from first sentence
        first_sent = method.split(".")[0].strip()
        # Keep first ~60 chars to stay focused
        queries.append(first_sent[:80])

    problem = idea.get("problem", "")
    if problem:
        if isinstance(problem, dict):
            problem = problem.get("description", "") or str(problem)
        first_sent = problem.split(".")[0].strip()
        queries.append(first_sent[:80])

    # Ensure at least 2, at most 3
    return queries[:3] if len(queries) >= 2 else (queries + queries)[:2]


_PRIOR_ART_PROMPT = """\
You are a research literature analyst assessing the novelty of a proposed idea \
against existing arXiv papers.

## Proposed Idea
Title: {idea_title}
Problem: {idea_problem}
Method: {idea_method}

## Retrieved arXiv Papers ({n_papers} total)
{papers_text}

## Your Task
Classify each retrieved paper as:
- **Prior work**: Directly foundational — the proposed idea builds on this
- **Similar work**: Addresses the same problem or uses a very similar method (high overlap)
- **Related but different**: Same general area but distinct approach/problem

Then assess overall research maturity:
- NASCENT: <3 closely related papers — very new territory
- GROWING: 3-8 closely related — active emerging area
- MATURE: 8-15 closely related — well-studied area
- SATURATED: >15 closely related — crowded, hard to differentiate

Reply ONLY with a JSON object (no markdown fences):
{{
  "prior_works": [{{"arxiv_id": "...", "title": "...", "relevance": "1-2 sentence explanation"}}],
  "similar_works": [{{"arxiv_id": "...", "title": "...", "overlap": "1-2 sentence explanation"}}],
  "maturity_level": "NASCENT|GROWING|MATURE|SATURATED",
  "total_related": <number of prior + similar works>,
  "novelty_assessment": "2-3 sentences on how novel this idea is given the literature",
  "recommendation": "PURSUE|DIFFERENTIATE|RECONSIDER",
  "recommendation_reason": "2-3 sentences explaining the recommendation"
}}

Guidelines for recommendation:
- PURSUE: Idea is novel enough to proceed as-is
- DIFFERENTIATE: Core concept exists but the specific angle is unique — needs clearer positioning
- RECONSIDER: Too much existing work — either pivot or find a stronger differentiator"""


def check_prior_art(idea: dict, cfg: dict) -> dict:
    """Search arXiv for prior/similar work and assess research maturity via LLM.

    Returns structured prior-art analysis dict.
    """
    log.info("Prior art check: %s", idea.get("title", "Untitled"))

    # Build queries and search arXiv
    queries = _build_prior_art_queries(idea)
    all_papers: list[dict] = []
    seen_ids: set[str] = set()

    for q in queries:
        results = search_by_query(q, max_results=20)
        for p in results:
            aid = p.get("arxiv_id", "")
            if aid and aid not in seen_ids:
                seen_ids.add(aid)
                all_papers.append(p)

    log.info("Prior art: %d unique papers from %d queries", len(all_papers), len(queries))

    if not all_papers:
        return {
            "prior_works": [],
            "similar_works": [],
            "maturity_level": "NASCENT",
            "total_related": 0,
            "novelty_assessment": "No related papers found on arXiv. This appears to be a very novel direction.",
            "recommendation": "PURSUE",
            "recommendation_reason": "No existing work found — this is either highly novel or the search terms need refinement.",
        }

    # Build paper text for LLM
    paper_lines = []
    for i, p in enumerate(all_papers[:60], 1):  # cap at 60 to fit context
        paper_lines.append(
            f"{i}. [{p['arxiv_id']}] {p['title']}\n"
            f"   Abstract: {p['abstract'][:300]}..."
        )
    papers_text = "\n\n".join(paper_lines)

    prompt = _PRIOR_ART_PROMPT.format(
        idea_title=idea.get("title", ""),
        idea_problem=idea.get("problem", ""),
        idea_method=idea.get("method", ""),
        n_papers=len(all_papers),
        papers_text=papers_text,
    )

    raw = call_cli(prompt, cfg)
    if not raw:
        return {
            "prior_works": [],
            "similar_works": [],
            "maturity_level": "UNKNOWN",
            "total_related": len(all_papers),
            "novelty_assessment": "LLM analysis failed. Manual review recommended.",
            "recommendation": "DIFFERENTIATE",
            "recommendation_reason": f"Found {len(all_papers)} potentially related papers but couldn't analyze them.",
        }

    parsed = _parse_json_safe(raw)
    if not isinstance(parsed, dict):
        return {
            "prior_works": [],
            "similar_works": [],
            "maturity_level": "UNKNOWN",
            "total_related": len(all_papers),
            "novelty_assessment": "Failed to parse LLM response. Manual review recommended.",
            "recommendation": "DIFFERENTIATE",
            "recommendation_reason": f"Found {len(all_papers)} potentially related papers but couldn't parse analysis.",
        }

    # Ensure all expected keys exist
    result = {
        "prior_works": parsed.get("prior_works", []),
        "similar_works": parsed.get("similar_works", []),
        "maturity_level": parsed.get("maturity_level", "UNKNOWN"),
        "total_related": parsed.get("total_related", 0),
        "novelty_assessment": parsed.get("novelty_assessment", ""),
        "recommendation": parsed.get("recommendation", "DIFFERENTIATE"),
        "recommendation_reason": parsed.get("recommendation_reason", ""),
    }

    log.info(
        "Prior art result: maturity=%s, recommendation=%s, prior=%d, similar=%d",
        result["maturity_level"],
        result["recommendation"],
        len(result["prior_works"]),
        len(result["similar_works"]),
    )
    return result
