# Deep-Dig — Agentic Paper Exploration → Insights → Ideas — Design Spec

## Problem

The tracker can summarize papers, generate cross-paper insights, and brainstorm
ideas over a topic's whole library, but there is no way to start from **one
specific paper** and dig outward along its intellectual neighborhood — what it
builds on (references), what built on it (citations), what is semantically
adjacent, and what math it could borrow — and turn that exploration into
synthesized **insights** and concrete **new ideas**.

Two capabilities are missing today:

1. **Citation chains.** Every source sets `cited_works = []`; nothing fetches a
   paper's references or its citing papers. OpenAlex returns `cited_by_count`
   but the parser drops `referenced_works`. There is no citation graph at all.
2. **A paper-seeded exploration pipeline** that combines citation traversal with
   the existing semantic (`rag`) and math (arXiv math / crossdomain) retrieval,
   then produces insights and ideas grounded in what it found.

## Solution

A new **deep-dig** feature: from one seed paper, an autonomous Claude agent
explores outward along three paths and produces, in a **two-phase, user-gated**
flow, an insights memo and then structured new ideas, with an agentic **Q&A**
follow-up. Built as **Approach B (agentic)**: the work is driven by Claude Agent
SDK tool-use loops (mirroring `brainstorm_agent.py`), not a fixed pipeline.

The three exploration paths:

- **Citation chain** — Semantic Scholar Graph API: a paper's `references` (what
  it cites) and `citations` (what cites it), influential-citation-aware. This is
  the only genuinely new external capability.
- **Semantic related** — reuse `rag.search_papers` over the topic library and
  the crossdomain corpus.
- **Math** — reuse arXiv math-category search + the crossdomain math corpus,
  driven by LLM-generated "math bridge" queries (à la `crossdomain`).

The two-phase, user-gated flow:

```
seed paper
  → [Phase 1: explore agent]  references/citations + semantic + math
                              → insights memo ([Pn]-grounded) + dig-corpus + graph
  → status = insights_ready   ── USER GATE (review, add steering, pick focus papers)
  → [Phase 2: ideate agent]   → structured new ideas (novelty/feasibility scored)
  → status = completed
  + [Q&A agent] available from insights_ready onward (agentic, can dig further)
```

**Key design move for Approach B:** the agent's tools, in addition to returning
text to the model, **record every paper they touch and every citation edge into
a shared collector** (`ctx["collector"]`). The exploration is autonomous, but
the dig-corpus, the `[Pn]` provenance, and the citation graph are reconstructed
deterministically from this tool-trace — so progress and graph visualization
work even though the traversal is model-driven.

**Robustness:** if `claude_agent_sdk` is missing or the agent fails (the same
failure surface `brainstorm_agent.py` already guards against), each phase falls
back to a deterministic path (S2 + `rag` + math → relevance prune →
`insights.generate_agentic` / `brainstorm` one-shot). The feature still works
headless.

Lives per-topic at `/topics/[id]/deep-dig`, mirroring deep-read.

## Non-goals (v1)

- **No experiment-reproduction entry point.** v1 seeds from a paper only. The
  "from an experiment reproduction" entry is deferred to a later spec.
- **No writing dig-corpus papers into the topic library or crossdomain corpus.**
  Dug papers live only as session JSON provenance (avoids library pollution). A
  "save to library/KB" action is future work.
- **No multi-hop budget tuning UI.** Budgets are config-level constants, not
  per-run knobs in the UI.
- **No new heavy frontend dependency.** The citation graph is rendered with a
  lightweight inline SVG component, not react-flow / vis-network.
- **No changes to existing pipelines** (`discovery`, `insights`, `brainstorm`,
  `research_plan`, `deep_read`, `chat`). deep-dig reuses them but does not modify
  their behavior.

## Architecture & file map

**New backend files**

| File | Purpose |
|---|---|
| `src/paper_tracker/sources/semantic_scholar.py` | S2 Graph API client: `get_references` / `get_citations` by arXiv id; maps to the standard paper dict + edge metadata; rate-limit & failure tolerant. |
| `src/paper_tracker/deep_dig_agent.py` | Agentic core (mirrors `brainstorm_agent.py`): tool backends, the shared collector, and three entry points: `explore_and_synthesize`, `generate_ideas`, `answer_question`. |
| `src/paper_tracker/deep_dig.py` | Orchestration: seed resolution, phase runners, deterministic fallbacks, milestone/progress callbacks, building `dig_corpus_json` / `graph_json` from the collector. |

**New frontend files**

| File | Purpose |
|---|---|
| `frontend/src/app/topics/[id]/deep-dig/page.tsx` | The deep-dig page (seed picker + sessions + insights + graph + gate + ideas + Q&A). |
| `frontend/src/components/DeepDigGraph.tsx` | Lightweight inline-SVG citation/related graph. |

**Modified files**

| File | Change |
|---|---|
| `src/paper_tracker/registry.py` | Add `deep_dig_sessions` + `deep_dig_messages` tables, migrations, CRUD, and `recover_stale_tasks` coverage. |
| `src/paper_tracker/server.py` | Add request models, two progress dicts, and the deep-dig endpoints. |
| `src/paper_tracker/config.py` | Add `_default_deep_dig()` and include it in `load()`. |
| `config.toml` | Add `[deep_dig]` section. |
| `frontend/src/lib/api.ts` | Add deep-dig API functions + TypeScript types. |
| `frontend/src/app/topics/[id]/layout.tsx` | Add a "Deep Dig" tab to `TABS`. |

**Reused unchanged**: `rag` (`search_papers`, `embed_query`, `embed_texts`,
`cosine_similarity`, `ensure_embeddings`), `summarizer.summarize_papers`,
`crossdomain` (`CROSSDOMAIN_ID`, corpus `Storage`, math-bridge query prompt),
`insights.generate_agentic` (fallback phase 1 + `[Pn]` grounding/audit pattern),
`brainstorm` (idea schema + `_parse_ideas`, fallback phase 2), `llm.call_cli`,
`sources.arxiv` (`fetch_by_id`, `search_by_query`, math-category search),
`sources.pdf` (`fetch_pdf_paper`, `download_and_extract_text`),
`components/IdeaCard.tsx`, `components/MathMarkdown.tsx`.

## Component 1 — Semantic Scholar source

New `src/paper_tracker/sources/semantic_scholar.py`.

### API

```python
S2_BASE = "https://api.semanticscholar.org/graph/v1"
_FIELDS = "paperId,externalIds,title,abstract,year,venue,authors,citationCount,influentialCitationCount"

def get_references(paper_id: str, cfg: dict, *, limit: int = 50) -> list[dict]:
    """Papers that `paper_id` cites (its bibliography).

    GET /paper/{s2_id}/references?fields={_FIELDS}&limit={limit}
    Each row: {"isInfluential", "intents", "contexts", "citedPaper": {...}}.
    Returns standard paper dicts with edge metadata attached (see below).
    Returns [] on any failure (network / 404 / 429-exhausted).
    """

def get_citations(paper_id: str, cfg: dict, *, limit: int = 50) -> list[dict]:
    """Papers that cite `paper_id`. Same shape; rows use "citingPaper".
    Influential citations are surfaced first (sort by isInfluential, then
    influentialCitationCount).
    """
```

`paper_id` accepts any S2-resolvable form so second-hop traversal works for
S2-only papers too: a bare arXiv id / arXiv URL → `ARXIV:{id}`; an `s2:{hash}`
node id → the raw `{hash}`; a `CorpusId:{n}` form passed through. A small
`_s2_id(paper_id)` helper normalizes to the path segment.

### Mapping to the standard paper dict

Each S2 paper object → the repo's standard dict, plus deep-dig edge fields:

```python
{
  "arxiv_id": externalIds.get("ArXiv") or f"s2:{paperId}",
  "paper_id": <same>,
  "source": "semantic_scholar",
  "title", "authors": ", ".join(a["name"]),  "abstract" or "",
  "url": arxiv abs URL if ArXiv else f"https://www.semanticscholar.org/paper/{paperId}",
  "published": str(year or ""),  "venue": venue or "",
  "summary": "", "key_insight": "", "method": "", "contribution": "",
  "math_concepts": [], "cited_works": [],
  "citation_count": citationCount or 0,
  "influential_citation_count": influentialCitationCount or 0,
  "doi": externalIds.get("DOI", ""),
  # edge metadata (used by the collector, not persisted on the paper):
  "_edge_influential": bool(isInfluential),
  "_edge_intents": intents or [],
}
```

### Auth, rate limits, failure

- Optional API key: header `x-api-key` from `cfg["deep_dig"]["s2_api_key"]` or
  env `S2_API_KEY`. Works without a key (lower shared rate limit).
- Use `httpx` with a per-request timeout (~20s).
- On HTTP 429: respect `Retry-After`, exponential backoff, max 3 attempts; if
  still limited, return what was gathered so far (possibly `[]`).
- On 404 / network error / parse error: log at DEBUG, return `[]`. **Never
  raise** — a missing citation path must degrade gracefully, not break the dig.
- Drop rows whose paper has no title (S2 returns nulls for some).

## Component 2 — The deep-dig agent (`deep_dig_agent.py`)

Mirrors `brainstorm_agent.py` exactly in mechanism: sync tool backends run via
`asyncio.to_thread`; `@tool` + `create_sdk_mcp_server` + `ClaudeAgentOptions` +
`query`; `_run_agent` collects `ResultMessage.result` or last text; every public
entry point returns a sentinel on failure so the orchestrator can fall back.

### The collector (the Approach-B enabler)

`ctx["collector"]` is a mutable accumulator shared by all tool backends in a run:

```python
collector = {
  "nodes": {},   # id -> node dict (deduped); first-seen assigns [Pn] index
  "edges": [],   # list of edge dicts
  "order": [],   # ids in first-seen order, to assign P1, P2, ...
}
```

Node dict:

```python
{
  "index": "P7", "id", "source", "title", "authors", "abstract", "url",
  "venue", "year", "citation_count", "influential_citation_count",
  "found_via": "seed" | "references" | "citations" | "search_related" | "search_math",
  "parents": [id, ...],          # which paper(s) this was reached from
  "math_concepts": [...],        # filled lazily if summarized
}
```

Edge dict:

```python
{"from": id, "to": id, "kind": "cites" | "cited_by" | "semantic" | "math",
 "influential": bool, "intents": [...]}
```

A helper `_add_to_collector(collector, paper, found_via, parents, edges)` dedups
by id, assigns the next `[Pn]` index on first sight, and appends edges. Tools
return text to the model that **includes each paper's assigned index** so the
model can cite `[Pn]` (same convention as the insights manifest).

`graph_json` and `dig_corpus_json` are produced from the collector after the run
(see Component 3), capped to `max_papers` nodes for the graph (extra nodes
collapse into per-path counts so the SVG stays readable).

### Tools

Built by `_build_tools(ctx)`; tool names registered in `_TOOL_NAMES`.

| Tool | Args | Backend | Records |
|---|---|---|---|
| `get_references` | `{paper: str}` (index or arxiv id) | `semantic_scholar.get_references` | nodes `found_via="references"`, edges `kind="cites"` (seed→ref) |
| `get_citations` | `{paper: str}` | `semantic_scholar.get_citations` | nodes `found_via="citations"`, edges `kind="cited_by"` |
| `search_related` | `{query: str}` | embedding search: `rag.search_papers(topic_store, query)` + `rag.search_papers(corpus_store, query)`; if a store has no embeddings, fall back to keyword (`brainstorm_agent._search_local`) | nodes `found_via="search_related"`, edges `kind="semantic"` |
| `search_math` | `{query: str}` | math-bridge query (LLM-phrased, math-topic) run two ways: `arxiv.search_by_query(query)` + the crossdomain corpus filtered to `venue` starting `math.`/`stat.` (ranked by embedding/keyword vs query) | nodes `found_via="search_math"`, edges `kind="math"` |
| `read_card` | `{paper: str}` | concept-card/abstract (reuse `_format_card`) | — |
| `read_full_text` | `{paper: str}` | `_fetch_fulltext` (reuse from brainstorm_agent: `pdf.download_and_extract_text`) | — |

`read_card` / `read_full_text` resolve `paper` as a `[Pn]` index first, else as
an arxiv id present in the collector. `get_references` / `get_citations` accept an
index or a raw arxiv id (so the agent can traverse a second hop from any node).

Per-tool result cap = `per_tool_topk` (default 8) papers returned to the model;
the citation tools fetch up to `limit` then return the top-K (influential first
for citations, then by citation_count).

### Entry points

```python
def explore_and_synthesize(seed: dict, *, data_dir, topic_id, cfg,
                           on_event=None) -> dict | None:
    """Phase 1. Seed the collector with `seed` (found_via='seed'), run the
    explorer agent, then read the final message as the insights memo.
    Returns {"insights_md", "dig_corpus", "graph"} or None on failure."""

def generate_ideas(*, insights_md, dig_corpus, steering_notes, focus_ids,
                   data_dir, topic_id, cfg, on_event=None) -> str:
    """Phase 2. Re-seed the collector from `dig_corpus` (so [Pn] indices and
    read_* tools work), put insights + focus papers + steering in the preamble,
    run the ideate agent. Returns ideas JSON raw (for brainstorm._parse_ideas)
    or "" on failure."""

def answer_question(*, question, insights_md, ideas, dig_corpus, prior_messages,
                    data_dir, topic_id, cfg, on_event=None) -> str:
    """Q&A. Re-seed collector from dig_corpus; context = insights + ideas +
    prior turns + question; full toolset so it can dig further. Returns the
    answer markdown or "" on failure."""
```

### Preambles & bounding

Three preambles (explore / ideate / qa) following `_AGENT_PREAMBLE`'s structure:
list indexed seed/dig papers, list tools, give a work order, and **state the
budget explicitly** ("explore roughly `max_papers` papers; prefer influential
citations and high-relevance matches; traverse a second hop only from the most
pivotal papers"). The final message must be ONLY the required output (memo for
explore; JSON array for ideate; markdown answer for qa).

Hard bounds from `cfg["deep_dig"]`: `max_turns` (18), `wall_clock` (900s) via
`asyncio.wait_for`, `max_papers` (50, enforced in `_add_to_collector` — once
reached, citation/search tools stop adding new nodes and say so), `per_tool_topk`
(8). These guarantee termination and cost ceiling even though traversal is
autonomous.

### Grounding

Reuse the insights `[Pn]` grounding idea: after explore, regex the memo for
`[Pn]` labels and drop any that aren't in the collector (hallucinated cites),
matching `insights._audit_and_revise`'s hard rule. (Keep it to validation in v1;
no extra revision call.)

## Component 3 — Orchestration (`deep_dig.py`)

```python
def resolve_seed(topic_id, data_dir, cfg, *, paper_id=None, seed_query=None) -> dict | None:
    """Library lookup (Storage.get_arxiv) → arxiv.fetch_by_id (if arxiv id/URL)
    → pdf.fetch_pdf_paper (if PDF URL). Returns a standard paper dict or None."""

def run_explore(reg, topic, data_dir, cfg, session_id, seed, on_milestone) -> None:
    """Phase 1 runner (called in a thread). Try deep_dig_agent.explore_and_
    synthesize; on None, fall back to _explore_deterministic. Persist
    insights_md / dig_corpus_json / graph_json; set status=insights_ready or
    failed+error_message."""

def run_ideas(reg, topic, data_dir, cfg, session_id, steering_notes, focus_ids, on_milestone) -> None:
    """Phase 2 runner. deep_dig_agent.generate_ideas → brainstorm._parse_ideas;
    on "" fall back to _ideas_deterministic. Persist ideas_json; status=completed."""

def run_qa(reg, topic, data_dir, cfg, session_id, msg_id, question, on_progress) -> None:
    """Q&A runner. deep_dig_agent.answer_question → store assistant message;
    on "" fall back to a one-shot llm.call_cli answer."""
```

### Deterministic fallbacks

- `_explore_deterministic`: `semantic_scholar.get_references` + `get_citations`
  on the seed; `rag.search_papers(seed_text)`; `search_math` from the seed's
  `math_concepts` (run `summarizer.summarize_papers([seed])` first if empty).
  Embed candidates (`rag.embed_texts`), rank by cosine vs the seed
  (`rag.cosine_similarity`), keep top `max_papers` (per-path quotas).
  `summarizer.summarize_papers` on the kept set, then `insights.generate_agentic`
  for the memo. Build `dig_corpus` / `graph` from the gathered nodes/edges.
- `_ideas_deterministic`: brainstorm's one-shot idea prompt via `llm.call_cli`
  over insights + dig-corpus, parsed by `brainstorm._parse_ideas`.
- `_qa_deterministic`: single `llm.call_cli` over the assembled context.

### Building outputs from the collector

`dig_corpus_json` = the node list projected to the insights-manifest shape
(`{index, id, source, title, authors, venue, url, abstract, summary,
key_insight, math_concepts, found_via, influential_citation_count}`), ordered by
`[Pn]`. `graph_json` = `{nodes: [{id, index, title, found_via, citation_count,
influential_citation_count}], edges: [...]}`, node-capped for viz.

## Component 4 — Registry schema

Two new tables (migrated in `Registry.__init__` following the existing
auto-migration pattern), plus CRUD mirroring the deep-read functions.

### `deep_dig_sessions`

| Column | Type | Notes |
|---|---|---|
| `id` | TEXT PK | `dd-{8hex}` |
| `topic_id` | TEXT FK | |
| `seed_paper_id` | TEXT | resolved seed id |
| `seed_title` | TEXT | |
| `status` | TEXT | `exploring` → `insights_ready` → `generating_ideas` → `completed`; or `failed` |
| `language` | TEXT | default `'en'` |
| `started_at` / `finished_at` | TEXT | |
| `insights_md` | TEXT | phase-1 memo |
| `dig_corpus_json` | TEXT | `[Pn]` provenance list |
| `graph_json` | TEXT | nodes+edges for viz |
| `steering_notes` | TEXT | user gate input |
| `ideas_json` | TEXT | phase-2 ideas |
| `error_message` | TEXT | |

### `deep_dig_messages` (Q&A — required in v1)

| Column | Type | Notes |
|---|---|---|
| `id` | TEXT PK | `ddm-{8hex}` |
| `session_id` | TEXT FK → deep_dig_sessions | |
| `topic_id` | TEXT FK | |
| `role` | TEXT | `user` / `assistant` |
| `content` | TEXT | |
| `status` | TEXT | `completed` / `pending` / `generating` / `failed` |
| `created_at` | TEXT | |

### CRUD (mirror deep-read)

`create_deep_dig_session(topic_id, seed_paper_id, seed_title, language='en')`,
`get_deep_dig_session(topic_id, session_id)` (returns session + its messages,
deserializing JSON columns), `list_deep_dig_sessions(topic_id)`,
`update_deep_dig_session(topic_id, session_id, updates)` (allowed fields:
status, finished_at, insights_md, dig_corpus_json, graph_json, steering_notes,
ideas_json, error_message), `delete_deep_dig_session`, and message helpers
`add_deep_dig_message`, `update_deep_dig_message`, `get_deep_dig_message`.
`recover_stale_tasks` marks any `exploring` / `generating_ideas` session and any
`pending` / `generating` message as `failed` on startup (extend the existing
loop).

## Component 5 — Server endpoints

Reuse `_brainstorm_executor` (ThreadPoolExecutor) and add two progress dicts
`_deep_dig_progress` and `_deep_dig_qa_progress`. All long work runs in `_run`
closures submitted to the executor; progress comes from the agent's `on_event`
callback writing into the progress dict.

```python
class DeepDigCreate(BaseModel):
    paper_id: str | None = None      # from library
    seed_query: str | None = None    # arxiv id / URL / PDF URL to fetch
    language: str = "en"

class DeepDigIdeasCreate(BaseModel):
    steering_notes: str = ""
    focus_paper_ids: list[str] = []

class DeepDigMessageCreate(BaseModel):
    content: str
```

| Method & path | Behavior |
|---|---|
| `POST /api/topics/{tid}/deep-dig` → 202 | resolve seed (400 if unresolvable); `create_deep_dig_session`; submit `run_explore`; return `{status, session_id}`. |
| `GET /api/topics/{tid}/deep-dig` | list sessions. |
| `GET /api/topics/{tid}/deep-dig/{sid}` | session detail incl. messages. |
| `GET /api/topics/{tid}/deep-dig/{sid}/progress` | in-memory `_deep_dig_progress` → DB fallback. |
| `POST /api/topics/{tid}/deep-dig/{sid}/generate-ideas` → 202 | require status `insights_ready` or `completed`; store steering_notes; set `generating_ideas`; submit `run_ideas`. Re-runnable (overwrites `ideas_json`). |
| `POST /api/topics/{tid}/deep-dig/{sid}/messages` → 202 | require status ≥ `insights_ready`; add user msg + pending assistant msg; submit `run_qa`; return `{user_msg_id, assistant_msg_id}`. |
| `GET /api/topics/{tid}/deep-dig/{sid}/messages/{mid}/progress` | `{msg_id, status}` (in-memory → DB). |
| `DELETE /api/topics/{tid}/deep-dig/{sid}` | delete session (+ messages). |

## Component 6 — Config

`config.toml`:

```toml
[deep_dig]
s2_api_key = ""      # optional; else env S2_API_KEY; works empty (lower rate limit)
max_turns = 18
wall_clock = 900     # seconds, agent loop budget
max_papers = 50      # total nodes added to the collector per run
per_tool_topk = 8    # papers returned to the model per tool call
```

`config.py`: add `_default_deep_dig()` returning these defaults and merge it in
`load()` (same pattern as `_default_summarizer`). The agent/source read
`cfg.get("deep_dig", {})`.

## Component 7 — Frontend

### `api.ts`

Types `DeepDigSession`, `DeepDigSessionDetail` (extends with `messages`,
`dig_corpus`, `graph`), `DeepDigMessage`, `DeepDigGraph` (`{nodes, edges}`).
Functions following the deep-read set: `startDeepDig(topicId, body)`,
`listDeepDigSessions(topicId)`, `getDeepDigSession(topicId, sid)`,
`getDeepDigProgress(topicId, sid)`, `generateDeepDigIdeas(topicId, sid, body)`,
`sendDeepDigMessage(topicId, sid, content)`,
`getDeepDigMessageProgress(topicId, sid, mid)`,
`deleteDeepDigSession(topicId, sid)`.

### `page.tsx` (mirrors deep-read page structure)

- **Left**: seed picker = library search dropdown (reuse deep-read's debounced
  paper search) **plus** a free-text input for arXiv id / URL / PDF URL
  (→ `seed_query`); language toggle; session list with status badges; delete.
- **Main**, driven by `session.status` (TanStack Query, `refetchInterval` while
  running; progress query while running):
  1. `exploring`: progress (turns / tools used / papers visited from
     `_deep_dig_progress`).
  2. `insights_ready`+: insights memo (`MathMarkdown`); dig-corpus list grouped
     by `found_via`; `DeepDigGraph`; **gate** = steering `<textarea>` + focus
     paper multiselect + "Generate ideas" button.
  3. `generating_ideas`: ideas progress.
  4. `completed`: ideas rendered via `IdeaCard`; "Regenerate ideas" allowed.
  - **Q&A panel** visible from `insights_ready` onward: message list + input;
    poll `getDeepDigMessageProgress` for the pending assistant message, then
    invalidate the session query (same pattern as deep-read Q&A).

### `DeepDigGraph.tsx`

Lightweight inline SVG, **no new dependency**. Seed node centered; other nodes
placed by `found_via` group (references / citations / semantic / math) in
columns or concentric arcs; edges as lines (influential citation edges drawn
bolder); node radius scales with `citation_count`; click → open paper URL /
scroll to its card. Node-capped to `graph_json` (already capped server-side).

### `layout.tsx`

Add `{ label: "Deep Dig", href: "/deep-dig" }` to `TABS` after "Deep Read".

## Error handling

- **S2 down / rate-limited**: tools return `[]`; the agent continues with
  semantic + math paths (degraded, not broken).
- **Agent SDK missing / agent error / timeout**: deterministic fallback per
  phase; logged at WARNING (mirrors brainstorm_agent).
- **Seed unresolvable**: `POST /deep-dig` → 400 with a clear message.
- **Server restart mid-run**: `recover_stale_tasks` flips running sessions and
  pending messages to `failed`.
- **Cost ceiling**: `max_turns` / `wall_clock` / `max_papers` / `per_tool_topk`.
- **Hallucinated citations** in the memo: stripped by the `[Pn]` grounding check.

## Implementation order

1. `sources/semantic_scholar.py` + tests (independent; no other deps).
2. `registry.py` tables + CRUD + `recover_stale_tasks` + tests.
3. `config.py` / `config.toml` `[deep_dig]` defaults.
4. `deep_dig_agent.py` (tools + collector + entry points) + `deep_dig.py`
   (orchestration + deterministic fallbacks) + tests (agent loop mocked).
5. `server.py` endpoints + tests (orchestration mocked).
6. Frontend: `api.ts` → `DeepDigGraph.tsx` → `page.tsx` → `layout.tsx` tab.
7. Manual end-to-end on a real topic + seed paper.

## Testing

Mirror existing suites:

- **`tests/test_semantic_scholar.py`** (like `test_sources.py`): mocked HTTP for
  references/citations — assert parsing, arXiv-id formatting (`ARXIV:` prefix),
  external-id → standard-dict mapping, influential-first ordering, and that 404 /
  429 / network errors return `[]` without raising.
- **`tests/test_deep_dig_agent.py`** (like `test_brainstorm_agent.py`): call tool
  backends directly — assert the collector dedups, assigns sequential `[Pn]`
  indices, records correct edge kinds, caps at `per_tool_topk` / `max_papers`,
  and that `read_full_text` degrades to abstract. Monkeypatch `_run_agent` to
  assert entry points parse/return correctly and return the sentinel on error.
- **`tests/test_deep_dig.py`**: orchestration with the agent monkeypatched —
  assert status transitions (`exploring`→`insights_ready`→`generating_ideas`
  →`completed`), that a `None`/`""` agent result triggers the deterministic
  fallback, and that `dig_corpus_json` / `graph_json` are built from the
  collector.
- **`tests/test_deep_dig_registry.py`** (like `test_deep_read_registry.py`):
  CRUD round-trips, JSON (de)serialization, and `recover_stale_tasks` behavior.
- **`tests/test_deep_dig_server.py`** (like `test_deep_read_server.py`): all
  endpoints with the pipeline mocked — start (incl. 400 on bad seed), gate
  (generate-ideas state guard), messages, progress (in-memory + DB fallback),
  delete.

## Open risks / trade-offs

- **S2 rate limits without a key** are tight; the per-tool cap + backoff keep us
  under them for single-seed runs, and `s2_api_key` lifts the ceiling. If S2 is
  the bottleneck, the deterministic fallback still produces a useful dig from the
  semantic + math paths alone.
- **Agentic non-determinism**: traversal varies run-to-run. Accepted (it is the
  point of Approach B); the collector makes outputs auditable regardless.
- **Graph readability** at the node cap: extra nodes collapse into per-path
  counts; a future "expand" interaction is out of scope.
