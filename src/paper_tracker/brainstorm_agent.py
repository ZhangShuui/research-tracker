"""Agentic, retrieval-augmented idea generation for brainstorm.

Runs idea generation as a Claude Agent SDK tool-use loop: the model can look up
the ORIGINAL text of the papers that fed the insights memo (by their [Pn] index)
and actively search the local library + KB corpus + arXiv for more, before
proposing ideas. The SDK drives the local ``claude`` CLI under the hood, so it
reuses the existing subscription auth (no API key needed).

This is a drop-in for brainstorm's one-shot idea-generation call: ``generate_ideas``
returns the final raw text (the ideas JSON, for ``_parse_ideas``). On ANY failure
(SDK missing, egress down, timeout, agent error) it returns "" so the caller falls
back to plain one-shot generation — no regression.
"""

from __future__ import annotations

import asyncio
import logging
import re

from paper_tracker.storage import Storage
from paper_tracker.sources import arxiv

log = logging.getLogger(__name__)

_MODEL = "opus"            # brainstorm is important — prefer opus
_MAX_TURNS = 14           # cap the agent's think/act loop
_AGENT_TIMEOUT = 600      # overall wall-clock budget for the loop (seconds)
_SEARCH_RESULTS = 6       # results per search tool call
_FULLTEXT_CHARS = 8000    # cap full-text returned to the model


# ---------------------------------------------------------------------------
# Tool backends (sync; run off the event loop via asyncio.to_thread)
# ---------------------------------------------------------------------------

def _format_card(p: dict) -> str:
    parts = [f"[{p.get('index', '?')}] {p.get('title', '')}"]
    if p.get("venue"):
        parts.append(f"Venue/domain: {p['venue']}")
    if p.get("authors"):
        parts.append(f"Authors: {p['authors']}")
    if p.get("summary"):
        parts.append(f"Concept card: {p['summary']}")
    if p.get("key_insight"):
        parts.append(f"Key insight: {p['key_insight']}")
    if p.get("math_concepts"):
        parts.append(f"Concepts: {', '.join(p['math_concepts'])}")
    if p.get("abstract"):
        parts.append(f"Abstract: {p['abstract'][:1500]}")
    if p.get("url"):
        parts.append(f"URL: {p['url']}")
    return "\n".join(parts)


def _fetch_fulltext(p: dict) -> str:
    """Best-effort original full text (first pages); falls back to abstract."""
    from paper_tracker.sources import pdf as pdf_source

    pid = (p.get("id") or "").strip()
    url = (p.get("url") or "").strip()
    src = (p.get("source") or "").lower()
    try:
        if src == "arxiv" or re.match(r"^\d{4}\.\d{4,5}", pid):
            txt = pdf_source.download_and_extract_text(f"https://arxiv.org/pdf/{pid}", max_pages=20)
            if txt:
                return txt[:_FULLTEXT_CHARS]
        if url:
            txt = pdf_source.download_and_extract_text(url, max_pages=20)
            if txt:
                return txt[:_FULLTEXT_CHARS]
    except Exception as e:  # network / parse — degrade to abstract
        log.debug("full-text fetch failed for %s: %s", pid or url, e)
    return (p.get("abstract") or p.get("summary") or "(full text unavailable)")[:_FULLTEXT_CHARS]


def _rank_keyword(papers: list[dict], terms: list[str], source_label: str) -> list[tuple]:
    scored = []
    for p in papers:
        hay = (f"{p.get('title', '')} {p.get('abstract', '') or p.get('summary', '')}").lower()
        score = sum(1 for t in terms if t in hay)
        if score:
            scored.append((score, p, source_label))
    return scored


def _search_local(query: str, data_dir: str, topic_id: str) -> str:
    terms = [t for t in re.split(r"\W+", (query or "").lower()) if len(t) > 2]
    if not terms:
        return "Provide a more specific query."
    rows: list[tuple] = []
    try:
        store = Storage(data_dir, topic_id)
        try:
            papers, _ = store.get_all_arxiv(limit=200, offset=0)
        finally:
            store.close()
        rows += _rank_keyword(papers, terms, "library")
    except Exception as e:
        log.debug("search_local (library) failed: %s", e)
    try:
        from paper_tracker import crossdomain
        corpus = Storage(data_dir, crossdomain.CROSSDOMAIN_ID)
        try:
            cpapers, _ = corpus.get_all_arxiv(limit=500, offset=0)
        finally:
            corpus.close()
        rows += _rank_keyword(cpapers, terms, "knowledge_base")
    except Exception as e:
        log.debug("search_local (KB) failed: %s", e)

    rows.sort(key=lambda r: r[0], reverse=True)
    if not rows:
        return "No local matches."
    out = []
    for _score, p, src in rows[:_SEARCH_RESULTS]:
        pid = p.get("arxiv_id") or p.get("paper_id") or ""
        snippet = (p.get("summary") or p.get("key_insight") or p.get("abstract") or "")[:240]
        out.append(f"[{src}] {p.get('title', '')} (id={pid})\n  {snippet}")
    return "\n\n".join(out)


def _search_arxiv(query: str) -> str:
    try:
        hits = arxiv.search_by_query(query, max_results=_SEARCH_RESULTS)
    except Exception as e:
        return f"arXiv search failed: {e}"
    if not hits:
        return "No arXiv results."
    out = []
    for h in hits:
        out.append(f"{h.get('title', '')} (arXiv:{h.get('arxiv_id', '')})\n  {(h.get('abstract', '') or '')[:240]}")
    return "\n\n".join(out)


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

_TOOL_NAMES = ("read_paper", "read_full_text", "search_local", "search_arxiv")


def _build_tools(ctx: dict):
    from claude_agent_sdk import tool

    @tool("read_paper",
          "Read the concept-card summary + abstract of a cited insights paper by its [Pn] index (e.g. index='P3').",
          {"index": str})
    async def read_paper(args):
        idx = str(args.get("index", "")).strip().strip("[]")
        p = ctx["by_index"].get(idx)
        text = _format_card(p) if p else (
            f"No paper with index {idx!r}. Available: {', '.join(ctx['by_index']) or '(none)'}")
        return {"content": [{"type": "text", "text": text}]}

    @tool("read_full_text",
          "Fetch the ORIGINAL full text (first pages) of a cited insights paper by its [Pn] index, on demand.",
          {"index": str})
    async def read_full_text(args):
        idx = str(args.get("index", "")).strip().strip("[]")
        p = ctx["by_index"].get(idx)
        if not p:
            return {"content": [{"type": "text", "text": f"No paper with index {idx!r}."}]}
        text = await asyncio.to_thread(_fetch_fulltext, p)
        return {"content": [{"type": "text", "text": text}]}

    @tool("search_local",
          "Search the local topic library AND the curated knowledge base (math/theory corpus) by keyword.",
          {"query": str})
    async def search_local(args):
        text = await asyncio.to_thread(_search_local, str(args.get("query", "")),
                                       ctx["data_dir"], ctx["topic_id"])
        return {"content": [{"type": "text", "text": text}]}

    @tool("search_arxiv",
          "Search arXiv by keyword for additional related or cross-domain papers.",
          {"query": str})
    async def search_arxiv(args):
        text = await asyncio.to_thread(_search_arxiv, str(args.get("query", "")))
        return {"content": [{"type": "text", "text": text}]}

    return [read_paper, read_full_text, search_local, search_arxiv]


async def _run_agent(prompt: str, tools, model: str, max_turns: int) -> str:
    from claude_agent_sdk import query, create_sdk_mcp_server, ClaudeAgentOptions

    server = create_sdk_mcp_server(name="papers", version="1.0.0", tools=tools)
    options = ClaudeAgentOptions(
        mcp_servers={"papers": server},
        allowed_tools=[f"mcp__papers__{n}" for n in _TOOL_NAMES],
        model=model,
        max_turns=max_turns,
    )
    result_text, last_text = "", ""
    async for msg in query(prompt=prompt, options=options):
        if type(msg).__name__ == "ResultMessage":
            r = getattr(msg, "result", None)
            if r:
                result_text = str(r)
        content = getattr(msg, "content", None)
        if isinstance(content, list):
            chunk = "".join(getattr(b, "text", "") or "" for b in content)
            if chunk.strip():
                last_text = chunk
    return result_text or last_text


_AGENT_PREAMBLE = """You are generating research ideas, and you have TOOLS to ground them in REAL \
papers instead of relying on memory.

Indexed papers from the insights analysis (use read_paper / read_full_text with these indices):
{index_list}

Available tools:
- read_paper(index): concept-card summary + abstract for an indexed paper above (e.g. index="P3").
- read_full_text(index): that paper's ORIGINAL full text (first pages), fetched on demand.
- search_local(query): search the local topic library + curated knowledge base (math/theory corpus).
- search_arxiv(query): search arXiv for more related or cross-domain work.

How to work:
1. FIRST read the originals of the most relevant indexed papers (read_paper; read_full_text when you need depth).
2. THEN use search_local and search_arxiv to pull in additional related or cross-domain papers that could spark or sharpen ideas.
3. ONLY THEN write your final answer.

Your FINAL message must be ONLY the JSON required by the task below — no tool calls, no commentary, no markdown fences.

================= TASK =================
{task}
"""


def generate_ideas(
    task_prompt: str,
    *,
    manifest: list[dict] | None,
    data_dir: str,
    topic_id: str,
    cfg: dict,
    max_turns: int = _MAX_TURNS,
) -> str:
    """Agentic, retrieval-augmented idea generation.

    Returns the agent's final raw text (the ideas JSON, for ``_parse_ideas``), or
    "" on ANY failure so the caller can fall back to one-shot generation.
    """
    try:
        manifest = manifest or []
        by_index = {str(p.get("index")): p for p in manifest if p.get("index")}
        ctx = {"by_index": by_index, "data_dir": data_dir, "topic_id": topic_id}
        tools = _build_tools(ctx)
        if by_index:
            index_list = "\n".join(
                f"- {p['index']}: {p.get('title', '')}" for p in manifest if p.get("index"))
        else:
            index_list = "(no indexed insights papers — rely on search_local / search_arxiv)"
        prompt = _AGENT_PREAMBLE.format(index_list=index_list, task=task_prompt)
        raw = asyncio.run(asyncio.wait_for(
            _run_agent(prompt, tools, _MODEL, max_turns), timeout=_AGENT_TIMEOUT))
        return (raw or "").strip()
    except Exception as e:
        log.warning("Agentic idea generation failed (%s) — caller will fall back to one-shot", e)
        return ""


_REVIEW_AGENT_PREAMBLE = """You are reviewing research ideas, and you have TOOLS to \
investigate the real PRIOR WORK instead of relying on memory.

Available tools:
- search_arxiv(query): search arXiv for the closest prior work on an idea.
- search_local(query): search the local topic library + curated knowledge base.
(read_paper / read_full_text exist too but are usually not needed for review.)

How to work:
1. For each idea, FIRST search (arXiv + local) for the CLOSEST prior work — does it \
already exist? does prior work break an assumption the idea relies on?
2. THEN judge each idea and CHALLENGE it against what you actually found.

Your FINAL message must be ONLY the JSON array required by the task below — no tool \
calls, no commentary, no markdown fences.

================= TASK =================
{task}
"""


def review_ideas(
    task_prompt: str,
    *,
    data_dir: str,
    topic_id: str,
    cfg: dict,
    max_turns: int = 12,
) -> str:
    """Agentic idea review: the judge searches real prior work before scoring.

    Returns the agent's final raw text (the review JSON, for the caller to parse),
    or "" on ANY failure so the caller can fall back to a one-shot review.
    """
    try:
        ctx = {"by_index": {}, "data_dir": data_dir, "topic_id": topic_id}
        tools = _build_tools(ctx)
        prompt = _REVIEW_AGENT_PREAMBLE.format(task=task_prompt)
        raw = asyncio.run(asyncio.wait_for(
            _run_agent(prompt, tools, _MODEL, max_turns), timeout=_AGENT_TIMEOUT))
        return (raw or "").strip()
    except Exception as e:
        log.warning("Agentic review failed (%s) — caller will fall back to one-shot", e)
        return ""
