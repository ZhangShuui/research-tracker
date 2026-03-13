"""Chat module: conversational Q&A grounded in a topic's paper library.

Uses RAG (vector embeddings + semantic search) for paper retrieval,
with keyword-based fallback when embeddings are unavailable.
"""

from __future__ import annotations

import logging
import re

from paper_tracker.llm import call_cli
from paper_tracker.storage import Storage

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paper search — RAG primary, keyword fallback
# ---------------------------------------------------------------------------


def _search_papers_rag(
    store: Storage,
    user_message: str,
    prior_messages: list[dict] | None = None,
    max_papers: int = 10,
) -> list[dict]:
    """Semantic search via RAG embeddings.

    Lazily builds the embedding index if needed (first call may be slower).
    Returns list of paper dicts sorted by relevance.
    """
    from paper_tracker.rag import ensure_embeddings, search_papers

    # Build index for any new papers (incremental — only missing ones)
    try:
        newly_embedded = ensure_embeddings(store)
        if newly_embedded:
            log.info("Embedded %d new papers for RAG index", newly_embedded)
    except Exception:
        log.exception("Failed to build embedding index, falling back to keyword search")
        return _search_papers_keyword(store, user_message, prior_messages, max_papers)

    # Combine user message with recent conversation context for query
    query = user_message
    if prior_messages:
        recent_user = [
            m["content"] for m in prior_messages[-3:]
            if m.get("role") == "user" and m.get("content")
        ]
        if recent_user:
            query = " ".join(recent_user[-2:]) + " " + user_message

    results = search_papers(store, query, max_results=max_papers)
    if not results:
        log.info("RAG returned no results, falling back to keyword search")
        return _search_papers_keyword(store, user_message, prior_messages, max_papers)

    return [paper for paper, _score in results]


def _search_papers_keyword(
    store: Storage,
    user_message: str,
    prior_messages: list[dict] | None = None,
    max_papers: int = 10,
) -> list[dict]:
    """Keyword-based fallback search (original logic)."""
    search_text = user_message
    if prior_messages:
        recent = [m["content"] for m in prior_messages[-3:] if m.get("content")]
        if recent:
            search_text = " ".join(recent) + " " + user_message

    keywords = _extract_keywords(search_text)
    bigrams = _extract_bigrams(search_text)

    seen_ids: set[str] = set()
    papers: list[dict] = []

    for phrase in bigrams:
        results, _ = store.get_all_arxiv(search=phrase, limit=5)
        for p in results:
            pid = p.get("arxiv_id") or p.get("paper_id", "")
            if pid and pid not in seen_ids:
                seen_ids.add(pid)
                papers.append(p)

    for kw in keywords:
        if len(papers) >= max_papers:
            break
        results, _ = store.get_all_arxiv(search=kw, limit=5)
        for p in results:
            pid = p.get("arxiv_id") or p.get("paper_id", "")
            if pid and pid not in seen_ids:
                seen_ids.add(pid)
                papers.append(p)

    if len(papers) < 3:
        results, _ = store.get_all_arxiv(limit=10)
        for p in results:
            pid = p.get("arxiv_id") or p.get("paper_id", "")
            if pid and pid not in seen_ids:
                seen_ids.add(pid)
                papers.append(p)
            if len(papers) >= max_papers:
                break

    return papers[:max_papers]


# ---------------------------------------------------------------------------
# Keyword extraction helpers (used by fallback)
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset(
    "a an the is are was were be been being have has had do does did will would "
    "shall should may might can could of in to for on with at by from as into "
    "through during before after above below between under about up down out "
    "off over again further then once here there when where why how all each "
    "every both few more most other some such no nor not only own same so than "
    "too very just also back even still already much now well way also and but "
    "or if while because although however this that these those it its i me my "
    "we our you your he him his she her they them their what which who whom "
    "how many any don doesn didn wouldn couldn shouldn".split()
)


def _extract_keywords(text: str, max_kw: int = 6) -> list[str]:
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    unique = list(dict.fromkeys(w for w in words if w not in _STOP_WORDS))
    unique.sort(key=len, reverse=True)
    return unique[:max_kw]


def _extract_bigrams(text: str, max_bg: int = 3) -> list[str]:
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    bigrams = []
    for i in range(len(words) - 1):
        if words[i] not in _STOP_WORDS and words[i + 1] not in _STOP_WORDS:
            bigrams.append(f"{words[i]} {words[i + 1]}")
    return list(dict.fromkeys(bigrams))[:max_bg]


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------


def _format_paper_context(papers: list[dict]) -> str:
    """Format papers as numbered references for the LLM prompt."""
    if not papers:
        return "(No papers available)"
    lines = []
    for i, p in enumerate(papers, 1):
        parts = [f"[P{i}] **{p.get('title', 'Untitled')}**"]
        if p.get("key_insight"):
            parts.append(f"  Key insight: {p['key_insight']}")
        if p.get("method"):
            parts.append(f"  Method: {p['method']}")
        if p.get("contribution"):
            parts.append(f"  Contribution: {p['contribution']}")
        if p.get("venue"):
            parts.append(f"  Venue: {p['venue']}")
        lines.append("\n".join(parts))
    return "\n\n".join(lines)


def _format_conversation_history(messages: list[dict], max_messages: int = 20) -> str:
    """Format prior messages as conversation history."""
    if not messages:
        return "(Start of conversation)"
    recent = messages[-max_messages:]
    lines = []
    for m in recent:
        role = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"**{role}:** {m['content']}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_CHAT_SYSTEM_PROMPT = """You are a research discussion partner for the topic "{topic_name}".
You have access to a curated library of papers on this topic. Use them to support your analysis.

## Available Papers

{paper_context}

## Conversation History

{conversation_history}

## Instructions

- Reference papers using their citation tags like [P1], [P2], etc.
- Be specific: mention architectures, loss functions, datasets, metrics, and experimental results when relevant.
- Clearly distinguish between claims supported by the papers vs. your own reasoning or speculation.
- When comparing methods, create structured comparisons (tables, pro/con lists).
- If the user asks about something not covered by the available papers, say so honestly and reason from first principles.
- Be concise but thorough. Prioritize actionable insights.
- Use LaTeX for mathematical notation when needed.
- Respond in the same language as the user's message."""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_chat_response(
    topic_id: str,
    topic_name: str,
    data_dir: str,
    cfg: dict,
    user_message: str,
    prior_messages: list[dict] | None = None,
) -> dict:
    """Generate a chat response grounded in the topic's paper library.

    Uses RAG (semantic search) to find relevant papers, with keyword
    fallback if embeddings are unavailable.

    Returns:
        {"content": str, "cited_papers": [{"arxiv_id": str, "title": str}, ...]}
    """
    store = Storage(data_dir, topic_id)
    try:
        papers = _search_papers_rag(store, user_message, prior_messages)
    finally:
        store.close()

    paper_context = _format_paper_context(papers)
    conversation_history = _format_conversation_history(prior_messages or [])

    system = _CHAT_SYSTEM_PROMPT.format(
        topic_name=topic_name,
        paper_context=paper_context,
        conversation_history=conversation_history,
    )

    prompt = f"{system}\n\n## User Message\n\n{user_message}"

    raw = call_cli(prompt, cfg, model="opus", timeout=120)
    if not raw:
        return {"content": "I'm sorry, I wasn't able to generate a response. Please try again.", "cited_papers": []}

    # Extract cited paper references [P1], [P2], ...
    cited_indices = set(int(m) for m in re.findall(r"\[P(\d+)\]", raw))
    cited_papers = []
    for idx in sorted(cited_indices):
        if 1 <= idx <= len(papers):
            p = papers[idx - 1]
            cited_papers.append({
                "arxiv_id": p.get("arxiv_id") or p.get("paper_id", ""),
                "title": p.get("title", ""),
            })

    return {"content": raw, "cited_papers": cited_papers}
