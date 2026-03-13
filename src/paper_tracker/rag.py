"""RAG module: vector embeddings + semantic search over the paper library.

Uses OpenAI-compatible embedding API for vectorization and cosine
similarity for retrieval.  Embeddings are cached in the per-topic
tracker.db so they only need to be computed once per paper.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Sequence

from openai import OpenAI

# Auto-load .env from project root if present
_ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

from paper_tracker.storage import Storage

log = logging.getLogger(__name__)

_EMBED_MODEL = "text-embedding-3-large"
_EMBED_BATCH_SIZE = 64  # OpenAI supports up to 2048 per call; 64 is safe & fast

# ---------------------------------------------------------------------------
# OpenAI client (lazy singleton)
# ---------------------------------------------------------------------------

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is required for RAG embeddings. "
                "Set it in .env or export it before starting the server."
            )
        _client = OpenAI(api_key=api_key, base_url=base_url)
    return _client


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


def _paper_text(paper: dict) -> str:
    """Build a single text string from a paper for embedding."""
    parts = [paper.get("title", "")]
    if paper.get("abstract"):
        parts.append(paper["abstract"][:500])
    if paper.get("key_insight"):
        parts.append(paper["key_insight"])
    if paper.get("method"):
        parts.append(paper["method"])
    if paper.get("contribution"):
        parts.append(paper["contribution"])
    return "\n".join(p for p in parts if p)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts via the OpenAI-compatible API.

    Returns list of embedding vectors (same order as input).
    """
    client = _get_client()
    all_embeddings: list[list[float]] = [[] for _ in texts]

    for start in range(0, len(texts), _EMBED_BATCH_SIZE):
        batch = texts[start : start + _EMBED_BATCH_SIZE]
        resp = client.embeddings.create(model=_EMBED_MODEL, input=batch)
        for item in resp.data:
            all_embeddings[start + item.index] = item.embedding

    return all_embeddings


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([query])[0]


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------


def ensure_embeddings(store: Storage, on_progress: callable = None) -> int:
    """Embed all papers that don't have embeddings yet.

    Args:
        store: An open Storage instance.
        on_progress: Optional callback(embedded_count, total_count).

    Returns:
        Number of newly embedded papers.
    """
    papers = store.get_papers_without_embeddings(limit=2000)
    if not papers:
        return 0

    total = len(papers)
    embedded = 0

    for start in range(0, total, _EMBED_BATCH_SIZE):
        batch = papers[start : start + _EMBED_BATCH_SIZE]
        texts = [_paper_text(p) for p in batch]

        try:
            vectors = embed_texts(texts)
        except Exception:
            log.exception("Embedding batch failed at offset %d", start)
            break

        items = [
            (p["arxiv_id"], vec, _EMBED_MODEL)
            for p, vec in zip(batch, vectors)
        ]
        store.save_embeddings_batch(items)
        embedded += len(items)

        if on_progress:
            on_progress(embedded, total)

    log.info("Embedded %d / %d new papers", embedded, total)
    return embedded


# ---------------------------------------------------------------------------
# Semantic search
# ---------------------------------------------------------------------------


def search_papers(
    store: Storage,
    query: str,
    max_results: int = 10,
    min_score: float = 0.25,
) -> list[tuple[dict, float]]:
    """Find papers most relevant to *query* via cosine similarity.

    Returns list of (paper_dict, score) sorted by descending score.
    Falls back to empty list if no embeddings exist.
    """
    all_embs = store.get_all_embeddings()
    if not all_embs:
        return []

    q_vec = embed_query(query)

    # Score every paper
    scored: list[tuple[str, float]] = []
    for arxiv_id, emb in all_embs:
        score = cosine_similarity(q_vec, emb)
        if score >= min_score:
            scored.append((arxiv_id, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:max_results]

    # Fetch full paper dicts
    results = []
    for arxiv_id, score in top:
        paper = store.get_arxiv(arxiv_id)
        if paper:
            results.append((paper, score))

    return results
