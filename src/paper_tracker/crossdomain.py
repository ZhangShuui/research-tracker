"""Curated cross-domain knowledge base: LLM theory-screen + vector retrieval.

A global corpus of authoritative, theory-leaning papers (math / physics /
statistics / foundational CS) used to cross-pollinate ML research. Papers are
screened on ingest (opus) and retrieved by semantic similarity to a selection.
"""

from __future__ import annotations

import json
import logging
import math
import random
import re

from paper_tracker import rag
from paper_tracker.llm import call_cli

log = logging.getLogger(__name__)

# Reserved "topic id" → the corpus lives at data/__crossdomain__/tracker.db,
# reusing Storage's paper CRUD + embedding tables.
CROSSDOMAIN_ID = "__crossdomain__"

# Default arXiv categories for the category importer (theory-leaning fields
# that tend to cross-pollinate ML). The user can override per-import.
DEFAULT_IMPORT_CATEGORIES = [
    "math.OC", "math.ST", "stat.ML", "math.PR", "math.NA",
    "math.IT", "math.DS", "math.FA", "physics.data-an",
]


_SCREEN_BATCH = 15  # screen this many papers per opus call (scales to big imports)


_SCREEN_PROMPT = """\
You curate a library of AUTHORITATIVE, THEORY-LEANING papers from mathematics, \
statistics, physics, and foundational computer science, used to cross-pollinate \
machine-learning research with deep ideas from other fields.

For each paper below decide whether it belongs in such a library.
KEEP (true) if it is a rigorous, foundational or theoretical contribution with \
lasting, transferable value — theorems, frameworks, principles, or mathematical \
tools that could illuminate problems in other fields.
REJECT (false) if it is primarily applied, empirical, incremental, a \
benchmark/dataset/survey, or a narrow engineering result.

Papers:
{papers}

Reply ONLY with a JSON array, one element per paper (match by id):
{{"id": "<id>", "keep": true|false, "domain": "<field, e.g. math.OC, physics, statistics>", "reason": "<one short sentence>"}}"""


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
    """Best-effort parse of a JSON object from an LLM reply."""
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


_CARD_PROMPT = """\
Summarize the MAIN MATHEMATICAL CONTENT of this work for a cross-disciplinary
research library. Capture the key definitions, theorems/results, and techniques
— concisely and self-contained, so a researcher from another field can grasp
what it offers. Focus on the mathematics; skip logistics and pleasantries.

Title: {title}

{body}

Reply ONLY with a JSON object:
{{"summary": "<3-6 sentences of the core mathematical content>", "math_concepts": ["<key concept / theorem / technique>", ...]}}"""


def generate_concept_card(paper: dict, cfg: dict, text: str | None = None) -> dict | None:
    """Generate a concept card — the paper's main mathematical content.

    Returns ``{"summary": str, "math_concepts": [str]}`` or None. ``text`` (e.g.
    a PDF's body text) is preferred over the abstract for depth on long works.
    """
    title = (paper.get("title") or "").strip()
    body = (text or paper.get("abstract") or "").strip()[:12000]
    if not title and not body:
        return None

    raw = call_cli(_CARD_PROMPT.format(title=title, body=body), cfg, timeout=600)
    obj = _parse_json_obj(raw or "")
    summary = (obj.get("summary") or "").strip()
    concepts = obj.get("math_concepts") or []
    if not isinstance(concepts, list):
        concepts = []
    concepts = [str(c).strip() for c in concepts if str(c).strip()][:12]
    if not summary and not concepts:
        return None
    return {"summary": summary, "math_concepts": concepts}


_SECTION_HEADING_RE = re.compile(
    r"^[ \t]*("
    r"(?:chapter|section|part|appendix)\s+[0-9ivxlcdm]+\b[^\n]{0,80}"   # "Chapter 3 …", "Part IV …"
    r"|[0-9]{1,2}(?:\.[0-9]{1,3}){0,2}\.?[ \t]+[A-Za-z][^\n]{2,80}"      # "3 Introduction", "3.2 Foo"
    r")[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)


def _cap_pieces(pieces: list[dict], max_pieces: int) -> list[dict]:
    """Merge adjacent pieces so the count fits ``max_pieces`` (keeps coverage)."""
    if len(pieces) <= max_pieces:
        return pieces
    group = math.ceil(len(pieces) / max_pieces)
    merged = []
    for i in range(0, len(pieces), group):
        grp = pieces[i : i + group]
        merged.append({"title": grp[0]["title"], "text": "\n\n".join(g["text"] for g in grp)})
    return merged


def split_into_sections(
    text: str,
    *,
    max_pieces: int = 20,
    min_section_chars: int = 400,
    chunk_chars: int = 9000,
    overlap: int = 400,
) -> list[dict]:
    """Split a long document into pieces for per-section concept cards.

    C (primary): split at detected section/chapter headings — each piece keeps
    its full local context. B (fallback): if no usable structure is found,
    fixed-size overlapping chunks. Returns ``[{title, text}]`` capped at
    ``max_pieces`` (adjacent pieces merged to fit). A short doc yields 1 piece.
    """
    text = (text or "").strip()
    if not text:
        return []

    # --- C: heading-based ---
    matches = list(_SECTION_HEADING_RE.finditer(text))
    sections: list[dict] = []
    if len(matches) >= 2:
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            if len(body) >= min_section_chars:
                sections.append({"title": " ".join(m.group(1).split())[:120], "text": body})
    if len(sections) >= 2:
        return _cap_pieces(sections, max_pieces)

    # --- B: fixed overlapping chunks ---
    chunks: list[dict] = []
    step = max(1, chunk_chars - overlap)
    i, n = 0, 1
    while i < len(text):
        chunks.append({"title": f"Part {n}", "text": text[i : i + chunk_chars]})
        n += 1
        i += step
    return _cap_pieces(chunks, max_pieces)


def screen_papers(papers: list[dict], cfg: dict) -> list[dict]:
    """Screen papers for the corpus. Returns one verdict per input paper:
    ``{arxiv_id, keep, domain, reason}`` (aligned to ``papers``).

    If the LLM is unavailable (no parseable output), falls back to keep-all
    (advisory) so ingestion isn't blocked by a transient outage — the curator
    can still remove papers manually.
    """
    if not papers:
        return []

    out: list[dict] = []
    for start in range(0, len(papers), _SCREEN_BATCH):
        batch = papers[start : start + _SCREEN_BATCH]
        lines = [
            f"[{p.get('paper_id') or p['arxiv_id']}] {p.get('title', '')}: "
            f"{(p.get('abstract', '') or '')[:400]}"
            for p in batch
        ]
        raw = call_cli(_SCREEN_PROMPT.format(papers="\n\n".join(lines)), cfg, timeout=600)
        verdicts = _parse_json_array(raw or "")

        if not verdicts:
            log.warning("Cross-domain screen unavailable for a batch — keeping all (advisory)")
            out.extend({"arxiv_id": p["arxiv_id"], "keep": True, "domain": "",
                        "reason": "screen unavailable"} for p in batch)
            continue

        lookup = {str(v.get("id", "")): v for v in verdicts if isinstance(v, dict)}
        for p in batch:
            pid = str(p.get("paper_id") or p["arxiv_id"])
            v = lookup.get(pid) or lookup.get(str(p["arxiv_id"])) or {}
            out.append({
                "arxiv_id": p["arxiv_id"],
                "keep": bool(v.get("keep", False)),
                "domain": (v.get("domain") or "").strip(),
                "reason": (v.get("reason") or "").strip(),
            })
    return out


def retrieve(selected_papers: list[dict], corpus_store, k: int = 12, *,
             random_k: int = 3, rng=None) -> list[dict]:
    """Vector-search the corpus for papers semantically related to the selection.

    Embeds each selected paper, mean-pools to a query vector, and ranks corpus
    papers by cosine similarity. Returns up to ``k`` candidate dicts (arxiv_id,
    title, authors, abstract, url, published, domain, rationale, source='kb',
    score, random). Returns [] if the corpus is empty or embeddings unavailable.

    To inject serendipity, ``random_k`` of the ``k`` slots are filled with RANDOM
    papers drawn from the lower-ranked tail (NOT the obvious top matches), each
    flagged ``"random": True`` — pure top-K returns the most-similar (often most-
    obvious) papers, so a few random math picks add variance and surface non-
    obvious bridges. Set ``random_k=0`` for pure top-K; pass ``rng`` (a
    ``random.Random``) for deterministic tests.
    """
    if not selected_papers:
        return []
    corpus_embs = corpus_store.get_all_embeddings()  # [(arxiv_id, vec)]
    if not corpus_embs:
        return []

    texts = [rag._paper_text(p) for p in selected_papers[:40]]
    try:
        vecs = [v for v in rag.embed_texts(texts) if v]
    except Exception as e:
        log.warning("Cross-domain retrieve: embedding failed: %s", e)
        return []
    if not vecs:
        return []

    dim = len(vecs[0])
    qvec = [sum(v[i] for v in vecs) / len(vecs) for i in range(dim)]

    selected_ids = {p.get("arxiv_id") for p in selected_papers}
    selected_ids |= {p.get("paper_id") for p in selected_papers}

    scored: list[tuple[float, str]] = []
    for aid, emb in corpus_embs:
        if not emb or aid in selected_ids:
            continue
        scored.append((rag.cosine_similarity(qvec, emb), aid))
    scored.sort(key=lambda t: t[0], reverse=True)

    # Reserve a few of the k slots for RANDOM tail papers (serendipity): top-K
    # alone returns the closest/most-obvious matches; random picks from below the
    # top cut inject variance and surface non-obvious cross-domain bridges.
    n_random = max(0, min(random_k, k))
    n_top = k - n_random
    chosen: list[tuple[float, str, bool]] = [(s, a, False) for s, a in scored[:n_top]]
    tail = scored[n_top:]
    if n_random and tail:
        sampler = rng or random
        for s, a in sampler.sample(tail, min(n_random, len(tail))):
            chosen.append((s, a, True))

    candidates = []
    for score, aid, is_random in chosen:
        p = corpus_store.get_arxiv(aid)
        if not p:
            continue
        candidates.append({
            "arxiv_id": aid,
            "title": p.get("title", ""),
            "authors": p.get("authors", ""),
            "abstract": p.get("abstract", ""),
            "url": p.get("url", ""),
            "published": p.get("published", ""),
            "domain": p.get("venue", ""),          # primary category stored in venue
            # prefer the concept card's math summary; fall back to screen reason
            "rationale": p.get("summary") or p.get("key_insight", ""),
            "source": "kb",
            "random": is_random,
            "score": round(float(score), 3),
        })
    return candidates
