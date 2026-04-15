"""Deep Read: LLM-powered section-by-section paper analysis + Q&A."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Callable

from paper_tracker.llm import call_cli
from paper_tracker.storage import Storage

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Repo detection, cloning & local context building
# ---------------------------------------------------------------------------

_GITHUB_URL_RE = re.compile(
    r"https?://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)"
)

# Files worth reading for a code review (order = priority)
_KEY_FILES = [
    "README.md", "readme.md", "README.rst",
    "setup.py", "setup.cfg", "pyproject.toml",
    "requirements.txt",
    "Makefile", "Dockerfile",
    "main.py", "train.py", "run.py", "app.py",
    "src/main.py", "src/train.py",
]

# Extensions considered source code
_CODE_EXTS = {".py", ".js", ".ts", ".go", ".rs", ".java", ".c", ".cpp", ".h", ".sh", ".yaml", ".yml", ".toml", ".cfg"}


def _extract_repo_urls(paper: dict) -> list[str]:
    """Extract unique GitHub repo URLs from all text fields of a paper."""
    text_fields = ["abstract", "summary", "key_insight", "method",
                   "contribution", "title", "url"]
    combined = ""
    for f in text_fields:
        val = paper.get(f, "")
        if val:
            combined += " " + str(val)
    # Also check the paper_id itself (some store URLs)
    combined += " " + paper.get("paper_id", "")

    matches = _GITHUB_URL_RE.findall(combined)
    # De-dup preserving order, strip trailing punctuation and .git
    seen = set()
    urls = []
    for m in matches:
        repo = m.rstrip("/").rstrip(".").removesuffix(".git")
        if repo and "/" in repo and repo not in seen:
            seen.add(repo)
            urls.append(f"https://github.com/{repo}")
    return urls


def _clone_repo(repo_url: str, dest_dir: str, timeout: int = 300) -> str | None:
    """Shallow-clone a repo to dest_dir/{owner}/{repo}. Returns local path or None."""
    match = _GITHUB_URL_RE.match(repo_url)
    if not match:
        return None
    owner_repo = match.group(1).rstrip("/").removesuffix(".git")
    local_path = os.path.join(dest_dir, owner_repo)

    # Already cloned — just pull
    if os.path.isdir(os.path.join(local_path, ".git")):
        log.info("Repo already cloned at %s, pulling latest", local_path)
        try:
            subprocess.run(
                ["git", "-C", local_path, "pull", "--ff-only"],
                capture_output=True, timeout=60,
            )
        except Exception as e:
            log.warning("git pull failed for %s: %s", local_path, e)
        return local_path

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    log.info("Cloning %s → %s", repo_url, local_path)
    try:
        subprocess.run(
            ["git", "clone", "--depth=1", f"{repo_url}.git", local_path],
            capture_output=True, timeout=timeout, check=True,
        )
        return local_path
    except subprocess.CalledProcessError as e:
        log.error("git clone failed for %s: %s\nstderr: %s", repo_url, e, e.stderr.decode(errors="replace"))
        return None
    except subprocess.TimeoutExpired:
        log.error("git clone timed out for %s", repo_url)
        return None


def _safe_read(root: Path, rel_path: str, max_bytes: int) -> str | None:
    """Read a file only if it resolves inside *root* (symlink-safe)."""
    fpath = root / rel_path
    try:
        resolved = fpath.resolve()
        if not str(resolved).startswith(str(root.resolve())):
            log.warning("Skipping symlink escape: %s → %s", rel_path, resolved)
            return None
        return fpath.read_text(errors="replace")[:max_bytes]
    except Exception:
        return None


def _build_repo_context(local_path: str, max_files: int = 10, max_file_bytes: int = 15000) -> str:
    """Build a text context block from a locally cloned repo for LLM consumption."""
    root = Path(local_path)
    owner_repo = "/".join(root.parts[-2:])
    parts: list[str] = [f"## Repository: {owner_repo}\nLocal path: {local_path}\n"]

    # 1) File tree (followlinks=False to avoid symlink loops)
    all_files: list[str] = []
    for dirpath, dirnames, filenames in os.walk(local_path, followlinks=False):
        # Skip .git and common non-source dirs
        dirnames[:] = [d for d in dirnames if d not in {".git", "node_modules", "__pycache__", ".tox", "venv", ".venv", "dist", "build", ".eggs"}]
        for fn in filenames:
            rel = os.path.relpath(os.path.join(dirpath, fn), local_path)
            all_files.append(rel)

    parts.append(f"### File tree ({len(all_files)} files)")
    for f in sorted(all_files)[:200]:
        parts.append(f"  {f}")
    if len(all_files) > 200:
        parts.append(f"  ... and {len(all_files) - 200} more files")
    parts.append("")

    # 2) Read key files first
    fetched = 0
    lower_files = {f.lower(): f for f in all_files}

    for candidate in _KEY_FILES:
        if fetched >= max_files:
            break
        actual = lower_files.get(candidate.lower())
        if not actual:
            continue
        content = _safe_read(root, actual, max_file_bytes)
        if content is None:
            continue
        ext = Path(actual).suffix
        lang_hint = "python" if ext == ".py" else ""
        parts.append(f"### {actual}")
        parts.append(f"```{lang_hint}\n{content}\n```\n")
        fetched += 1

    # 3) Read additional source files (shallow depth, by extension)
    if fetched < max_files:
        source_files = [
            f for f in all_files
            if Path(f).suffix in _CODE_EXTS
            and f.lower() not in {k.lower() for k in _KEY_FILES}
            and len(Path(f).parts) <= 2  # top-level or one dir deep
        ]
        for sf in sorted(source_files)[:max_files - fetched]:
            content = _safe_read(root, sf, max_file_bytes)
            if content is None:
                continue
            ext = Path(sf).suffix.lstrip(".")
            parts.append(f"### {sf}")
            parts.append(f"```{ext}\n{content}\n```\n")
            fetched += 1

    return "\n".join(parts)


def _build_paper_context(paper: dict) -> str:
    """Build a rich text context block from a paper's stored fields."""
    parts = [
        f"Title: {paper.get('title', '')}",
        f"Authors: {paper.get('authors', '')}",
        f"Published: {paper.get('published', '')}",
    ]
    if paper.get("venue"):
        parts.append(f"Venue: {paper['venue']}")
    if paper.get("abstract"):
        parts.append(f"\nAbstract:\n{paper['abstract']}")
    if paper.get("summary"):
        parts.append(f"\nSummary:\n{paper['summary']}")
    if paper.get("key_insight"):
        parts.append(f"\nKey Insight: {paper['key_insight']}")
    if paper.get("method"):
        parts.append(f"\nMethod: {paper['method']}")
    if paper.get("contribution"):
        parts.append(f"\nContribution: {paper['contribution']}")
    if paper.get("math_concepts"):
        concepts = paper["math_concepts"]
        if isinstance(concepts, list):
            parts.append(f"\nMathematical Concepts: {', '.join(concepts)}")
    if paper.get("cited_works"):
        works = paper["cited_works"]
        if isinstance(works, list):
            parts.append(f"\nNotable Cited Works: {', '.join(works)}")
    return "\n".join(parts)


_LANG_INSTRUCTIONS = {
    "en": "",
    "zh": "\n\nIMPORTANT: You MUST write your entire response in Chinese (简体中文). Use Chinese for all analysis text, but keep technical terms, paper titles, author names, and mathematical notation in their original form.",
}


def _lang_suffix(language: str) -> str:
    return _LANG_INSTRUCTIONS.get(language, "")


def generate_deep_read(
    topic_id: str,
    topic_name: str,
    data_dir: str,
    cfg: dict,
    paper_id: str,
    on_section_done: Callable[[str, str], None] | None = None,
    language: str = "en",
) -> dict:
    """Run a 6-stage deep reading analysis on a single paper.

    Stages:
      1. motivation_analysis
      2. method_analysis
      3. experiment_analysis
      4. contribution_analysis
      5. summary_card_json (structured JSON)
      6. code_review (if GitHub repos detected)

    Args:
        topic_id: The topic this paper belongs to.
        topic_name: Human-readable topic name for prompt context.
        data_dir: Root data directory path.
        cfg: Pipeline config dict.
        paper_id: The arxiv_id of the paper to analyse.
        on_section_done: Optional callback(section_name, content) after each stage.

    Returns:
        Dict with all analysis fields.
    """
    store = Storage(data_dir, topic_id)
    try:
        paper = store.get_arxiv(paper_id)
    finally:
        store.close()

    if not paper:
        raise ValueError(f"Paper not found: {paper_id}")

    ctx = _build_paper_context(paper)
    lang = _lang_suffix(language)
    result: dict = {}

    # --- Stage 1: Motivation ---
    prompt_motivation = f"""You are a senior research scientist performing a deep reading of the following paper.

PAPER:
{ctx}

TASK: Analyse the **motivation and research question** of this paper. Cover:
1. What is the core research question or problem being addressed?
2. Why is this problem important right now? What gap in prior work does it fill?
3. What assumptions does the paper make about the problem setting?
4. How timely and relevant is the research topic?

Write a thorough yet concise analysis (300-500 words). Use markdown formatting.
Output ONLY the analysis, no preamble.{lang}"""

    motivation = call_cli(prompt_motivation, cfg, model="opus", timeout=300) or ""
    result["motivation_analysis"] = motivation
    if on_section_done:
        on_section_done("motivation_analysis", motivation)

    # --- Stage 2: Method ---
    prompt_method = f"""You are a senior research scientist performing a deep reading of the following paper.

PAPER:
{ctx}

PREVIOUS ANALYSIS — Motivation:
{motivation}

TASK: Analyse the **methodology and technical approach**. Cover:
1. What is the overall technical pipeline / architecture?
2. Break down the key algorithmic steps, formulas, or model components.
3. What design choices were made, and what alternatives exist?
4. How does the method relate to or differ from prior approaches?

Write a thorough analysis (400-600 words). Include relevant equations in LaTeX if applicable. Use markdown formatting.
Output ONLY the analysis, no preamble.{lang}"""

    method = call_cli(prompt_method, cfg, model="opus", timeout=300) or ""
    result["method_analysis"] = method
    if on_section_done:
        on_section_done("method_analysis", method)

    # --- Stage 3: Experiments ---
    prompt_experiment = f"""You are a senior research scientist performing a deep reading of the following paper.

PAPER:
{ctx}

PREVIOUS ANALYSIS — Motivation:
{motivation}

Method:
{method}

TASK: Analyse the **experimental evaluation**. Cover:
1. What datasets, baselines, and metrics are used? Are they appropriate?
2. Evaluate the experimental design quality (controls, ablations, statistical rigor).
3. Do the results convincingly support the paper's claims?
4. How reproducible is the work based on the information provided?

Write a thorough analysis (300-500 words). Use markdown formatting.
Output ONLY the analysis, no preamble.{lang}"""

    experiment = call_cli(prompt_experiment, cfg, model="opus", timeout=300) or ""
    result["experiment_analysis"] = experiment
    if on_section_done:
        on_section_done("experiment_analysis", experiment)

    # --- Stage 4: Contribution ---
    prompt_contribution = f"""You are a senior research scientist performing a deep reading of the following paper.

PAPER:
{ctx}

PREVIOUS ANALYSIS — Motivation:
{motivation}

Method:
{method}

Experiments:
{experiment}

TASK: Evaluate the **contribution and impact**. Cover:
1. What is genuinely novel about this work? Rate novelty (low / medium / high).
2. Actual impact vs. claimed contribution — any overclaims?
3. Key limitations and failure modes.
4. How well does the approach generalise beyond the tested settings?
5. What follow-up research directions does this work enable?

Write a thorough analysis (300-500 words). Use markdown formatting.
Output ONLY the analysis, no preamble.{lang}"""

    contribution = call_cli(prompt_contribution, cfg, model="opus", timeout=300) or ""
    result["contribution_analysis"] = contribution
    if on_section_done:
        on_section_done("contribution_analysis", contribution)

    # --- Stage 5: Summary Card ---
    prompt_card = f"""You are a senior research scientist. Based on the deep reading analysis below, produce a structured summary card as JSON.

PAPER:
{ctx}

ANALYSIS — Motivation:
{motivation}

Method:
{method}

Experiments:
{experiment}

Contribution:
{contribution}

TASK: Output a JSON object with these fields (all values are arrays of short strings):
- "key_innovations": 2-4 bullet points of the paper's main innovations
- "strengths": 3-5 bullet points
- "weaknesses": 2-4 bullet points
- "field_comparison": 2-3 bullets comparing this work to the state of the field
- "inspiration_points": 2-4 ideas this paper could inspire for future research

Output ONLY valid JSON, no markdown fences, no explanation.{lang}"""

    card_raw = call_cli(prompt_card, cfg, model="opus", timeout=300) or "{}"
    # Try to parse JSON, cleaning common issues
    card_raw = card_raw.strip()
    if card_raw.startswith("```"):
        # Strip markdown code fences
        lines = card_raw.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        card_raw = "\n".join(lines).strip()
    try:
        card = json.loads(card_raw)
    except json.JSONDecodeError:
        log.warning("Failed to parse summary card JSON, storing raw text")
        card = {
            "key_innovations": [],
            "strengths": [],
            "weaknesses": [],
            "field_comparison": [],
            "inspiration_points": [],
            "_raw": card_raw,
        }

    result["summary_card_json"] = card
    if on_section_done:
        on_section_done("summary_card_json", json.dumps(card))

    # --- Stage 6: Code Review (conditional) ---
    repo_urls = _extract_repo_urls(paper)
    result["repo_urls"] = repo_urls

    repos_dir = os.path.join(data_dir, topic_id, "repos")
    cloned_paths: list[str] = []

    if repo_urls:
        log.info("Found %d repo(s) for paper %s: %s", len(repo_urls), paper_id, repo_urls)

        # Clone repos locally (cap at 3)
        for url in repo_urls[:3]:
            local = _clone_repo(url, repos_dir)
            if local:
                cloned_paths.append(local)

        result["repo_local_paths"] = cloned_paths

        if cloned_paths:
            repo_contexts = [_build_repo_context(p) for p in cloned_paths]
            repo_block = "\n\n---\n\n".join(repo_contexts)

            prompt_code_review = f"""You are a senior research engineer reviewing the code implementation of a research paper.

PAPER:
{ctx}

METHOD ANALYSIS:
{method}

CONTRIBUTION ANALYSIS:
{contribution}

CODE REPOSITORIES:
{repo_block}

TASK: Perform a thorough **code review** of the paper's implementation. Cover:
1. **Repository Overview**: What does the codebase implement? Overall architecture and structure.
2. **Implementation Quality**: Code organization, readability, documentation quality, dependency management.
3. **Paper-Code Alignment**: How faithfully does the code implement the method described in the paper? Any discrepancies?
4. **Reproducibility**: Are configs/hyperparams/seeds provided? Can experiments be reproduced from the repo alone?
5. **Engineering Practices**: Testing, CI/CD, error handling, logging. Are there any red flags?
6. **Key Technical Decisions**: Notable design patterns, frameworks, or libraries used. Are they appropriate choices?
7. **Potential Issues**: Bugs, performance concerns, scalability limitations, or technical debt.

Write a thorough analysis (500-800 words). Use markdown formatting with clear section headers.
Output ONLY the analysis, no preamble.{lang}"""

            code_review = call_cli(prompt_code_review, cfg, model="opus", timeout=300) or ""
        else:
            code_review = ""

        result["code_review"] = code_review
        if on_section_done:
            on_section_done("code_review", code_review)
    else:
        log.info("No GitHub repos found for paper %s, skipping code review", paper_id)
        result["code_review"] = ""
        result["repo_local_paths"] = []
        if on_section_done:
            on_section_done("code_review", "")

    return result


def generate_deep_read_qa(
    paper: dict,
    existing_analysis: dict,
    prior_messages: list[dict],
    user_question: str,
    cfg: dict,
    language: str = "en",
) -> dict:
    """Answer a follow-up question in the context of a deep-read session.

    Args:
        paper: Full paper dict from Storage.
        existing_analysis: Dict with motivation/method/experiment/contribution analyses.
        prior_messages: List of prior message dicts (role, content).
        user_question: The user's new question.
        cfg: Pipeline config dict.

    Returns:
        {"content": str} with the assistant's answer.
    """
    ctx = _build_paper_context(paper)

    analysis_parts = []
    for key, label in [
        ("motivation_analysis", "Motivation Analysis"),
        ("method_analysis", "Method Analysis"),
        ("experiment_analysis", "Experiment Analysis"),
        ("contribution_analysis", "Contribution Analysis"),
        ("code_review", "Code Review"),
    ]:
        val = existing_analysis.get(key, "")
        if val:
            analysis_parts.append(f"## {label}\n{val}")
    analysis_block = "\n\n".join(analysis_parts)

    history_lines = []
    for msg in prior_messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if content:
            history_lines.append(f"[{role.upper()}]: {content}")
    history_block = "\n\n".join(history_lines) if history_lines else "(no prior conversation)"

    prompt = f"""You are a research assistant helping a user understand a paper in depth.
You have already produced a detailed analysis of the paper. Use it as context to answer the user's question.

PAPER:
{ctx}

EXISTING DEEP READ ANALYSIS:
{analysis_block}

CONVERSATION HISTORY:
{history_block}

USER'S QUESTION:
{user_question}

Provide a clear, thorough answer. Reference specific parts of the paper or your analysis when relevant.
Use markdown formatting. Include equations in LaTeX if helpful.
Output ONLY the answer, no preamble.{_lang_suffix(language)}"""

    answer = call_cli(prompt, cfg, model="opus", timeout=120) or "Sorry, I was unable to generate an answer."
    return {"content": answer}
