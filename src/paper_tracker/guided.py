"""Guided topic configuration assistant.

Streams a conversational LLM session (via Claude CLI stream-json) that helps
the user define a paper-tracker topic config through 1-2 clarifying questions,
then proposes a fenced JSON block with the config fields.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from collections.abc import Generator
from datetime import date, timedelta

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a research topic configuration assistant for a paper-tracking system.
Your job is to help the user define a focused research monitoring topic.

Guidelines:
- Ask 1-2 SHORT clarifying questions to understand: (a) the precise research
  focus, (b) relevant venues/communities, and (c) whether they care more about
  theory or implementation/applications.
- Once you have enough context (after at most one round of clarification),
  propose a complete topic configuration as a fenced JSON block.
- Use the same language the user is writing in.

When you are ready to propose a config, output it as:

```json
{
  "name": "<short descriptive name>",
  "description": "<one-sentence description>",
  "arxiv_keywords": ["<kw1>", "<kw2>", ...],
  "arxiv_categories": ["cs.LG", ...],
  "github_keywords": ["<kw1>", ...]
}
```

Only include the ```json block when you are proposing the final config.
Do NOT include partial or placeholder JSON blocks before that.
"""

# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


def build_prompt(messages: list[dict]) -> str:
    """Format a conversation history into a plain-text prompt.

    Each message dict must have ``role`` ("user" or "assistant") and
    ``content`` (string).

    Example output::

        [User]
        What is your research interest?

        [Assistant]
        I need a bit more context — are you focusing on …
    """
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        label = "[User]" if role == "user" else "[Assistant]"
        parts.append(f"{label}\n{content}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Streaming generator
# ---------------------------------------------------------------------------


def stream_guided_response(messages: list[dict]) -> Generator[str, None, None]:
    """Stream SSE events from a guided topic-config conversation.

    Spawns ``claude`` CLI with ``--output-format stream-json`` and yields
    Server-Sent Events::

        event: delta
        data: {"text": "<incremental text>"}

        event: done
        data: {"message": "<full text>", "stage": "chatting"|"ready",
               "draft_config": null | {...}}

    Args:
        messages: Conversation history as ``[{"role": ..., "content": ...}]``.

    Yields:
        SSE-formatted strings (each ending with ``\\n\\n``).
    """
    prompt = build_prompt(messages)

    cmd = [
        "claude",
        "-p", "-",
        "--output-format", "stream-json",
        "--verbose",
        "--include-partial-messages",
        "--model", "sonnet",
        "--system-prompt", SYSTEM_PROMPT,
        "--bare",
    ]

    env = os.environ.copy()
    env.pop("CLAUDE_CODE_ENTRYPOINT", None)
    env.pop("CLAUDECODE", None)

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    try:
        proc.stdin.write(prompt.encode())
        proc.stdin.close()
    except OSError:
        pass

    prev_text = ""
    full_text = ""

    for raw_line in proc.stdout:
        line = raw_line.decode(errors="replace").strip()
        if not line:
            continue

        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            log.debug("guided: non-JSON line: %s", line[:200])
            continue

        etype = event.get("type", "")

        if etype == "assistant":
            # Extract text from content blocks
            current_text = ""
            message = event.get("message", event)
            content_blocks = message.get("content", [])
            if isinstance(content_blocks, list):
                for block in content_blocks:
                    if isinstance(block, dict) and block.get("type") == "text":
                        current_text += block.get("text", "")
            elif isinstance(content_blocks, str):
                current_text = content_blocks

            delta = current_text[len(prev_text):]
            if delta:
                prev_text = current_text
                payload = json.dumps({"text": delta}, ensure_ascii=False)
                yield f"event: delta\ndata: {payload}\n\n"

        elif etype == "result":
            # Final result — extract full text
            result_val = event.get("result", "")
            if isinstance(result_val, str):
                full_text = result_val
            else:
                # Fallback: use last accumulated assistant text
                full_text = prev_text

            draft_config = _extract_draft_config(full_text)
            stage = "ready" if draft_config is not None else "chatting"
            payload = json.dumps(
                {
                    "message": full_text,
                    "stage": stage,
                    "draft_config": draft_config,
                },
                ensure_ascii=False,
            )
            yield f"event: done\ndata: {payload}\n\n"

    proc.wait()


# ---------------------------------------------------------------------------
# Config extraction helpers
# ---------------------------------------------------------------------------


def _extract_draft_config(text: str) -> dict | None:
    """Extract a fenced ```json block from *text* and return a filled config.

    Returns None if no valid block with the required keys is found.
    """
    pattern = r"```json\s*(\{.*?\})\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None

    raw = match.group(1)
    try:
        config = json.loads(raw)
    except json.JSONDecodeError:
        log.debug("guided: failed to parse JSON block: %s", raw[:300])
        return None

    if "name" not in config or "arxiv_keywords" not in config:
        log.debug("guided: JSON block missing required keys")
        return None

    return _fill_defaults(config)


def _fill_defaults(config: dict) -> dict:
    """Fill missing fields in a draft config with sensible defaults.

    Required caller-supplied fields (must already be present):
        name, arxiv_keywords

    All other fields get defaults if absent.
    """
    today = date.today()
    one_year_ago = today - timedelta(days=365)

    kws = config.get("arxiv_keywords", [])
    first_four = kws[:4]

    defaults = {
        "description": "",
        "arxiv_categories": [],
        "arxiv_lookback_days": 365,
        "github_keywords": [],
        "github_lookback_days": 365,
        "search_date_from": one_year_ago.isoformat(),
        "search_date_to": today.isoformat(),
        "schedule_cron": "",
        "enabled": True,
        "openalex_enabled": True,
        "openalex_keywords": first_four,
        "openalex_lookback_days": 365,
        "openalex_venues": [],
        "openalex_max_results": 200,
        "openreview_enabled": True,
        "openreview_venues": [],
        "openreview_keywords": first_four,
        "openreview_max_results": 100,
    }

    for key, value in defaults.items():
        config.setdefault(key, value)

    return config
