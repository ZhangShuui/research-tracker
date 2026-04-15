# Guided Topic Creation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a conversational guided-create flow for topics, streaming LLM responses via Claude CLI's stream-json mode, alongside the existing quick-create.

**Architecture:** New backend SSE endpoint (`POST /api/topics/guided`) spawns a `claude -p --output-format stream-json` subprocess per turn, forwarding partial messages as SSE events. Frontend adds a `GuidedCreateModal` with chat phase + editable preview phase. Dashboard gets a dropdown button with Quick Create / Guided Create options.

**Tech Stack:** FastAPI StreamingResponse (SSE), Claude CLI stream-json, Next.js 14, React, TypeScript, Tailwind CSS

---

## File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `src/paper_tracker/guided.py` | System prompt, prompt builder, CLI subprocess streaming, JSON extraction |
| Modify | `src/paper_tracker/server.py` | Add `POST /api/topics/guided` SSE endpoint |
| Modify | `frontend/src/lib/api.ts` | Add `guidedCreateStream()` SSE fetch helper |
| Create | `frontend/src/components/GuidedCreateModal.tsx` | Chat UI + editable preview modal |
| Modify | `frontend/src/app/page.tsx` | Split "New Topic" button into dropdown |

---

### Task 1: Backend — `guided.py` module

**Files:**
- Create: `src/paper_tracker/guided.py`

This module handles: system prompt definition, building the full prompt from message history, spawning the Claude CLI subprocess with stream-json output, yielding parsed SSE events, and extracting draft_config JSON from the final result.

- [ ] **Step 1: Create `guided.py` with system prompt and prompt builder**

```python
"""Guided topic creation — conversational flow via Claude CLI streaming."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from collections.abc import Generator
from datetime import datetime, timedelta, timezone

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a research topic configuration assistant for a paper-tracking system.

The user has a vague research interest and needs help turning it into a well-defined topic with:
- A clear topic name
- A one-sentence description
- arXiv search keywords (6-8 phrases, at least 3 must be 1-2 words)
- arXiv categories (e.g. cs.CL, cs.AI, cs.LG)
- GitHub search keywords (2-4 phrases)

## Your approach:
1. Ask 1-2 clarifying questions to understand what the user cares about — scope, angle, what makes it distinct, whether it spans subfields.
2. When you feel you understand enough, propose a topic configuration as a fenced JSON block:

```json
{
  "name": "Topic Name",
  "description": "One-sentence description",
  "arxiv_keywords": ["keyword1", "keyword2", ...],
  "arxiv_categories": ["cs.CL", "cs.AI"],
  "github_keywords": ["keyword1", "keyword2"]
}
```

3. If the user wants changes, adjust and output the updated JSON block.
4. Keep responses concise. Use the user's language (if they write in Chinese, reply in Chinese).
"""


def build_prompt(messages: list[dict[str, str]]) -> str:
    """Build a single prompt string from conversation history.

    The last message must be from the user. Prior messages provide context.
    """
    parts: list[str] = []
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        parts.append(f"[{role}]\n{msg['content']}")
    return "\n\n".join(parts)
```

- [ ] **Step 2: Add `stream_guided_response()` generator**

This function spawns `claude -p --output-format stream-json --verbose --include-partial-messages --model sonnet --system-prompt "..." --bare`, reads stdout line-by-line, and yields SSE-formatted strings.

```python
def stream_guided_response(
    messages: list[dict[str, str]],
) -> Generator[str, None, None]:
    """Spawn Claude CLI and yield SSE event strings.

    Yields:
        "event: delta\\ndata: {...}\\n\\n"  — partial token
        "event: done\\ndata: {...}\\n\\n"   — final result with stage + optional draft_config
    """
    prompt = build_prompt(messages)

    env = os.environ.copy()
    env.pop("CLAUDE_CODE_ENTRYPOINT", None)
    env.pop("CLAUDECODE", None)

    cmd = [
        "claude", "-p", "-",
        "--output-format", "stream-json",
        "--verbose",
        "--include-partial-messages",
        "--model", "sonnet",
        "--system-prompt", SYSTEM_PROMPT,
        "--bare",
    ]

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

    full_text = ""
    prev_text = ""

    for raw_line in proc.stdout:
        line = raw_line.decode(errors="replace").strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        if event.get("type") == "assistant":
            # Extract text content from partial message
            msg = event.get("message", {})
            content_blocks = msg.get("content", [])
            text = ""
            for block in content_blocks:
                if block.get("type") == "text":
                    text = block.get("text", "")
            if text and len(text) > len(prev_text):
                delta = text[len(prev_text):]
                prev_text = text
                yield f"event: delta\ndata: {json.dumps({'text': delta})}\n\n"

        elif event.get("type") == "result":
            full_text = event.get("result", "")
            draft = _extract_draft_config(full_text)
            stage = "ready" if draft else "chatting"
            payload = {
                "message": full_text,
                "stage": stage,
                "draft_config": draft,
            }
            yield f"event: done\ndata: {json.dumps(payload)}\n\n"

    proc.wait(timeout=10)
```

- [ ] **Step 3: Add `_extract_draft_config()` and `_fill_defaults()`**

```python
def _extract_draft_config(text: str) -> dict | None:
    """Extract a JSON config block from the LLM response text.

    Looks for a fenced ```json block containing "name" and "arxiv_keywords".
    Returns the config with defaults filled in, or None if not found.
    """
    pattern = r"```json\s*\n(.*?)\n\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None
    try:
        config = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    if "name" not in config or "arxiv_keywords" not in config:
        return None
    return _fill_defaults(config)


def _fill_defaults(config: dict) -> dict:
    """Fill in default values for fields not provided by the LLM."""
    date_from = (datetime.now(timezone.utc) - timedelta(days=365)).strftime("%Y-%m-%d")
    date_to = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    arxiv_kws = config.get("arxiv_keywords", [])

    defaults = {
        "description": "",
        "arxiv_keywords": [],
        "arxiv_categories": ["cs.CL", "cs.AI", "cs.LG"],
        "arxiv_lookback_days": 365,
        "github_keywords": [],
        "github_lookback_days": 365,
        "schedule_cron": "",
        "enabled": True,
        "search_date_from": date_from,
        "search_date_to": date_to,
        "openalex_enabled": True,
        "openalex_keywords": arxiv_kws[:4],
        "openalex_lookback_days": 365,
        "openalex_venues": [],
        "openalex_max_results": 200,
        "openreview_enabled": True,
        "openreview_venues": [],
        "openreview_keywords": arxiv_kws[:4],
        "openreview_max_results": 100,
    }

    for key, val in defaults.items():
        if key not in config:
            config[key] = val

    return config
```

- [ ] **Step 4: Commit**

```bash
git add src/paper_tracker/guided.py
git commit -m "feat: add guided topic creation backend module with CLI streaming"
```

---

### Task 2: Backend — SSE endpoint in `server.py`

**Files:**
- Modify: `src/paper_tracker/server.py`

- [ ] **Step 1: Add Pydantic model and endpoint**

Add after the existing `quick_create_topic` endpoint (around line 346):

```python
from fastapi.responses import StreamingResponse

class GuidedMessage(BaseModel):
    role: str
    content: str

class GuidedRequest(BaseModel):
    messages: list[GuidedMessage]

@app.post("/api/topics/guided")
async def guided_create_topic(body: GuidedRequest):
    """SSE streaming endpoint for guided topic creation conversation."""
    from paper_tracker.guided import stream_guided_response

    messages = [{"role": m.role, "content": m.content} for m in body.messages]

    return StreamingResponse(
        stream_guided_response(messages),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
```

- [ ] **Step 2: Add `StreamingResponse` import at the top of server.py**

Add to the imports section (around line 13):

```python
from fastapi.responses import StreamingResponse
```

- [ ] **Step 3: Commit**

```bash
git add src/paper_tracker/server.py
git commit -m "feat: add POST /api/topics/guided SSE endpoint"
```

---

### Task 3: Frontend — API client SSE helper

**Files:**
- Modify: `frontend/src/lib/api.ts`

- [ ] **Step 1: Add types and `guidedCreateStream` function**

Add at the end of `api.ts`, after the `api` object closing brace:

```typescript
// --- Guided Create SSE ---

export interface GuidedDelta {
  text: string;
}

export interface GuidedDone {
  message: string;
  stage: "chatting" | "ready";
  draft_config: TopicCreate | null;
}

export type GuidedEvent =
  | { type: "delta"; data: GuidedDelta }
  | { type: "done"; data: GuidedDone };

export async function guidedCreateStream(
  messages: { role: string; content: string }[],
  onEvent: (event: GuidedEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${BASE}/api/topics/guided`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages }),
    signal,
  });

  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`${res.status} ${text}`);
  }

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // Parse SSE frames from buffer
    const parts = buffer.split("\n\n");
    buffer = parts.pop()!; // keep incomplete frame

    for (const part of parts) {
      const lines = part.split("\n");
      let eventType = "";
      let dataStr = "";
      for (const line of lines) {
        if (line.startsWith("event: ")) eventType = line.slice(7);
        else if (line.startsWith("data: ")) dataStr = line.slice(6);
      }
      if (!eventType || !dataStr) continue;
      try {
        const data = JSON.parse(dataStr);
        onEvent({ type: eventType, data } as GuidedEvent);
      } catch {
        // skip malformed JSON
      }
    }
  }
}
```

- [ ] **Step 2: Commit**

```bash
cd frontend && git add src/lib/api.ts
git commit -m "feat: add guidedCreateStream SSE helper in api.ts"
```

---

### Task 4: Frontend — `GuidedCreateModal` component

**Files:**
- Create: `frontend/src/components/GuidedCreateModal.tsx`

This is the main UI component with chat phase and editable preview phase.

- [ ] **Step 1: Create the component with chat phase**

```tsx
"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  X,
  Send,
  ArrowLeft,
  ChevronDown,
  ChevronRight,
  Loader2,
} from "lucide-react";
import {
  api,
  TopicCreate,
  GuidedEvent,
  guidedCreateStream,
} from "@/lib/api";

interface Props {
  onClose: () => void;
}

interface Message {
  role: "user" | "assistant";
  content: string;
}

export function GuidedCreateModal({ onClose }: Props) {
  const qc = useQueryClient();
  const [phase, setPhase] = useState<"chat" | "preview">("chat");
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [streamingText, setStreamingText] = useState("");
  const [draftConfig, setDraftConfig] = useState<TopicCreate | null>(null);
  const [error, setError] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingText]);

  const sendMessage = useCallback(async (userText: string) => {
    if (!userText.trim() || streaming) return;
    setError("");

    const userMsg: Message = { role: "user", content: userText.trim() };
    const newMessages = [...messages, userMsg];
    setMessages(newMessages);
    setInput("");
    setStreaming(true);
    setStreamingText("");

    const abort = new AbortController();
    abortRef.current = abort;

    try {
      await guidedCreateStream(
        newMessages,
        (event: GuidedEvent) => {
          if (event.type === "delta") {
            setStreamingText((prev) => prev + event.data.text);
          } else if (event.type === "done") {
            const assistantMsg: Message = {
              role: "assistant",
              content: event.data.message,
            };
            setMessages((prev) => [...prev, assistantMsg]);
            setStreamingText("");
            setStreaming(false);
            if (event.data.stage === "ready" && event.data.draft_config) {
              setDraftConfig(event.data.draft_config);
              setPhase("preview");
            }
          }
        },
        abort.signal,
      );
    } catch (err: unknown) {
      if (err instanceof Error && err.name !== "AbortError") {
        setError(err.message);
      }
      setStreaming(false);
      setStreamingText("");
    }
  }, [messages, streaming]);

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    sendMessage(input);
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input);
    }
  }

  // --- Chat Phase ---
  if (phase === "chat") {
    return (
      <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50 p-4">
        <div className="bg-white rounded-2xl shadow-xl w-full max-w-2xl h-[80vh] flex flex-col">
          {/* Header */}
          <div className="flex items-center justify-between p-5 border-b">
            <div>
              <h2 className="font-semibold text-lg">Guided Topic Creation</h2>
              <p className="text-xs text-slate-400 mt-0.5">
                Describe your research interest and I&apos;ll help define the topic
              </p>
            </div>
            <button
              onClick={onClose}
              className="p-1 rounded-lg hover:bg-gray-100 transition-colors"
            >
              <X size={18} />
            </button>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-5 space-y-4">
            {messages.length === 0 && !streaming && (
              <div className="text-center text-slate-400 text-sm py-12">
                Tell me about the research area you&apos;re interested in.
                <br />
                It&apos;s okay to be vague — I&apos;ll help you refine it.
              </div>
            )}
            {messages.map((msg, i) => (
              <MessageBubble key={i} message={msg} />
            ))}
            {streaming && streamingText && (
              <MessageBubble
                message={{ role: "assistant", content: streamingText }}
                isStreaming
              />
            )}
            {streaming && !streamingText && (
              <div className="flex items-center gap-2 text-slate-400 text-sm">
                <Loader2 size={14} className="animate-spin" />
                Thinking...
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Error */}
          {error && (
            <div className="mx-5 mb-2 text-sm text-red-600 bg-red-50 rounded-lg p-3">
              {error}
            </div>
          )}

          {/* Input */}
          <form onSubmit={handleSubmit} className="p-4 border-t">
            <div className="flex gap-2">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Describe your research interest..."
                rows={1}
                className="flex-1 resize-none rounded-xl border border-slate-200 px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500/40 focus:border-indigo-400"
                disabled={streaming}
                autoFocus
              />
              <button
                type="submit"
                disabled={!input.trim() || streaming}
                className="p-2.5 rounded-xl bg-indigo-600 text-white hover:bg-indigo-700 transition-colors disabled:opacity-40"
              >
                <Send size={16} />
              </button>
            </div>
          </form>
        </div>
      </div>
    );
  }

  // --- Preview Phase (Task 5) ---
  return (
    <GuidedPreview
      draftConfig={draftConfig!}
      onBack={() => setPhase("chat")}
      onClose={onClose}
    />
  );
}

function MessageBubble({
  message,
  isStreaming,
}: {
  message: Message;
  isStreaming?: boolean;
}) {
  const isUser = message.role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[80%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed whitespace-pre-wrap ${
          isUser
            ? "bg-indigo-600 text-white"
            : "bg-slate-100 text-slate-800"
        } ${isStreaming ? "animate-pulse" : ""}`}
      >
        {message.content}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Add `GuidedPreview` sub-component for the editable preview phase**

Append to the same file, after the `MessageBubble` component:

```tsx
function GuidedPreview({
  draftConfig,
  onBack,
  onClose,
}: {
  draftConfig: TopicCreate;
  onBack: () => void;
  onClose: () => void;
}) {
  const qc = useQueryClient();
  const [name, setName] = useState(draftConfig.name);
  const [description, setDescription] = useState(draftConfig.description ?? "");
  const [arxivKeywords, setArxivKeywords] = useState(
    (draftConfig.arxiv_keywords ?? []).join("\n")
  );
  const [arxivCategories, setArxivCategories] = useState(
    (draftConfig.arxiv_categories ?? []).join(", ")
  );
  const [arxivLookback, setArxivLookback] = useState(
    draftConfig.arxiv_lookback_days ?? 365
  );
  const [githubKeywords, setGithubKeywords] = useState(
    (draftConfig.github_keywords ?? []).join("\n")
  );
  const [githubLookback, setGithubLookback] = useState(
    draftConfig.github_lookback_days ?? 365
  );
  const [scheduleCron, setScheduleCron] = useState(draftConfig.schedule_cron ?? "");
  const [searchDateFrom, setSearchDateFrom] = useState(draftConfig.search_date_from ?? "");
  const [searchDateTo, setSearchDateTo] = useState(draftConfig.search_date_to ?? "");
  const [showSources, setShowSources] = useState(true);
  const [oaEnabled, setOaEnabled] = useState(draftConfig.openalex_enabled ?? true);
  const [oaKeywords, setOaKeywords] = useState(
    (draftConfig.openalex_keywords ?? []).join("\n")
  );
  const [oaLookback, setOaLookback] = useState(draftConfig.openalex_lookback_days ?? 365);
  const [oaVenues, setOaVenues] = useState(
    (draftConfig.openalex_venues ?? []).join(", ")
  );
  const [oaMaxResults, setOaMaxResults] = useState(draftConfig.openalex_max_results ?? 200);
  const [orEnabled, setOrEnabled] = useState(draftConfig.openreview_enabled ?? true);
  const [orVenues, setOrVenues] = useState(
    (draftConfig.openreview_venues ?? []).join(", ")
  );
  const [orKeywords, setOrKeywords] = useState(
    (draftConfig.openreview_keywords ?? []).join("\n")
  );
  const [orMaxResults, setOrMaxResults] = useState(draftConfig.openreview_max_results ?? 100);
  const [error, setError] = useState("");

  const createMut = useMutation({
    mutationFn: (body: TopicCreate) => api.createTopic(body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["topics"] });
      onClose();
    },
    onError: (e: Error) => setError(e.message),
  });

  function handleCreate(e: React.FormEvent) {
    e.preventDefault();
    const body: TopicCreate = {
      name: name.trim(),
      description: description.trim(),
      arxiv_keywords: arxivKeywords.split("\n").map((s) => s.trim()).filter(Boolean),
      arxiv_categories: arxivCategories.split(",").map((s) => s.trim()).filter(Boolean),
      arxiv_lookback_days: Number(arxivLookback),
      github_keywords: githubKeywords.split("\n").map((s) => s.trim()).filter(Boolean),
      github_lookback_days: Number(githubLookback),
      schedule_cron: scheduleCron.trim(),
      enabled: true,
      search_date_from: searchDateFrom,
      search_date_to: searchDateTo,
      openalex_enabled: oaEnabled,
      openalex_keywords: oaKeywords.split("\n").map((s) => s.trim()).filter(Boolean),
      openalex_lookback_days: Number(oaLookback),
      openalex_venues: oaVenues.split(",").map((s) => s.trim()).filter(Boolean),
      openalex_max_results: Number(oaMaxResults),
      openreview_enabled: orEnabled,
      openreview_venues: orVenues.split(",").map((s) => s.trim()).filter(Boolean),
      openreview_keywords: orKeywords.split("\n").map((s) => s.trim()).filter(Boolean),
      openreview_max_results: Number(orMaxResults),
    };
    createMut.mutate(body);
  }

  return (
    <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl shadow-xl w-full max-w-2xl max-h-[90vh] overflow-y-auto">
        <div className="flex items-center justify-between p-5 border-b">
          <div className="flex items-center gap-3">
            <button
              onClick={onBack}
              className="p-1.5 rounded-lg hover:bg-gray-100 transition-colors"
              title="Back to Chat"
            >
              <ArrowLeft size={18} />
            </button>
            <h2 className="font-semibold text-lg">Review & Create</h2>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg hover:bg-gray-100 transition-colors"
          >
            <X size={18} />
          </button>
        </div>

        <form onSubmit={handleCreate} className="p-5 space-y-4">
          <Field label="Name *">
            <input
              required
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="input"
            />
          </Field>
          <Field label="Description">
            <input
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className="input"
            />
          </Field>
          <Field label="arXiv Keywords (one per line)">
            <textarea
              value={arxivKeywords}
              onChange={(e) => setArxivKeywords(e.target.value)}
              className="input h-24 resize-none"
            />
          </Field>
          <Field label="arXiv Categories (comma-separated)">
            <input
              value={arxivCategories}
              onChange={(e) => setArxivCategories(e.target.value)}
              className="input"
            />
          </Field>
          <div className="grid grid-cols-2 gap-4">
            <Field label="Date From">
              <input
                type="date"
                value={searchDateFrom}
                onChange={(e) => setSearchDateFrom(e.target.value)}
                className="input"
              />
            </Field>
            <Field label="Date To">
              <input
                type="date"
                value={searchDateTo}
                onChange={(e) => setSearchDateTo(e.target.value)}
                className="input"
              />
            </Field>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <Field label="arXiv Lookback Days">
              <input
                type="number"
                min={1}
                max={365}
                value={arxivLookback}
                onChange={(e) => setArxivLookback(Number(e.target.value))}
                className="input"
              />
            </Field>
            <Field label="GitHub Lookback Days">
              <input
                type="number"
                min={1}
                max={365}
                value={githubLookback}
                onChange={(e) => setGithubLookback(Number(e.target.value))}
                className="input"
              />
            </Field>
          </div>
          <Field label="GitHub Keywords (one per line)">
            <textarea
              value={githubKeywords}
              onChange={(e) => setGithubKeywords(e.target.value)}
              className="input h-20 resize-none"
            />
          </Field>
          <Field label="Schedule (cron, optional)">
            <input
              value={scheduleCron}
              onChange={(e) => setScheduleCron(e.target.value)}
              className="input font-mono"
              placeholder="0 8 * * * (daily at 8am)"
            />
          </Field>

          {/* Additional Sources */}
          <div className="border rounded-xl overflow-hidden">
            <button
              type="button"
              onClick={() => setShowSources(!showSources)}
              className="w-full flex items-center justify-between px-4 py-3 text-sm font-medium text-gray-700 hover:bg-gray-50 transition-colors"
            >
              <span>Additional Sources</span>
              {showSources ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
            </button>
            {showSources && (
              <div className="px-4 pb-4 space-y-4 border-t">
                {/* OpenAlex */}
                <div className="pt-3">
                  <label className="flex items-center gap-2 text-sm font-medium text-gray-700 mb-2">
                    <input
                      type="checkbox"
                      checked={oaEnabled}
                      onChange={(e) => setOaEnabled(e.target.checked)}
                      className="rounded"
                    />
                    OpenAlex
                  </label>
                  {oaEnabled && (
                    <div className="ml-6 space-y-3">
                      <Field label="Keywords (one per line)">
                        <textarea
                          value={oaKeywords}
                          onChange={(e) => setOaKeywords(e.target.value)}
                          className="input h-16 resize-none text-sm"
                        />
                      </Field>
                      <Field label="Venue Filter (comma-separated)">
                        <input
                          value={oaVenues}
                          onChange={(e) => setOaVenues(e.target.value)}
                          className="input text-sm"
                        />
                      </Field>
                      <div className="grid grid-cols-2 gap-3">
                        <Field label="Lookback Days">
                          <input
                            type="number"
                            min={1}
                            max={365}
                            value={oaLookback}
                            onChange={(e) => setOaLookback(Number(e.target.value))}
                            className="input text-sm"
                          />
                        </Field>
                        <Field label="Max Results">
                          <input
                            type="number"
                            min={10}
                            max={1000}
                            value={oaMaxResults}
                            onChange={(e) => setOaMaxResults(Number(e.target.value))}
                            className="input text-sm"
                          />
                        </Field>
                      </div>
                    </div>
                  )}
                </div>
                {/* OpenReview */}
                <div className="pt-1 border-t">
                  <label className="flex items-center gap-2 text-sm font-medium text-gray-700 mb-2 mt-3">
                    <input
                      type="checkbox"
                      checked={orEnabled}
                      onChange={(e) => setOrEnabled(e.target.checked)}
                      className="rounded"
                    />
                    OpenReview
                  </label>
                  {orEnabled && (
                    <div className="ml-6 space-y-3">
                      <Field label="Venues (comma-separated)">
                        <input
                          value={orVenues}
                          onChange={(e) => setOrVenues(e.target.value)}
                          className="input text-sm"
                        />
                      </Field>
                      <Field label="Keywords (one per line)">
                        <textarea
                          value={orKeywords}
                          onChange={(e) => setOrKeywords(e.target.value)}
                          className="input h-16 resize-none text-sm"
                        />
                      </Field>
                      <Field label="Max Results per Venue">
                        <input
                          type="number"
                          min={10}
                          max={500}
                          value={orMaxResults}
                          onChange={(e) => setOrMaxResults(Number(e.target.value))}
                          className="input text-sm"
                        />
                      </Field>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          {error && (
            <p className="text-sm text-red-600 bg-red-50 rounded-lg p-3">{error}</p>
          )}

          <div className="flex justify-end gap-3 pt-2">
            <button
              type="button"
              onClick={onBack}
              className="px-4 py-2 text-sm text-gray-600 hover:text-gray-900 transition-colors"
            >
              Back to Chat
            </button>
            <button
              type="submit"
              disabled={createMut.isPending}
              className="px-5 py-2.5 text-sm bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors disabled:opacity-50 font-medium"
            >
              {createMut.isPending ? "Creating..." : "Create Topic"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-sm font-medium text-gray-700">{label}</label>
      {children}
    </div>
  );
}
```

- [ ] **Step 3: Commit**

```bash
cd frontend && git add src/components/GuidedCreateModal.tsx
git commit -m "feat: add GuidedCreateModal component with chat and editable preview phases"
```

---

### Task 5: Frontend — Dashboard dropdown entry point

**Files:**
- Modify: `frontend/src/app/page.tsx`

- [ ] **Step 1: Add guided state and import**

Add import for `GuidedCreateModal` and `MessageCircle` icon, and a new state variable:

```tsx
import { GuidedCreateModal } from "@/components/GuidedCreateModal";
```

Add to the icon imports from lucide-react:

```tsx
import {
  Plus,
  RefreshCw,
  BookOpen,
  Cpu,
  TrendingUp,
  Activity,
  Sigma,
  ChevronDown,
  MessageCircle,
  Sparkles,
} from "lucide-react";
```

Add state variable alongside existing `showForm`:

```tsx
const [showForm, setShowForm] = useState(false);
const [showGuided, setShowGuided] = useState(false);
const [showDropdown, setShowDropdown] = useState(false);
```

- [ ] **Step 2: Replace the "New Topic" button with a split dropdown**

Replace the existing `<button onClick={() => setShowForm(true)} ...>New Topic</button>` (around lines 72-78) with:

```tsx
<div className="relative">
  <div className="flex">
    <button
      onClick={() => setShowForm(true)}
      className="flex items-center gap-2 px-4 py-2.5 bg-indigo-500 text-white rounded-l-lg hover:bg-indigo-400 transition-all text-sm font-medium shadow-lg shadow-indigo-500/25"
    >
      <Plus size={16} />
      New Topic
    </button>
    <button
      onClick={() => setShowDropdown(!showDropdown)}
      className="px-2 py-2.5 bg-indigo-500 text-white rounded-r-lg hover:bg-indigo-400 transition-all border-l border-indigo-400/50 shadow-lg shadow-indigo-500/25"
    >
      <ChevronDown size={14} />
    </button>
  </div>
  {showDropdown && (
    <div className="absolute right-0 mt-2 w-56 bg-white rounded-xl shadow-xl border border-slate-200 py-1 z-50">
      <button
        onClick={() => { setShowForm(true); setShowDropdown(false); }}
        className="w-full flex items-center gap-3 px-4 py-2.5 text-sm text-slate-700 hover:bg-slate-50 transition-colors"
      >
        <Sparkles size={16} className="text-indigo-500" />
        <div className="text-left">
          <div className="font-medium">Quick Create</div>
          <div className="text-xs text-slate-400">Name only, auto-generate config</div>
        </div>
      </button>
      <button
        onClick={() => { setShowGuided(true); setShowDropdown(false); }}
        className="w-full flex items-center gap-3 px-4 py-2.5 text-sm text-slate-700 hover:bg-slate-50 transition-colors"
      >
        <MessageCircle size={16} className="text-indigo-500" />
        <div className="text-left">
          <div className="font-medium">Guided Create</div>
          <div className="text-xs text-slate-400">Chat to refine your idea step by step</div>
        </div>
      </button>
    </div>
  )}
</div>
```

- [ ] **Step 3: Update the empty-state button**

Replace the "Create your first topic" button (around line 164-170) with two buttons:

```tsx
<div className="flex gap-3 justify-center">
  <button
    onClick={() => setShowForm(true)}
    className="inline-flex items-center gap-2 px-5 py-2.5 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors text-sm font-medium shadow-md shadow-indigo-500/20"
  >
    <Sparkles size={16} />
    Quick Create
  </button>
  <button
    onClick={() => setShowGuided(true)}
    className="inline-flex items-center gap-2 px-5 py-2.5 bg-white text-indigo-600 border border-indigo-200 rounded-lg hover:bg-indigo-50 transition-colors text-sm font-medium"
  >
    <MessageCircle size={16} />
    Guided Create
  </button>
</div>
```

- [ ] **Step 4: Add GuidedCreateModal render**

Replace the existing `{showForm && <TopicForm onClose={() => setShowForm(false)} />}` at the bottom (line 182) with:

```tsx
{showForm && <TopicForm onClose={() => setShowForm(false)} />}
{showGuided && <GuidedCreateModal onClose={() => setShowGuided(false)} />}
```

- [ ] **Step 5: Commit**

```bash
cd frontend && git add src/app/page.tsx
git commit -m "feat: add Guided Create dropdown entry on dashboard"
```

---

### Task 6: Manual smoke test

- [ ] **Step 1: Start backend and frontend**

```bash
cd /home/shurui/wkspace/codex-test/paper-tracker
uv run uvicorn paper_tracker.server:app --reload --port 8000 &
cd frontend && npm run dev &
```

- [ ] **Step 2: Test quick create (regression)**

Open http://localhost:3000. Click "New Topic" main button. Enter a name. Verify it still works as before.

- [ ] **Step 3: Test guided create**

1. Click the dropdown arrow next to "New Topic"
2. Select "Guided Create"
3. Type a vague description like "LLM在自我纠错上的问题"
4. Verify streaming text appears in chat
5. Continue the conversation, verify multi-turn works
6. When LLM proposes a config, verify modal switches to editable preview
7. Edit a field, click "Create Topic"
8. Verify topic appears on dashboard

- [ ] **Step 4: Test edge cases**

- Close modal mid-stream (should abort cleanly)
- Click "Back to Chat" from preview (should return to chat with history preserved)
- Empty input (send button should be disabled)
