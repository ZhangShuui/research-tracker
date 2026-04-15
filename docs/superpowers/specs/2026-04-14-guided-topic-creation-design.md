# Guided Topic Creation — Design Spec

## Problem

Users sometimes don't know how to articulate a research topic precisely. The existing quick-create flow (name only) works when you already have a clear topic in mind, but fails when the idea is vague. We need a conversational flow that helps users refine a fuzzy idea into a well-defined topic configuration.

## Solution

Two entry points on the Dashboard for creating topics:

1. **Quick Create** (existing) — name input, LLM auto-generates config
2. **Guided Create** (new) — multi-turn chat with LLM to iteratively refine topic name, description, and keywords, then present a full editable config for final confirmation

## Backend

### New endpoint: `POST /api/topics/guided`

SSE streaming endpoint for one turn of the guided conversation.

**Request body:**
```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "..."}
  ]
}
```

`messages` contains the full conversation history. The backend reconstructs context each call — no server-side session state.

**Implementation:**

1. Build a prompt from the conversation history, prepended with a system prompt that instructs the LLM to:
   - Act as a research topic configuration assistant
   - Ask 1-2 clarifying questions to understand the user's vague idea
   - When confident, propose a topic name, description, and keyword set as a JSON block
   - After user confirms/adjusts, output a final `draft_config` JSON containing all fields needed by `TopicCreate`
2. Spawn a Claude CLI subprocess:
   ```
   claude -p --output-format stream-json --verbose --include-partial-messages \
     --model sonnet --system-prompt "..." --bare
   ```
   Feed the constructed prompt via stdin.
3. Read stdout line by line. For each JSON line:
   - `{"type": "assistant", ...}` with partial content → forward as SSE `event: delta`
   - `{"type": "result", ...}` → parse the full response text:
     - If it contains a fenced JSON block with a `name` + `arxiv_keywords` → extract as `draft_config`, emit SSE `event: done` with `stage: "ready"` and `draft_config`
     - Otherwise → emit SSE `event: done` with `stage: "chatting"`

**SSE event format:**
```
event: delta
data: {"text": "partial token text"}

event: done
data: {"message": "full reply", "stage": "chatting"|"ready", "draft_config": null|{...}}
```

**System prompt responsibilities:**
- Guide the user through refining their idea
- Ask about: scope/angle, what makes it distinct, whether it spans subfields
- Propose topic name + description + keywords in a structured JSON block when ready
- The JSON block must contain: `name`, `description`, `arxiv_keywords`, `arxiv_categories`, `github_keywords`
- Remaining fields (lookback days, source toggles, schedule) are filled with defaults by the backend

**`--bare` flag** is used to skip hooks, LSP, plugin sync, and other overhead — this is a headless LLM call, not an interactive session.

### Config completion

When `draft_config` is extracted from the LLM output, the backend fills in defaults for fields the LLM didn't specify:
- `arxiv_lookback_days`: 365
- `github_lookback_days`: 365
- `search_date_from` / `search_date_to`: 1 year ago to today
- `openalex_enabled`: true, reusing first 4 arxiv keywords
- `openreview_enabled`: true, reusing first 4 arxiv keywords
- `schedule_cron`: empty

This mirrors the existing `quick_create_topic` defaults.

### Topic creation

The actual topic creation still uses the existing `POST /api/topics` endpoint. The guided flow only helps produce the config — creation is a separate step triggered by the frontend after the user confirms the editable preview.

## Frontend

### Dashboard entry point

Replace the current "New Topic" button with a split button / dropdown:
- **Quick Create** — opens existing `TopicForm` modal (name-only quick create)
- **Guided Create** — opens `GuidedCreateModal`

Same treatment for the "Create your first topic" empty-state button.

### GuidedCreateModal component

A large modal/drawer with two phases:

**Phase 1: Chat**
- Message list displaying the conversation (user bubbles right-aligned, assistant bubbles left-aligned)
- Assistant messages stream in token-by-token via SSE `delta` events
- Input bar at the bottom with send button
- When the LLM returns a structured suggestion (keywords list etc.), render as selectable tag/chip components inline in the assistant message — user can add/remove/edit tags before sending their next message
- Frontend maintains `messages: {role, content}[]` array, sends full history on each turn

**Phase 2: Editable Preview**
- Triggered when SSE returns `stage: "ready"` with `draft_config`
- Chat area collapses or fades to background
- Full topic configuration form appears (reuse TopicForm's edit-mode field layout):
  - Name, Description
  - arXiv Keywords (textarea, one per line)
  - arXiv Categories
  - Date range (from/to)
  - Lookback days (arXiv, GitHub)
  - GitHub Keywords
  - Additional Sources (OpenAlex, OpenReview) — collapsible section
  - Schedule Cron
- Two action buttons:
  - **Back to Chat** — return to Phase 1 to continue refining via conversation
  - **Create Topic** — call `POST /api/topics` with the form data, then close modal and refresh dashboard

### API client addition

New function in `api.ts`:
```typescript
function guidedCreateStream(
  messages: {role: string, content: string}[]
): EventSource | ReadableStream
```

Uses `fetch()` to POST to `/api/topics/guided` and reads the SSE stream. Returns parsed events to the component.

### Structured choice rendering

When the assistant proposes keywords/categories in a JSON block during the chat phase (before `stage: "ready"`), the frontend detects the JSON in the message and renders it as interactive chips instead of raw text. This allows the user to toggle keywords on/off, add new ones, or edit inline — then their next message includes the modified selection.

Detection: look for a fenced ```json block in the assistant message text containing `arxiv_keywords` or similar fields.

## Data flow summary

```
User types vague idea
  → Frontend sends {messages} to POST /api/topics/guided
  → Backend spawns claude CLI with system prompt + history
  → Claude streams response
  → Backend forwards as SSE (delta + done events)
  → Frontend renders streamed text + structured choices
  → User refines, sends again (loop)
  → Eventually LLM outputs draft_config, stage="ready"
  → Frontend shows editable preview form
  → User edits and clicks "Create Topic"
  → Frontend calls POST /api/topics with final config
  → Topic created, modal closes, dashboard refreshes
```

## Non-goals

- No persistent server-side conversation state — frontend owns the message history
- No conversation saving/resuming — guided creation is ephemeral
- No streaming input (`--input-format stream-json`) — each turn is an independent CLI process
- No changes to existing quick-create flow
