# Research Tracker

AI-powered research paper tracker that automates literature discovery, brainstorming, and research planning across multiple academic sources.

## What It Does

- **Multi-source paper search** — Automatically fetches papers from arXiv, OpenAlex, and OpenReview with cross-source deduplication
- **Structured summarization** — Extracts key insights, methods, contributions, math concepts, and venue info from each paper
- **Research insights** — Cross-paper analysis identifying research gaps and opportunities
- **Brainstorm pipeline** — Multi-stage idea generation with dual-model (Claude + Codex) agreement, novelty screening, and prior art checking
- **Research plan generation** — Full research proposals with multi-round review and refinement
- **Discovery feeds** — Trending papers, math-focused insights, and community discussions
- **Translation** — On-demand translation for paper summaries

## Architecture

```
┌─────────────────────────────────────────────┐
│              Next.js Frontend               │
│         (React 18 + TanStack Query)         │
└──────────────────┬──────────────────────────┘
                   │ REST API
┌──────────────────▼──────────────────────────┐
│            FastAPI Backend (:8000)           │
├─────────────┬───────────┬───────────────────┤
│  Scheduler  │  Pipeline │    Brainstorm     │
│  (APSched)  │  (search  │  (idea gen +      │
│             │  + summ)  │   review + plan)  │
├─────────────┴───────────┴───────────────────┤
│  Sources: arXiv | OpenAlex | OpenReview |   │
│           GitHub | Web (Brave/HN)           │
├─────────────────────────────────────────────┤
│  LLM: Claude (opus/sonnet) | Codex | Copilot│
├─────────────────────────────────────────────┤
│  Storage: SQLite (registry.db + tracker.db) │
└─────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python >= 3.14 with [uv](https://docs.astral.sh/uv/)
- Node.js >= 18
- [Claude CLI](https://docs.anthropic.com/en/docs/claude-code) installed and authenticated
- (Optional) [Codex CLI](https://github.com/openai/codex) for dual-model brainstorming

### Setup

```bash
# Clone
git clone https://github.com/ZhangShuui/research-tracker.git
cd research-tracker

# Backend
uv sync
cp config.toml.example config.toml  # edit with your settings

# Frontend
cd frontend
npm install
```

### Run

```bash
# Terminal 1 — Backend
uv run uvicorn paper_tracker.server:app --reload --port 8000

# Terminal 2 — Frontend
cd frontend && npm run dev
```

Open http://localhost:3000

## Usage

### 1. Create a Topic

Just enter a topic name (e.g., "Video Generation with Diffusion Models") — the system auto-generates search keywords, arXiv categories, and description via LLM.

### 2. Run a Search

Click "Run" to fetch papers from all configured sources. Papers are deduplicated across sources, summarized with structured extraction, and quality-filtered.

### 3. Brainstorm

The brainstorm pipeline:
1. Loads configurable context (insights, reports, GitHub repos, brainstorm history, research plans)
2. Generates research questions (dual-model agreement)
3. Generates ideas with novelty prescreening
4. Multi-round review and refinement
5. Prior art checking against the paper library
6. Optional code proof-of-concept verification

### 4. Research Plan

Select a promising idea to generate a full research proposal with introduction, related work, methodology, experimental design, expected results, and timeline. Supports iterative refinement with feedback.

## Configuration

Copy `config.toml.example` to `config.toml`:

```toml
[search]
arxiv_lookback_days = 365
github_lookback_days = 7

[summarizer]
claude_path = "claude"
claude_model = "opus"
codex_path = "codex"

[notify.toast]
enabled = true

[paths]
data_dir = "data"
```

### Paper Sources

Each topic can enable/disable sources independently:

| Source | API Key Required | Notes |
|--------|-----------------|-------|
| arXiv | No | Primary source, keyword + category search |
| OpenAlex | No | 250M+ works, venue filtering |
| OpenReview | No | Conference papers (ICLR, NeurIPS, ICML, etc.) |
| GitHub | No | Related repositories and implementations |

## Project Structure

```
src/paper_tracker/
├── server.py          # FastAPI REST API
├── main.py            # Search pipeline (parallel multi-source)
├── brainstorm.py      # Multi-stage brainstorm pipeline
├── research_plan.py   # Research proposal generation
├── summarizer.py      # Structured paper extraction
├── insights.py        # Cross-paper analysis
├── discovery.py       # Trending/math discovery feeds
├── registry.py        # Topics + sessions DB (registry.db)
├── storage.py         # Papers + repos DB (tracker.db)
├── llm.py             # Shared LLM interface (Claude/Codex/Copilot)
├── config.py          # Configuration builder
├── scheduler.py       # Cron-based scheduling
├── sources/
│   ├── arxiv.py
│   ├── openalex.py
│   ├── openreview_api.py
│   ├── github.py
│   └── web.py         # Brave Search + HackerNews
└── notifiers/
    ├── email.py
    └── toast.py

frontend/src/
├── app/
│   ├── page.tsx                        # Dashboard
│   ├── topics/[id]/
│   │   ├── page.tsx                    # Topic overview
│   │   ├── papers/page.tsx             # Paper library
│   │   ├── insights/page.tsx           # Research insights
│   │   ├── brainstorm/page.tsx         # Brainstorm sessions
│   │   └── research-plan/page.tsx      # Research plans
│   ├── discovery/
│   │   ├── trending/page.tsx
│   │   ├── math-insights/page.tsx
│   │   └── community/page.tsx
│   └── usage/page.tsx                  # API usage stats
├── components/                         # Reusable UI components
└── lib/
    ├── api.ts                          # API client + types
    └── hooks.ts
```

## Tech Stack

**Backend**: Python 3.14, FastAPI, SQLite (WAL mode), APScheduler, httpx

**Frontend**: Next.js 14, React 18, TypeScript, Tailwind CSS, TanStack Query, KaTeX, Recharts

**LLM**: Claude CLI (opus/sonnet), Codex CLI, Copilot CLI

## License

MIT
