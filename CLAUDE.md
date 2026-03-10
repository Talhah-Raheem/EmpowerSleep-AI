# CLAUDE.md - EmpowerSleep Technical Reference

This file contains technical details about the codebase for Claude Code.

## Rules for Claude

- **Never commit or push to GitHub directly.** Claude must only provide the git commands for the user to run themselves. This applies to all git operations: commits, pushes, force-pushes, merges, rebases, and PRs.

## Architecture Overview

This is a **RAG chatbot** for sleep education with a split architecture:
- **Backend**: FastAPI serving the RAG pipeline
- **Frontend**: Next.js (App Router) with a modern chat UI

```
User Question
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│              frontend (Next.js @ :3000)                 │
│                                                         │
│  1. User types question in chat UI                      │
│  2. POST /chat/stream to backend (SSE streaming)        │
│  3. Tokens stream in live; sources attach on done       │
│  4. POST /suggestions after done → follow-up chips      │
└─────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│              backend (FastAPI @ :8000)                  │
│                                                         │
│  1. ChatEngine.stream_question()                        │
│  2. embed_query() ──► OpenAI text-embedding-3-small    │
│  3. retrieve_chunks() ──► FAISS search                 │
│  4. yield sources (before generation)                   │
│  5. AsyncOpenAI stream ──► GPT-4o-mini tokens          │
│  6. yield DISCLAIMER_SUFFIX as final token              │
└─────────────────────────────────────────────────────────┘
```

## Key Files

### Backend

**`backend/main.py`** - FastAPI application
- `POST /chat` - Original non-streaming endpoint (backward compat)
- `POST /chat/stream` - SSE streaming endpoint (used by frontend)
- `POST /suggestions` - Returns 3 LLM-generated follow-up questions after each response
- `POST /feedback` - Stores thumbs up/down in Supabase with session_id, rating, environment tag
- `GET /health` - Health check
- `GET /stats` - Index statistics broken down by source type
- CORS configured for localhost:3000, empowersleep.com domains, and `CORS_ORIGINS` env var

**`rag/chat_engine.py`** - Core RAG logic
- `ChatEngine` class - Main interface
- `ask_question(message, history)` - Sync, returns (answer, sources) — used by /chat
- `stream_question(message, history)` - Async generator for /chat/stream; timeout=30s
- `_build_search_query()` - Smart follow-up detection:
  - `AFFIRMATIONS` set — pure "yes/sure/ok" messages search ONLY the last assistant question (not combined with original topic, to avoid diluting the embedding)
  - Short (≤20 words): combines original question + last assistant question
  - Medium (21–40 words) with FOLLOWUP_SIGNALS: combines original + current
  - Long (>40 words): uses current message as-is
- `_build_llm_messages()` - Injects `AFFIRMATION DETECTED` instruction when current message is a pure affirmation
- `AFFIRMATIONS` constant — module-level set of affirmation words
- `FOLLOWUP_SIGNALS` constant — module-level set of follow-up signal words
- System prompt: non-diagnostic, pattern-based language, strict conversation binding, only offers follow-up questions on topics covered in retrieved context
- Configuration constants at top of file

**`rag/ingestion/textbook_ingestor.py`** - PDF ingestion
- Extracts text from PDFs using PyMuPDF
- Detects chapters, removes headers/footers
- Creates chunks with page tracking

### Frontend

**`frontend/app/page.tsx`** - Main chat page
- `isStreaming` state — true while tokens are arriving
- `runStream()` — shared streaming logic called by both handleSubmit and handleRegenerate
- `handleRegenerate(index)` — drops assistant message at index and re-streams
- `pendingSourcesRef` — holds sources until streaming completes
- `assistantAddedRef` — tracks whether assistant message was actually added (prevents user message deletion on early stream error)
- `suggestionsAbortRef` — AbortController for in-flight suggestions fetch; aborted on new chat or new stream start
- `chatContainerRef` + `userScrolledUpRef` — free-scroll during generation
- `onWheel` stops auto-scroll immediately on upward scroll; `onScroll` resumes when within 30px of bottom
- Hero landing page (no messages): night sky / sunrise gradient, stars, centered input + sample questions
- Chat view (messages exist): same gradient persists, frosted glass header + footer
- Follow-up suggestion chips appear below messages after each response; clicking pre-fills input

**`frontend/components/`**
- `ChatMessage.tsx` - Message bubble; `streaming` prop shows blinking cursor; `onRegenerate` prop shows ↺ button; feedback thumbs up/down; assistant bubbles are full-width frosted glass
- `SourceCard.tsx` - Source citation display; blog/notion sources show EmpowerLogo + link; textbook shows static card
- `SleepLoader.tsx` - Branded loading animation (shows until first streaming token)
- `StarField.tsx` - 120 twinkling stars + shooting star every 12–20s (dark mode only); both timers cleaned up on unmount with `mountedRef` guard

**`frontend/lib/`**
- `api.ts` - `sendMessage()` (non-streaming) + `streamMessage()` (SSE) + `getSuggestions(message, response, history, signal?)` + `submitFeedback()`
- `sleepThoughts.ts` - Calm messages shown during loading
- `sampleQuestions.ts` - 33 rotating sample questions: general sleep education + EmpowerSleep program questions (Foundation, Optimization, Longevity, pricing, testing philosophy)

**`frontend/app/globals.css`**
- `animate-blink` keyframe — streaming cursor
- `twinkle` / `shoot` keyframes — star animations
- `.input-sky:focus` — blue glow (dark mode), amber glow (light mode)

### Scripts

**`scripts/scrape_empowersleep_blog.py`**
- Scrapes articles from empowersleep.com/blog
- Output: `data/blog_docs.jsonl`

**`scripts/build_blog_index.py`**
- Chunks articles (~1000 words, 150 word overlap)
- Generates embeddings with OpenAI
- Builds FAISS index
- Output: `rag_artifacts/`

**`scripts/ingest_textbook.py`**
- CLI for ingesting PDF textbooks
- Merges with existing index
- Tracks via manifest for idempotency

**`scripts/ingest_notion_export.py`**
- Ingests Notion markdown export from `data/raw/empower_sleep_notion/`
- Cleans Notion-exported markdown (strips image refs, flattens internal links)
- Chunks at ~400 words (smaller than blog — Notion pages are shorter)
- Source type: `"notion"` — displays with EmpowerLogo linking to empowersleep.com
- Run with `--rebuild` to replace previously ingested Notion content

## RAG Index State

| Source | Chunks | Notes |
|--------|--------|-------|
| Blog | 148 | No `source` field — identified by `url` presence, defaults to "blog" |
| Textbook | 154 | `source: "textbook"` — Sleep_And_Health.pdf |
| Notion | 40 | `source: "notion"` — 12 pages from empower_sleep_notion/ |
| **Total** | **342** | FAISS IndexFlatIP (cosine similarity) |

## Configuration

All config is in `rag/chat_engine.py`:

| Setting | Value |
|---------|-------|
| Embedding Model | `text-embedding-3-small` |
| Embedding Dim | 1536 |
| Top-K Results | 6 |
| LLM Model | `gpt-4o-mini` |
| LLM Temperature | 0.3 |
| Max Tokens | 750 |
| Stream Timeout | 30s |
| Suggestions Timeout | 10s |
| Max History Turns | 3 |

## Data Flow

1. **Scraping**: `scrape_empowersleep_blog.py` → `data/blog_docs.jsonl`
2. **Indexing**: `build_blog_index.py` → `rag_artifacts/`
3. **Textbooks**: `ingest_textbook.py` → merges into `rag_artifacts/`
4. **Notion**: `ingest_notion_export.py` → merges into `rag_artifacts/`
5. **Serving**: Backend loads index, streams responses via `/chat/stream`

## Important Behaviors

### System Prompt
- **Non-diagnostic**: Never labels user with conditions
- Pattern-based language ("This is often associated with...")
- Strict conversation binding: "yes/sure/ok" = affirm last question; must deliver exactly what was offered
- Only offers follow-up questions on topics present in the retrieved context
- SYMPTOM–MECHANISM ALIGNMENT: matches explanation direction to what user described

### Source Types
- **Blog**: `{source_type: "blog", title, url}` — EmpowerLogo + link
- **Textbook**: `{source_type: "textbook", title, chapter, page_start, page_end}` — static card
- **Notion**: `{source_type: "notion", title, url: "https://www.empowersleep.com/"}` — EmpowerLogo + link to site

### Streaming SSE Format (backend → frontend)
```
data: {"type": "sources", "sources": [...]}
data: {"type": "token", "text": "Sleep "}
data: {"type": "token", "text": "hygiene "}
data: [DONE]
```
Sources are sent before the first token. Disclaimer is yielded as the final token.

### Follow-up Suggestions
- After `[DONE]`, frontend calls `POST /suggestions` with the user message + history
- Backend sends to GPT-4o-mini: returns 3 short contextual questions
- Rendered as clickable chips below the last message
- Fetch is aborted if user starts a new chat or sends another message
- Suggestions timeout: 10s on backend

### Feedback
- Thumbs up (1) / thumbs down (-1) stored in Supabase `feedback` table
- Row includes: session_id, user_question, ai_response, rating, environment (dev/prod via `APP_ENV`)

## Dependencies

### Python (requirements.txt)
- `faiss-cpu` - Vector similarity search
- `openai` - Embeddings + LLM (sync + async)
- `fastapi` + `uvicorn` - Backend API
- `PyMuPDF` - PDF extraction
- `supabase` - Feedback storage
- `python-dotenv` - Environment variables

### Node.js (frontend/package.json)
- `next` 14.1.0 - React framework
- `next-themes` - Dark mode provider
- `react-markdown` - Markdown rendering
- `tailwindcss` - Styling

## Running Locally

```bash
# Terminal 1: Backend
source venv/bin/activate
python -m uvicorn backend.main:app --reload --port 8000

# Terminal 2: Frontend
cd frontend
npm install
npm run dev
```

Open http://localhost:3000

## Environment Variables

**Backend (.env)**
```
OPENAI_API_KEY=sk-...
SUPABASE_URL=https://...
SUPABASE_SECRET_KEY=...
APP_ENV=development
```

**Frontend (frontend/.env.local)**
```
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

## Common Tasks

### Rebuild index after blog content changes
```bash
python scripts/scrape_empowersleep_blog.py
python scripts/build_blog_index.py
```

### Add a textbook
```bash
python scripts/ingest_textbook.py --pdf data/raw/Book.pdf --book-title "Book Name"
```

### Add / update Notion content
```bash
# First time
python scripts/ingest_notion_export.py

# After re-exporting from Notion
python scripts/ingest_notion_export.py --rebuild
```
Notion export lives at: `data/raw/empower_sleep_notion/`

### Modify retrieval behavior
Edit `TOP_K_RESULTS` in `rag/chat_engine.py`

### Modify LLM behavior
Edit `SYSTEM_PROMPT` in `rag/chat_engine.py`

## Deployment (Railway)

Both frontend and backend are deployed on Railway as separate services.

**Backend Service env vars:**
| Variable | Value |
|----------|-------|
| `OPENAI_API_KEY` | sk-... |
| `SUPABASE_URL` | https://... |
| `SUPABASE_SECRET_KEY` | ... |
| `CORS_ORIGINS` | https://frontend-xxx.up.railway.app |
| `APP_ENV` | production |

**Frontend Service env vars:**
| Variable | Value |
|----------|-------|
| `NEXT_PUBLIC_API_BASE_URL` | https://backend-xxx.up.railway.app |

## Planned / Next Steps

### Done ✓
- [x] SSE streaming with sources-first delivery
- [x] Notion ingestion (programs, pricing, team, FAQs)
- [x] Feedback buttons (thumbs up/down) → Supabase
- [x] Follow-up question suggestions (LLM-generated chips after each response)
- [x] Night sky / sunrise hero landing page with StarField
- [x] Full-width frosted glass assistant bubbles
- [x] Terms & Privacy links in disclaimer footer
- [x] Smart affirmation detection (pure "yes" searches only the offered topic)
- [x] AI only offers follow-ups on topics it has context for

### Up Next
- [ ] **Cookie/consent banner** — needed before adding analytics
- [ ] **Analytics** (PostHog free tier) — understand what users ask
- [ ] **Error monitoring** (Sentry free tier) — catch production errors
- [ ] **Refresh blog index** — scrape + rebuild since last run was March 2026
- [ ] **Guided sleep assessment mode** — hardcoded question flow for exploratory users

### Bigger Features
- [ ] Shareable conversation links (requires DB work)
- [ ] User accounts / saved conversation history
- [ ] Voice conversations (do last)
