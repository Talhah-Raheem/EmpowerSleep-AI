# CLAUDE.md - EmpowerSleep Technical Reference

This file contains technical details about the codebase for Claude Code.

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
- `POST /chat` - Original non-streaming endpoint (still works, backward compat)
- `POST /chat/stream` - SSE streaming endpoint (used by frontend)
- `GET /health` - Health check
- `GET /stats` - Index statistics
- CORS configured for localhost:3000

**`rag/chat_engine.py`** - Core RAG logic
- `ChatEngine` class - Main interface
- `ask_question(message, history)` - Sync, returns (answer, sources) — used by /chat
- `stream_question(message, history)` - Async generator for /chat/stream
- `_build_llm_messages(query, context, history)` - Shared by both paths
- `_async_client` property — lazy AsyncOpenAI instance
- Configuration constants at top of file
- System prompt enforces educational, non-diagnostic tone

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
- `chatContainerRef` + `userScrolledUpRef` — free-scroll during generation
- `onWheel` stops auto-scroll immediately on upward scroll; `onScroll` resumes when back at bottom
- Logo/title links to https://www.empowersleep.com/

**`frontend/components/`**
- `ChatMessage.tsx` - Message bubble; `streaming` prop shows blinking cursor; `onRegenerate` prop shows ↺ button
- `SourceCard.tsx` - Source citation display; blog/notion sources show EmpowerLogo + link; textbook shows static card
- `SleepLoader.tsx` - Branded loading animation (shows until first streaming token)

**`frontend/lib/`**
- `api.ts` - `sendMessage()` (non-streaming) + `streamMessage()` (SSE callbacks: onToken, onSources, onDone, onError)
- `sleepThoughts.ts` - Calm messages shown during loading
- `sampleQuestions.ts` - Rotating sample questions for welcome screen

**`frontend/app/globals.css`**
- `animate-blink` keyframe — used by streaming cursor in ChatMessage

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

**`scripts/ingest_notion_export.py`** ← NEW
- Ingests Notion markdown export from `data/raw/empower_sleep_notion/`
- Cleans Notion-exported markdown (strips image refs, flattens internal links)
- Chunks at ~400 words (smaller than blog — Notion pages are shorter)
- Source type: `"notion"` — displays with EmpowerLogo linking to empowersleep.com
- Run with `--rebuild` to replace previously ingested Notion content

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

## Data Flow

1. **Scraping**: `scrape_empowersleep_blog.py` → `data/blog_docs.jsonl`
2. **Indexing**: `build_blog_index.py` → `rag_artifacts/`
3. **Textbooks**: `ingest_textbook.py` → merges into `rag_artifacts/`
4. **Notion**: `ingest_notion_export.py` → merges into `rag_artifacts/`
5. **Serving**: Backend loads index, streams responses via `/chat/stream`

## Important Behaviors

### System Prompt
- **Non-diagnostic**: Never labels user with conditions
- Uses pattern-based language ("This is often associated with...")
- Asks clarifying questions when context is incomplete
- Maintains conversation continuity

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

## Dependencies

### Python (requirements.txt)
- `faiss-cpu` - Vector similarity search
- `openai` - Embeddings + LLM (sync + async)
- `fastapi` + `uvicorn` - Backend API
- `PyMuPDF` - PDF extraction
- `python-dotenv` - Environment variables

### Node.js (frontend/package.json)
- `next` - React framework
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

### Prerequisites
- Railway account (railway.app)
- GitHub repo with code pushed
- `rag_artifacts/` folder committed (contains FAISS index)

### Step 1: Create Railway Project
1. Go to railway.app → New Project → Deploy from GitHub
2. Select your repository
3. This creates the first service (backend by default)

### Step 2: Configure Backend Service
1. In Railway dashboard, click on the service
2. Go to Settings → change name to "backend"
3. Add environment variables:
   - `OPENAI_API_KEY` = your OpenAI key
   - `CORS_ORIGINS` = (leave empty for now, add frontend URL later)
4. Railway auto-detects Python from `requirements.txt` and uses `Procfile`
5. Deploy and copy the generated URL (e.g., `https://backend-xxx.up.railway.app`)

### Step 3: Add Frontend Service
1. In Railway project, click "New" → "GitHub Repo" → same repo
2. Go to Settings:
   - Change name to "frontend"
   - Set Root Directory to `frontend`
3. Add environment variable:
   - `NEXT_PUBLIC_API_BASE_URL` = backend URL from Step 2
4. Deploy and copy the frontend URL

### Step 4: Update Backend CORS
1. Go back to backend service in Railway
2. Add environment variable:
   - `CORS_ORIGINS` = frontend URL (e.g., `https://frontend-xxx.up.railway.app`)
3. Redeploy backend

### Environment Variables Summary

**Backend Service:**
| Variable | Value |
|----------|-------|
| `OPENAI_API_KEY` | sk-... |
| `CORS_ORIGINS` | https://frontend-xxx.up.railway.app |

**Frontend Service:**
| Variable | Value |
|----------|-------|
| `NEXT_PUBLIC_API_BASE_URL` | https://backend-xxx.up.railway.app |

### Custom Domain (Optional)
1. In Railway service settings → Domains
2. Add custom domain and configure DNS

## Planned Enhancements

### No Database Required
- [ ] Follow-up question suggestions after each answer
- [ ] Export conversation as PDF (client-side rendering)
- [ ] Guided sleep assessment questionnaire mode (hardcoded question flow)

### Requires Database
- [ ] Feedback buttons (thumbs up/down) with storage
- [ ] Shareable conversation links
- [ ] User accounts / saved conversation history

### Bigger Features
- [ ] Voice conversations (voice input + AI voice output, full conversational mode — do last)

## Structure for Growth

### Must-haves Before Going Public
- [ ] **Privacy Policy** page — required since user health-related questions are sent to OpenAI
- [ ] **Terms of Service** page — legally protect the "not medical advice" disclaimer
- [ ] **Cookie/consent banner** — needed if analytics are added

### Infrastructure
- [ ] **Database** (Supabase or Railway Postgres) — unlocks feedback storage, conversation history, shareable links, analytics
- [ ] **Auth** (even optional/anonymous) — for user accounts, saved chats, usage limits
- [ ] **Error monitoring** (Sentry free tier) — know when things break in production
- [ ] **Analytics** (PostHog or Plausible free tier) — understand what users actually ask
