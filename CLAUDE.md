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
│  2. POST /chat to backend                               │
│  3. Display answer + sources                            │
└─────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│              backend (FastAPI @ :8000)                  │
│                                                         │
│  1. ChatEngine.ask_question()                           │
│  2. embed_query() ──► OpenAI text-embedding-3-small    │
│  3. retrieve_chunks() ──► FAISS search                 │
│  4. generate_answer() ──► GPT-4o-mini with context     │
│  5. Return JSON { answer, sources }                     │
└─────────────────────────────────────────────────────────┘
```

## Key Files

### Backend

**`backend/main.py`** - FastAPI application
- `POST /chat` - Main chat endpoint
- `GET /health` - Health check
- `GET /stats` - Index statistics
- CORS configured for localhost:3000

**`rag/chat_engine.py`** - Core RAG logic
- `ChatEngine` class - Main interface
- `ask_question(message, history)` - Returns (answer, sources)
- Configuration constants at top of file
- System prompt enforces educational, non-diagnostic tone

**`rag/ingestion/textbook_ingestor.py`** - PDF ingestion
- Extracts text from PDFs using PyMuPDF
- Detects chapters, removes headers/footers
- Creates chunks with page tracking

### Frontend

**`frontend/app/page.tsx`** - Main chat page
- Chat interface with message bubbles
- Calls `/chat` endpoint
- Displays sources (textbook with pages, blog with links)

**`frontend/components/`**
- `ChatMessage.tsx` - Message bubble component
- `SourceCard.tsx` - Source citation display
- `SleepLoader.tsx` - Branded loading animation with sleep thoughts

**`frontend/lib/`**
- `api.ts` - API client for backend (supports request cancellation)
- `sleepThoughts.ts` - Calm messages shown during loading
- `sampleQuestions.ts` - Rotating sample questions for welcome screen

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

## Configuration

All config is in `rag/chat_engine.py`:

| Setting | Value |
|---------|-------|
| Embedding Model | `text-embedding-3-small` |
| Embedding Dim | 1536 |
| Top-K Results | 4 |
| LLM Model | `gpt-4o-mini` |
| LLM Temperature | 0.3 |
| Max Tokens | 600 |

## Data Flow

1. **Scraping**: `scrape_empowersleep_blog.py` → `data/blog_docs.jsonl`
2. **Indexing**: `build_blog_index.py` → `rag_artifacts/`
3. **Textbooks**: `ingest_textbook.py` → merges into `rag_artifacts/`
4. **Serving**: Backend loads index, handles queries via `/chat`

## Important Behaviors

### System Prompt
- **Non-diagnostic**: Never labels user with conditions
- Uses pattern-based language ("This is often associated with...")
- Asks clarifying questions when context is incomplete
- Maintains conversation continuity

### Source Types
- **Blog**: `{source_type: "blog", title, url}`
- **Textbook**: `{source_type: "textbook", title, chapter, page_start, page_end}`

## Dependencies

### Python (requirements.txt)
- `faiss-cpu` - Vector similarity search
- `openai` - Embeddings + LLM
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

### Rebuild index after content changes
```bash
python scripts/scrape_empowersleep_blog.py
python scripts/build_blog_index.py
```

### Add a textbook
```bash
python scripts/ingest_textbook.py --pdf data/raw/Book.pdf --book-title "Book Name"
```

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

## Completed Enhancements
- [x] Add favicon (EmpowerSleep logo in layout.tsx)
- [x] Branded loading animation (SleepLoader with zzz + sleep thoughts)
- [x] Smooth scrolling (scrollIntoView in page.tsx)

## Planned Enhancements

### UI & UX
- [ ] Copy button on assistant messages
- [ ] Feedback buttons (thumbs up/down) on responses
- [ ] Follow-up question suggestions after each answer
- [ ] Keyboard shortcuts (Shift+Enter for newline)
- [ ] Dark mode support

### Export & Sharing
- [ ] Export conversation as PDF
- [ ] Shareable conversation links

### Bigger Features
- [ ] Guided sleep assessment questionnaire mode
- [ ] Voice conversations (voice input + AI voice output, full conversational mode — do last)
