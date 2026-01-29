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
- `SleepLoader.tsx` - Branded loading animation

**`frontend/lib/api.ts`** - API client for backend

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
