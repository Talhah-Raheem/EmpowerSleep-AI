# EmpowerSleep

AI-powered sleep education chatbot for [empowersleep.com](https://www.empowersleep.com). Answers sleep questions grounded in expert content from the EmpowerSleep blog, textbooks, and program materials.

**Live at**: https://www.empowersleep.com

---

## Architecture

- **Backend**: FastAPI + RAG pipeline (Python) — deployed on Railway
- **Frontend**: Next.js App Router — deployed on Railway
- **Vector store**: FAISS (cosine similarity, `text-embedding-3-small`)
- **LLM**: GPT-4o-mini via OpenAI
- **Database**: Supabase (feedback storage)

```
User → Next.js frontend → FastAPI backend → FAISS retrieval → GPT-4o-mini → SSE stream → User
```

## Project Structure

```
EMPOWERSLEEP/
├── backend/
│   └── main.py                    # FastAPI app (chat, feedback, health, stats endpoints)
├── frontend/
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx               # Main chat UI
│   │   └── globals.css
│   ├── components/
│   │   ├── ChatMessage.tsx        # Message bubbles with feedback buttons
│   │   ├── SourceCard.tsx         # Source citations
│   │   ├── SleepLoader.tsx        # Branded loading animation
│   │   └── EmpowerLogo.tsx        # SVG logo
│   └── lib/
│       ├── api.ts                 # API client (streaming + feedback)
│       ├── sleepThoughts.ts       # Loading screen messages
│       └── sampleQuestions.ts     # Welcome screen questions
├── rag/
│   ├── chat_engine.py             # Core RAG logic
│   └── ingestion/
│       └── textbook_ingestor.py
├── scripts/
│   ├── scrape_empowersleep_blog.py
│   ├── build_blog_index.py
│   ├── ingest_textbook.py
│   └── ingest_notion_export.py
├── data/
│   ├── blog_docs.jsonl
│   └── raw/                       # PDFs + Notion export
├── rag_artifacts/                 # FAISS index (committed)
└── requirements.txt
```

## Local Development

```bash
# Terminal 1 — backend
source venv/bin/activate
python -m uvicorn backend.main:app --reload --port 8000

# Terminal 2 — frontend
cd frontend && npm run dev
```

Make sure `frontend/.env.local` has `NEXT_PUBLIC_API_BASE_URL=http://localhost:8000` and `.env` has `APP_ENV=development`.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/chat` | Non-streaming chat (returns full response) |
| `POST` | `/chat/stream` | SSE streaming chat (used by frontend) |
| `POST` | `/feedback` | Submit thumbs up/down feedback |
| `GET` | `/health` | Health check |
| `GET` | `/stats` | Knowledge base statistics |

### SSE streaming format

```
data: {"type": "sources", "sources": [...]}
data: {"type": "token", "text": "Sleep "}
data: {"type": "token", "text": "hygiene "}
data: [DONE]
```

## Knowledge Base

The RAG index combines three content sources:

| Source | Script | Output |
|--------|--------|--------|
| EmpowerSleep blog | `scrape_empowersleep_blog.py` + `build_blog_index.py` | `data/blog_docs.jsonl` |
| PDF textbooks | `ingest_textbook.py` | merged into `rag_artifacts/` |
| Notion export | `ingest_notion_export.py` | merged into `rag_artifacts/` |

### Rebuild index after content changes

```bash
# Blog
python scripts/scrape_empowersleep_blog.py
python scripts/build_blog_index.py

# Add a textbook
python scripts/ingest_textbook.py --pdf data/raw/Book.pdf --book-title "Book Title"

# Notion (re-export from Notion first, place in data/raw/empower_sleep_notion/)
python scripts/ingest_notion_export.py --rebuild
```

## Deployment (Railway)

Both services are deployed on Railway from this repository. See `CLAUDE.md` for full deployment steps.

**Backend env vars on Railway:**
| Variable | Value |
|----------|-------|
| `OPENAI_API_KEY` | sk-... |
| `SUPABASE_URL` | https://xxxx.supabase.co |
| `SUPABASE_SECRET_KEY` | service role key |
| `CORS_ORIGINS` | https://frontend-url.up.railway.app |

**Frontend env vars on Railway:**
| Variable | Value |
|----------|-------|
| `NEXT_PUBLIC_API_BASE_URL` | https://backend-url.up.railway.app |

> `APP_ENV` is intentionally not set on Railway — the backend defaults to `"production"`.

---

*Built with FastAPI · Next.js · FAISS · OpenAI · Supabase*
