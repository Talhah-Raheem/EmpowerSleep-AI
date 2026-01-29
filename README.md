# EmpowerSleep

A sleep education chatbot powered by RAG (Retrieval-Augmented Generation) using content from the EmpowerSleep blog and textbooks.

## Architecture

The application uses a modern split architecture:
- **Backend**: FastAPI serving the RAG pipeline
- **Frontend**: Next.js (App Router) with a modern chat UI

## Quick Start

### 1. Setup Python Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Add your OpenAI API key
echo "OPENAI_API_KEY=sk-..." > .env
```

### 3. Build the Index

```bash
# Scrape blog content and build index
python scripts/scrape_empowersleep_blog.py
python scripts/build_blog_index.py

# (Optional) Add a textbook
python scripts/ingest_textbook.py \
    --pdf data/raw/Sleep_And_Health.pdf \
    --book-title "Sleep and Health"
```

### 4. Run the Application

```bash
# Terminal 1: Start the backend
python -m uvicorn backend.main:app --reload --port 8000

# Terminal 2: Start the frontend
cd frontend
npm install
npm run dev
```

Then open http://localhost:3000

## Project Structure

```
EMPOWERSLEEP/
├── backend/
│   └── main.py                 # FastAPI backend
├── frontend/
│   ├── app/
│   │   ├── layout.tsx          # Root layout
│   │   ├── page.tsx            # Chat page
│   │   └── globals.css         # Styles
│   ├── components/
│   │   ├── ChatMessage.tsx     # Message bubbles
│   │   └── SourceCard.tsx      # Source citations
│   ├── lib/
│   │   └── api.ts              # API client
│   └── package.json
├── rag/
│   ├── chat_engine.py          # Core RAG logic (used by backend)
│   └── ingestion/
│       └── textbook_ingestor.py
├── scripts/
│   ├── scrape_empowersleep_blog.py
│   ├── build_blog_index.py
│   └── ingest_textbook.py
├── data/
│   ├── blog_docs.jsonl
│   └── raw/                    # PDF textbooks
├── rag_artifacts/              # FAISS index + chunks
└── requirements.txt
```

## API Endpoints

### POST /chat

Send a message and get a response with sources.

**Request:**
```json
{
  "message": "What is sleep hygiene?",
  "history": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

**Response:**
```json
{
  "answer": "Sleep hygiene refers to...",
  "sources": [
    {
      "source_type": "textbook",
      "title": "Sleep and Health",
      "chapter": "Chapter 3: Sleep Hygiene",
      "page_start": 45,
      "page_end": 48
    },
    {
      "source_type": "blog",
      "title": "5 Tips for Better Sleep",
      "url": "https://empowersleep.com/..."
    }
  ]
}
```

### GET /health

Health check endpoint.

### GET /stats

Get knowledge base statistics.

## Textbook Ingestion

Add PDF textbooks to enhance the knowledge base:

```bash
python scripts/ingest_textbook.py \
    --pdf data/raw/YourTextbook.pdf \
    --book-title "Your Book Title" \
    --version v1
```

**Options:**
- `--pdf`: Path to PDF file
- `--book-title`: Display title
- `--version`: Version string (change to force re-index)
- `--rebuild`: Force re-processing

**Smoke Test:**
```bash
python scripts/ingest_textbook.py --smoke-test "What is REM sleep?"
```

## Source Citations

The chat displays sources differently based on type:

- **Textbook**: 📖 **Sleep and Health** – Chapter 3 (pp. 45–48)
- **Blog**: [Article Title](https://empowersleep.com/...)

## How It Works

1. User asks a sleep-related question
2. Question is embedded using OpenAI text-embedding-3-small
3. FAISS retrieves the most relevant chunks (blog + textbook)
4. GPT-4o-mini generates a grounded, educational answer
5. Sources are cited with page numbers (textbooks) or links (blog)

## Requirements

- Python 3.9+
- Node.js 18+ (for frontend)
- OpenAI API key

## Environment Variables

**Backend (.env):**
```
OPENAI_API_KEY=sk-...
```

**Frontend (.env.local):**
```
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

---

*Built with FastAPI + Next.js + FAISS + OpenAI*
