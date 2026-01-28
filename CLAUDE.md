# CLAUDE.md - EmpowerSleep Technical Reference

This file contains technical details about the codebase for Claude Code.

## Architecture Overview

This is a **self-contained RAG chatbot** for sleep education. The entire application logic lives in `app.py` - there is no separate `core/` module.

```
User Question
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│                    app.py (Streamlit)                   │
│                                                         │
│  1. embed_query() ──► OpenAI text-embedding-3-small    │
│  2. retrieve_relevant_chunks() ──► FAISS search        │
│  3. generate_answer() ──► GPT-4o-mini with context     │
│  4. Display answer + sources                           │
└─────────────────────────────────────────────────────────┘
```

## Key Files

### app.py (Main Application)
- **Self-contained** Streamlit app with all RAG logic
- Uses `rag_artifacts/` for FAISS index and chunk metadata
- Configuration constants at top of file (lines 30-60)
- Key functions:
  - `embed_query()` - Generate query embedding (line 179)
  - `retrieve_relevant_chunks()` - FAISS similarity search (line 213)
  - `generate_answer()` - LLM call with context (line 439)
  - `answer_question()` - Main RAG pipeline (line 504)
- Conversation history support for multi-turn context
- System prompt enforces educational, non-diagnostic tone

### scripts/scrape_empowersleep_blog.py
- Scrapes articles from empowersleep.com/blog
- Output: `data/blog_docs.jsonl`
- Each article: `{title, url, text}`

### scripts/build_blog_index.py
- Reads `data/blog_docs.jsonl`
- Chunks articles (~1000 words, 150 word overlap)
- Generates embeddings with OpenAI `text-embedding-3-small`
- Builds FAISS `IndexFlatIP` (cosine similarity)
- Output: `rag_artifacts/faiss.index`, `chunks.jsonl`, `build_meta.json`

## Configuration

All config is in `app.py` constants (no separate config.py):

| Setting | Value | Location |
|---------|-------|----------|
| Embedding Model | `text-embedding-3-small` | Line 43 |
| Embedding Dim | 1536 | Line 44 |
| Top-K Results | 4 | Line 47 |
| LLM Model | `gpt-4o-mini` | Line 50 |
| LLM Temperature | 0.3 | Line 51 |
| Max Tokens | 600 | Line 52 |

## Data Flow

1. **Scraping**: `scrape_empowersleep_blog.py` → `data/blog_docs.jsonl`
2. **Indexing**: `build_blog_index.py` → `rag_artifacts/`
3. **Serving**: `app.py` loads index, handles queries

## Important Behaviors

### System Prompt (lines 307-365)
- **Non-diagnostic**: Never labels user with conditions
- Uses pattern-based language ("This is often associated with...")
- Asks clarifying questions when context is incomplete
- Maintains conversation continuity

### Conversation Context
- `build_search_query()` combines original topic with follow-ups for better retrieval
- `format_conversation_history()` includes last 3 turns in LLM prompt
- Prevents topic drift on clarifying answers

## Dependencies

- `faiss-cpu` - Vector similarity search
- `openai` - Embeddings + LLM
- `streamlit` - Web UI
- `python-dotenv` - Environment variables
- `sentence-transformers` - Listed but not used (legacy)
- `langchain` / `langchain-community` - Listed but not used (legacy)

## Running Locally

```bash
source venv/bin/activate
streamlit run app.py
```

Requires:
- `OPENAI_API_KEY` in `.env`
- `rag_artifacts/` populated (run build_blog_index.py first)

## Common Tasks

### Rebuild index after content changes
```bash
python scripts/scrape_empowersleep_blog.py
python scripts/build_blog_index.py
```

### Modify retrieval behavior
Edit `TOP_K_RESULTS` in app.py line 47

### Modify LLM behavior
Edit `SYSTEM_PROMPT` in app.py lines 307-365
