# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EmpowerSleep is a Python-based RAG (Retrieval-Augmented Generation) chatbot providing sleep education with safety triage capabilities. It uses FAISS for vector search, sentence-transformers for embeddings, and OpenAI for response generation.

## Commands

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Then add OPENAI_API_KEY

# Build FAISS index (required before first run)
python scripts/build_index.py

# Run application
streamlit run app.py

# Test individual modules (each has demo code)
python core/safety.py
python core/service.py
python core/vectorstore.py
```

No formal test suite or linter is configured.

## Architecture

```
app.py (UI) → core/service.py (orchestration) → core/safety.py (triage)
                                              → core/vectorstore.py (RAG)
                                              → OpenAI API (generation)
```

**Request Flow:**
1. User question arrives at `app.py` (thin Streamlit UI)
2. `core/service.py:answer_question()` orchestrates the workflow
3. `core/safety.py:check_triage()` screens for crisis/urgent keywords first
4. If safe, `core/vectorstore.py` performs semantic search (top-4 chunks)
5. Context sufficiency guardrail checks if enough content (400+ chars)
6. LLM generates grounded response using retrieved context
7. Returns `ChatResponse` with answer, triage_level, and sources

**Key Design Decisions:**
- Safety triage happens before RAG (blocks dangerous queries immediately)
- Keyword-based triage is an MVP; production would need ML classification
- Context sufficiency guardrail prevents hallucination on out-of-scope queries
- All configuration centralized in `config.py`

## Key Files

- `config.py` - All settings (embedding model, chunk sizes, LLM params, safety keywords)
- `core/service.py` - Main service layer, `answer_question()` is the entry point
- `core/vectorstore.py` - FAISS index operations, embedding with `all-MiniLM-L6-v2`
- `core/safety.py` - Crisis/urgent keyword detection and triage responses
- `data/raw/` - Source documents (sleep_hygiene.txt, circadian_rhythm.txt, common_myths.txt)
- `scripts/build_index.py` - Builds FAISS index from raw documents

## Configuration

Key settings in `config.py`:
- Embedding: `all-MiniLM-L6-v2` (384-dim, local)
- Chunking: 500 chars with 100 overlap
- Retrieval: Top-4 results, 400-char minimum context
- LLM: `gpt-4o-mini`, temperature 0.3, 500 max tokens
- Safety keywords: `CRISIS_KEYWORDS` and `URGENT_KEYWORDS` lists

## Adding New Content

1. Add documents to `data/raw/`
2. Run `python scripts/build_index.py` to rebuild index
3. Index persists to `data/faiss_index/`
