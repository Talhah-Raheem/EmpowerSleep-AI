# EmpowerSleep

A sleep education chatbot powered by RAG (Retrieval-Augmented Generation) using content from the EmpowerSleep blog.

## Quick Start

```bash
# 1. Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Add your OpenAI API key to .env
echo "OPENAI_API_KEY=sk-..." > .env

# 3. Scrape blog content and build index
python scripts/scrape_empowersleep_blog.py
python scripts/build_blog_index.py

# 4. Run the app
streamlit run app.py
```

## How It Works

1. User asks a sleep-related question
2. Question is embedded and matched against blog content using FAISS
3. Relevant content is retrieved and used as context
4. GPT-4o-mini generates a grounded, educational answer
5. Sources are cited for transparency

## Project Structure

```
EMPOWERSLEEP/
├── app.py                  # Streamlit app (self-contained)
├── requirements.txt        # Dependencies
├── .env                    # OpenAI API key (create this)
├── rag/
│   └── ingestion/
│       └── textbook_ingestor.py  # PDF textbook ingestion
├── scripts/
│   ├── scrape_empowersleep_blog.py  # Scrape blog articles
│   ├── build_blog_index.py          # Build FAISS index
│   └── ingest_textbook.py           # Ingest PDF textbooks
├── data/
│   ├── blog_docs.jsonl     # Scraped articles (generated)
│   └── raw/                # Place PDF textbooks here
└── rag_artifacts/          # FAISS index + chunks (generated)
    ├── faiss.index         # Combined vector index
    ├── chunks.jsonl        # All chunks (blog + textbook)
    └── textbook_manifest.json  # Tracks indexed textbooks
```

## Textbook Ingestion

You can add PDF textbooks to the knowledge base for richer educational content.

### Ingest a Textbook

```bash
# Place your PDF in data/raw/
cp ~/Downloads/Sleep_And_Health.pdf data/raw/

# Run ingestion
python scripts/ingest_textbook.py \
    --pdf data/raw/Sleep_And_Health.pdf \
    --book-title "Sleep and Health" \
    --version v1
```

### Options

- `--pdf`: Path to the PDF file
- `--book-title`: Human-readable title for display
- `--version`: Version string for idempotency (change to force re-index)
- `--rebuild`: Force re-processing even if already indexed

### Smoke Test

Verify the index includes textbook content:

```bash
python scripts/ingest_textbook.py --smoke-test "What is sleep architecture?"
```

### Features

- **Automatic chapter detection**: Chapters are extracted and shown in citations
- **Page tracking**: Sources show page numbers (e.g., "pp. 42-44")
- **Idempotent**: Re-running skips already-indexed documents
- **Merged index**: Textbook content is searchable alongside blog articles
- **Smart cleaning**: TOC, index, and bibliography pages are excluded

## Requirements

- Python 3.9+
- OpenAI API key

---

*Built with Streamlit + FAISS + OpenAI*
