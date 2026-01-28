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
├── scripts/
│   ├── scrape_empowersleep_blog.py  # Scrape blog articles
│   └── build_blog_index.py          # Build FAISS index
├── data/
│   └── blog_docs.jsonl     # Scraped articles (generated)
└── rag_artifacts/          # FAISS index + chunks (generated)
```

## Requirements

- Python 3.9+
- OpenAI API key

---

*Built with Streamlit + FAISS + OpenAI*
