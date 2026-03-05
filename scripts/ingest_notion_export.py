#!/usr/bin/env python3
"""
scripts/ingest_notion_export.py
================================

Ingests a Notion markdown export into the EmpowerSleep RAG system.

Reads all .md files from data/raw/empower_sleep_notion/, cleans them,
chunks them, generates embeddings, and merges into the existing FAISS index.

Usage:
    # First-time ingest
    python scripts/ingest_notion_export.py

    # Replace previously ingested Notion content
    python scripts/ingest_notion_export.py --rebuild
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

# =============================================================================
# CONFIGURATION
# =============================================================================

NOTION_DIR = PROJECT_ROOT / "data" / "raw" / "empower_sleep_notion"
RAG_ARTIFACTS_DIR = PROJECT_ROOT / "rag_artifacts"
FAISS_INDEX_PATH = RAG_ARTIFACTS_DIR / "faiss.index"
CHUNKS_PATH = RAG_ARTIFACTS_DIR / "chunks.jsonl"

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536

# Smaller chunks than the blog — Notion pages are shorter, more focused
TARGET_CHUNK_WORDS = 400
MIN_CHUNK_WORDS = 50
OVERLAP_WORDS = 50

EMBEDDING_BATCH_SIZE = 100
SOURCE_TYPE = "notion"

# Notion export appends a 32-char hex UUID to every filename
_UUID_SUFFIX = re.compile(r'\s+[0-9a-f]{32}$')


# =============================================================================
# TEXT CLEANING
# =============================================================================

def clean_title(filename: str) -> str:
    """Extract a clean page title from a Notion export filename.

    e.g. 'Our Team 3018328fc76f80c78f56fe0f1228b1ea.md' → 'Our Team'
    """
    stem = Path(filename).stem
    return _UUID_SUFFIX.sub('', stem).strip()


def clean_markdown(text: str) -> str:
    """Clean Notion-exported markdown for ingestion.

    - Remove image references: ![alt](path)
    - Flatten internal Notion links: [text](page.md) → text
    - Strip HTML-like tags (<aside>, etc.)
    - Normalize whitespace
    """
    # Remove image lines
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)

    # Flatten internal Notion page links
    text = re.sub(r'\[([^\]]+)\]\([^)]+\.md[^)]*\)', r'\1', text)

    # Remove remaining markdown links (external URLs) — keep the label
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

    # Strip HTML-like tags
    text = re.sub(r'<[^>]+>', '', text)

    # Drop lines that are pure horizontal rules after stripping
    text = re.sub(r'^-{3,}$', '', text, flags=re.MULTILINE)

    # Normalize whitespace — collapse blank lines, trim each line
    lines = [' '.join(line.split()) for line in text.split('\n')]
    lines = [l for l in lines if l]

    return '\n'.join(lines)


# =============================================================================
# CHUNKING
# =============================================================================

def chunk_text(text: str) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()

    if len(words) < MIN_CHUNK_WORDS:
        return []

    if len(words) <= TARGET_CHUNK_WORDS:
        return [text]

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + TARGET_CHUNK_WORDS, len(words))
        chunk = ' '.join(words[start:end])
        if len(words[start:end]) >= MIN_CHUNK_WORDS:
            chunks.append(chunk)
        if end >= len(words):
            break
        start = end - OVERLAP_WORDS

    return chunks


# =============================================================================
# LOADING
# =============================================================================

def load_notion_pages() -> list[dict]:
    """Find and read all .md files from the Notion export folder."""
    if not NOTION_DIR.exists():
        raise FileNotFoundError(
            f"Notion export folder not found: {NOTION_DIR}\n"
            "Expected: data/raw/empower_sleep_notion/"
        )

    md_files = sorted(NOTION_DIR.rglob("*.md"))
    pages = []

    for path in md_files:
        title = clean_title(path.name)
        raw = path.read_text(encoding='utf-8')
        text = clean_markdown(raw)
        word_count = len(text.split())

        if word_count >= MIN_CHUNK_WORDS:
            pages.append({"title": title, "text": text})
            print(f"  ✓ {title:45s} ({word_count} words)")
        else:
            print(f"  – {title} (skipped — too short)")

    return pages


def create_chunks(pages: list[dict]) -> list[dict]:
    """Create RAG chunks from all Notion pages."""
    all_chunks = []
    chunk_id = 0

    for page in pages:
        text_chunks = chunk_text(page['text'])
        for idx, chunk_content in enumerate(text_chunks):
            all_chunks.append({
                "id": f"notion_{chunk_id:05d}",
                "title": page['title'],
                "source": SOURCE_TYPE,
                "chunk_index": idx,
                "text": chunk_content,
            })
            chunk_id += 1

    return all_chunks


# =============================================================================
# EMBEDDINGS
# =============================================================================

def embed_chunks(chunks: list[dict], client: OpenAI) -> np.ndarray:
    """Generate embeddings for all chunks via OpenAI."""
    all_embeddings = []
    total = len(chunks)

    for i in range(0, total, EMBEDDING_BATCH_SIZE):
        batch = chunks[i:i + EMBEDDING_BATCH_SIZE]
        texts = [c['text'] for c in batch]
        end = min(i + len(batch), total)

        print(f"  Embedding chunks {i + 1}–{end} / {total}")

        for attempt in range(3):
            try:
                response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
                all_embeddings.extend([item.embedding for item in response.data])
                break
            except Exception as e:
                if attempt < 2:
                    wait = 2 ** attempt
                    print(f"  Retry in {wait}s ({e})")
                    time.sleep(wait)
                else:
                    raise

        if i + EMBEDDING_BATCH_SIZE < total:
            time.sleep(0.3)

    return np.array(all_embeddings, dtype=np.float32)


# =============================================================================
# MERGE INTO INDEX
# =============================================================================

def merge_into_index(new_chunks: list[dict], new_embeddings: np.ndarray, rebuild: bool):
    """Merge new Notion chunks into the existing FAISS index."""
    RAG_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing chunks
    existing_chunks = []
    if CHUNKS_PATH.exists():
        with open(CHUNKS_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_chunks.append(json.loads(line))

    # Handle existing Notion content
    existing_notion = [c for c in existing_chunks if c.get('source') == SOURCE_TYPE]

    if existing_notion and not rebuild:
        print(f"  {len(existing_notion)} Notion chunks already in index.")
        print("  Use --rebuild to replace them.")
        return

    if rebuild and existing_notion:
        existing_chunks = [c for c in existing_chunks if c.get('source') != SOURCE_TYPE]
        print(f"  Removed {len(existing_notion)} old Notion chunks (rebuild mode)")

    # Extract existing embeddings from the FAISS index
    existing_embeddings = None
    if FAISS_INDEX_PATH.exists() and existing_chunks:
        try:
            existing_index = faiss.read_index(str(FAISS_INDEX_PATH))
            n = len(existing_chunks)
            if existing_index.ntotal >= n:
                existing_embeddings = np.zeros((n, EMBEDDING_DIMENSION), dtype=np.float32)
                for i in range(n):
                    existing_embeddings[i] = existing_index.reconstruct(i)
                print(f"  Loaded {n} existing embeddings from index")
        except Exception as e:
            print(f"  Warning: could not load existing embeddings: {e}")
            existing_chunks = []

    # Combine chunks and embeddings
    combined_chunks = existing_chunks + new_chunks

    if existing_embeddings is not None and len(existing_embeddings) > 0:
        combined_embeddings = np.vstack([existing_embeddings, new_embeddings])
    else:
        combined_embeddings = new_embeddings

    # Normalize (cosine similarity) and build index
    faiss.normalize_L2(combined_embeddings)
    index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
    index.add(combined_embeddings)

    # Save
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    print(f"  Saved FAISS index: {index.ntotal} total vectors")

    with open(CHUNKS_PATH, 'w', encoding='utf-8') as f:
        for chunk in combined_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    print(f"  Saved chunks.jsonl: {len(combined_chunks)} total chunks")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ingest Notion export into EmpowerSleep RAG system"
    )
    parser.add_argument(
        '--rebuild',
        action='store_true',
        help='Replace previously ingested Notion content'
    )
    args = parser.parse_args()

    print("=" * 60)
    print("EmpowerSleep Notion Ingestor")
    print("=" * 60)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set in .env")
        sys.exit(1)
    client = OpenAI(api_key=api_key)

    # Step 1: Load pages
    print(f"\n[1/4] Loading Notion pages from:\n      {NOTION_DIR}\n")
    pages = load_notion_pages()
    print(f"\n  Total: {len(pages)} pages loaded")

    if not pages:
        print("No pages found — check that the folder contains .md files.")
        sys.exit(1)

    # Step 2: Chunk
    print(f"\n[2/4] Chunking (target ~{TARGET_CHUNK_WORDS} words, {OVERLAP_WORDS} overlap)")
    chunks = create_chunks(pages)
    print(f"  Created {len(chunks)} chunks")

    # Step 3: Embed
    print(f"\n[3/4] Generating embeddings ({EMBEDDING_MODEL})")
    embeddings = embed_chunks(chunks, client)
    print(f"  Embeddings shape: {embeddings.shape}")

    # Step 4: Merge
    print(f"\n[4/4] Merging into FAISS index")
    merge_into_index(chunks, embeddings, rebuild=args.rebuild)

    print("\n" + "=" * 60)
    print(f"Done! {len(chunks)} Notion chunks added to the index.")
    print("=" * 60)


if __name__ == "__main__":
    main()
