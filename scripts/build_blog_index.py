#!/usr/bin/env python3
"""
scripts/build_blog_index.py
============================

Builds a FAISS vector index from scraped EmpowerSleep blog articles.

Pipeline:
1. Load articles from data/blog_docs.jsonl
2. Chunk each article into overlapping segments (~1000 words, ~150 word overlap)
3. Generate embeddings using OpenAI text-embedding-3-small
4. Build FAISS index (IndexFlatIP for cosine similarity)
5. Save artifacts to rag_artifacts/

Usage:
    export OPENAI_API_KEY="sk-..."
    python scripts/build_blog_index.py

Requirements:
    pip install openai faiss-cpu numpy

Output:
    rag_artifacts/
    ├── faiss.index         # FAISS index file
    ├── chunks.jsonl        # Chunk metadata (id, title, url, chunk_index, text)
    └── build_meta.json     # Build metadata (counts, timestamp)
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
from openai import OpenAI

from dotenv import load_dotenv

# Load .env from project root (parent of scripts/)
_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(_env_path)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Input/Output paths
INPUT_PATH = Path(__file__).parent.parent / "data" / "blog_docs.jsonl"
OUTPUT_DIR = Path(__file__).parent.parent / "rag_artifacts"

# Chunking parameters (word-based)
TARGET_CHUNK_WORDS = 1000  # Target words per chunk (800-1200 range)
MIN_CHUNK_WORDS = 100      # Skip chunks smaller than this
OVERLAP_WORDS = 150        # Overlap between consecutive chunks

# Embedding parameters
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536  # Dimension for text-embedding-3-small

# Retry parameters
MAX_RETRIES = 3
RETRY_BACKOFF = 2  # seconds, doubles each retry

# Batch size for embeddings (OpenAI allows up to 2048 inputs per request)
EMBEDDING_BATCH_SIZE = 100


# =============================================================================
# TEXT CHUNKING
# =============================================================================

def clean_text(text: str) -> str:
    """
    Clean text for chunking.

    - Normalize whitespace
    - Keep markdown-style headings (## ) for structure
    - Remove excessive newlines
    """
    # Normalize whitespace within lines
    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        line = ' '.join(line.split())  # Collapse internal whitespace
        if line:
            cleaned_lines.append(line)

    # Join with single newlines, collapse multiple blank lines
    return '\n'.join(cleaned_lines)


def chunk_text(text: str, target_words: int = TARGET_CHUNK_WORDS,
               overlap_words: int = OVERLAP_WORDS) -> List[str]:
    """
    Split text into overlapping chunks based on word count.

    Strategy:
    - Split into words
    - Create chunks of ~target_words with overlap_words overlap
    - Try to break at sentence boundaries when possible

    Args:
        text: The text to chunk
        target_words: Target number of words per chunk
        overlap_words: Number of words to overlap between chunks

    Returns:
        List of text chunks
    """
    # Clean the text first
    text = clean_text(text)

    # Split into words (preserving structure)
    words = text.split()

    if len(words) <= target_words:
        # Text is small enough to be one chunk
        return [text] if len(words) >= MIN_CHUNK_WORDS else []

    chunks = []
    start_idx = 0

    while start_idx < len(words):
        # Calculate end index for this chunk
        end_idx = min(start_idx + target_words, len(words))

        # Try to find a sentence boundary near the end
        # Look for period/question mark/exclamation followed by space or end
        if end_idx < len(words):
            # Search backwards from end_idx for a sentence boundary
            best_break = end_idx
            search_start = max(start_idx + target_words - 200, start_idx)  # Look back up to 200 words

            for i in range(end_idx - 1, search_start, -1):
                word = words[i]
                if word.endswith(('.', '?', '!', '."', '?"', '!"')):
                    best_break = i + 1
                    break

            end_idx = best_break

        # Extract chunk
        chunk_words = words[start_idx:end_idx]
        chunk_text = ' '.join(chunk_words)

        # Only add if chunk meets minimum size
        if len(chunk_words) >= MIN_CHUNK_WORDS:
            chunks.append(chunk_text)

        # Move start index, accounting for overlap
        # If we're at the end, break
        if end_idx >= len(words):
            break

        start_idx = end_idx - overlap_words

        # Ensure we make progress
        if start_idx <= chunks[-1].count(' ') if chunks else 0:
            start_idx = end_idx

    return chunks


def create_chunks_from_articles(articles: List[Dict]) -> List[Dict]:
    """
    Create chunks from all articles with metadata.

    Args:
        articles: List of article dicts with title, url, text

    Returns:
        List of chunk dicts with id, title, url, chunk_index, text
    """
    all_chunks = []
    chunk_id = 0

    for article in articles:
        title = article.get("title", "Untitled")
        url = article.get("url", "")
        text = article.get("text", "")

        # Chunk the article text
        text_chunks = chunk_text(text)

        for idx, chunk_content in enumerate(text_chunks):
            all_chunks.append({
                "id": f"chunk_{chunk_id:05d}",
                "title": title,
                "url": url,
                "chunk_index": idx,
                "text": chunk_content,
            })
            chunk_id += 1

    return all_chunks


# =============================================================================
# EMBEDDINGS
# =============================================================================

def get_openai_client() -> OpenAI:
    """
    Initialize OpenAI client.

    Reads API key from OPENAI_API_KEY environment variable.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set. "
            "Set it with: export OPENAI_API_KEY='sk-...'"
        )
    return OpenAI(api_key=api_key)


def embed_texts_batch(client: OpenAI, texts: List[str],
                      retries: int = MAX_RETRIES) -> Optional[List[List[float]]]:
    """
    Get embeddings for a batch of texts with retry logic.

    Args:
        client: OpenAI client
        texts: List of texts to embed
        retries: Number of retries on failure

    Returns:
        List of embedding vectors, or None if all retries failed
    """
    backoff = RETRY_BACKOFF

    for attempt in range(retries):
        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts,
            )
            # Extract embeddings in order
            embeddings = [item.embedding for item in response.data]
            return embeddings

        except Exception as e:
            if attempt < retries - 1:
                print(f"  Warning: Embedding failed ({e}), retrying in {backoff}s...")
                time.sleep(backoff)
                backoff *= 2
            else:
                print(f"  Error: Embedding failed after {retries} attempts: {e}")
                return None

    return None


def embed_all_chunks(chunks: List[Dict]) -> Optional[np.ndarray]:
    """
    Generate embeddings for all chunks.

    Args:
        chunks: List of chunk dicts with 'text' field

    Returns:
        numpy array of shape (n_chunks, embedding_dim), or None on failure
    """
    client = get_openai_client()

    all_embeddings = []
    total = len(chunks)

    # Process in batches
    for i in range(0, total, EMBEDDING_BATCH_SIZE):
        batch = chunks[i:i + EMBEDDING_BATCH_SIZE]
        batch_texts = [c["text"] for c in batch]

        print(f"  Embedding batch {i//EMBEDDING_BATCH_SIZE + 1}/{(total + EMBEDDING_BATCH_SIZE - 1)//EMBEDDING_BATCH_SIZE} "
              f"(chunks {i+1}-{min(i+len(batch), total)}/{total})")

        embeddings = embed_texts_batch(client, batch_texts)
        if embeddings is None:
            print(f"  Error: Failed to embed batch starting at chunk {i}")
            return None

        all_embeddings.extend(embeddings)

        # Small delay between batches to avoid rate limits
        if i + EMBEDDING_BATCH_SIZE < total:
            time.sleep(0.5)

    # Convert to numpy array
    return np.array(all_embeddings, dtype=np.float32)


# =============================================================================
# FAISS INDEX
# =============================================================================

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS index from embeddings.

    Uses IndexFlatIP (Inner Product) with L2-normalized vectors,
    which is equivalent to cosine similarity. This is preferred for
    text embeddings because:
    - Cosine similarity is standard for semantic search
    - OpenAI embeddings are already normalized

    Args:
        embeddings: numpy array of shape (n, dim)

    Returns:
        FAISS index
    """
    n_vectors, dim = embeddings.shape

    # Normalize vectors for cosine similarity
    # (OpenAI embeddings should already be normalized, but let's ensure)
    faiss.normalize_L2(embeddings)

    # Create index using Inner Product (equivalent to cosine for normalized vectors)
    # IndexFlatIP = exact search, no approximation
    index = faiss.IndexFlatIP(dim)

    # Add vectors to index
    index.add(embeddings)

    print(f"  Built FAISS IndexFlatIP with {index.ntotal} vectors, dim={dim}")

    return index


# =============================================================================
# FILE I/O
# =============================================================================

def load_articles(path: Path) -> List[Dict]:
    """Load articles from JSONL file."""
    articles = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                articles.append(json.loads(line))
    return articles


def save_chunks_jsonl(chunks: List[Dict], path: Path) -> None:
    """Save chunks to JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def save_build_meta(meta: Dict, path: Path) -> None:
    """Save build metadata to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Main entry point."""
    print("=" * 60)
    print("EmpowerSleep Blog Index Builder")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Step 1: Load articles
    # -------------------------------------------------------------------------
    print(f"\n[1/5] Loading articles from {INPUT_PATH}")

    if not INPUT_PATH.exists():
        print(f"Error: Input file not found: {INPUT_PATH}")
        print("Run scripts/scrape_empowersleep_blog.py first.")
        return

    articles = load_articles(INPUT_PATH)
    print(f"  Loaded {len(articles)} articles")

    if not articles:
        print("Error: No articles found")
        return

    # -------------------------------------------------------------------------
    # Step 2: Create chunks
    # -------------------------------------------------------------------------
    print(f"\n[2/5] Chunking articles (target ~{TARGET_CHUNK_WORDS} words, {OVERLAP_WORDS} word overlap)")

    chunks = create_chunks_from_articles(articles)
    print(f"  Created {len(chunks)} chunks from {len(articles)} articles")

    if not chunks:
        print("Error: No chunks created")
        return

    # Print some stats
    word_counts = [len(c["text"].split()) for c in chunks]
    print(f"  Chunk word counts: min={min(word_counts)}, max={max(word_counts)}, "
          f"avg={sum(word_counts)//len(word_counts)}")

    # -------------------------------------------------------------------------
    # Step 3: Generate embeddings
    # -------------------------------------------------------------------------
    print(f"\n[3/5] Generating embeddings using {EMBEDDING_MODEL}")

    embeddings = embed_all_chunks(chunks)
    if embeddings is None:
        print("Error: Failed to generate embeddings")
        return

    print(f"  Generated {len(embeddings)} embeddings, shape: {embeddings.shape}")

    # -------------------------------------------------------------------------
    # Step 4: Build FAISS index
    # -------------------------------------------------------------------------
    print(f"\n[4/5] Building FAISS index")

    index = build_faiss_index(embeddings)

    # -------------------------------------------------------------------------
    # Step 5: Save artifacts
    # -------------------------------------------------------------------------
    print(f"\n[5/5] Saving artifacts to {OUTPUT_DIR}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save FAISS index
    index_path = OUTPUT_DIR / "faiss.index"
    faiss.write_index(index, str(index_path))
    print(f"  Saved FAISS index: {index_path}")

    # Save chunks metadata
    chunks_path = OUTPUT_DIR / "chunks.jsonl"
    save_chunks_jsonl(chunks, chunks_path)
    print(f"  Saved chunks metadata: {chunks_path}")

    # Save build metadata
    meta = {
        "build_timestamp": datetime.utcnow().isoformat() + "Z",
        "source_file": str(INPUT_PATH),
        "num_articles": len(articles),
        "num_chunks": len(chunks),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "chunk_target_words": TARGET_CHUNK_WORDS,
        "chunk_overlap_words": OVERLAP_WORDS,
        "faiss_index_type": "IndexFlatIP (cosine similarity)",
    }
    meta_path = OUTPUT_DIR / "build_meta.json"
    save_build_meta(meta, meta_path)
    print(f"  Saved build metadata: {meta_path}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Build complete!")
    print("=" * 60)
    print(f"  Articles: {len(articles)}")
    print(f"  Chunks: {len(chunks)}")
    print(f"  Index size: {index.ntotal} vectors")
    print(f"  Output: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
