"""
rag/ingestion/textbook_ingestor.py
===================================

Core module for ingesting PDF textbooks into the EmpowerSleep RAG system.

Features:
- PDF text extraction using PyMuPDF (fitz)
- Chapter detection (pattern-based)
- Intelligent chunking with page tracking
- Repeated header/footer removal
- TOC, index, and bibliography detection
- Idempotent processing via document hashing
- Merges with existing FAISS index

Usage:
    from rag.ingestion import TextbookIngestor

    ingestor = TextbookIngestor()
    ingestor.ingest(
        pdf_path="data/raw/Sleep_And_Health.pdf",
        book_title="Sleep and Health",
        version="v1"
    )
"""

import hashlib
import json
import os
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from openai import OpenAI

try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError(
        "PyMuPDF is required for PDF extraction. Install with: pip install PyMuPDF"
    )


# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
RAG_ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "rag_artifacts"
FAISS_INDEX_PATH = RAG_ARTIFACTS_DIR / "faiss.index"
CHUNKS_PATH = RAG_ARTIFACTS_DIR / "chunks.jsonl"
MANIFEST_PATH = RAG_ARTIFACTS_DIR / "textbook_manifest.json"

# Embedding configuration (must match app.py and build_blog_index.py)
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536

# Chunking parameters (textbook-specific: smaller than blog for precision)
TARGET_CHUNK_WORDS = 550  # 500-800 tokens ≈ 400-650 words
MIN_CHUNK_WORDS = 80
OVERLAP_WORDS = 85  # ~15% overlap

# Retry parameters
MAX_RETRIES = 3
RETRY_BACKOFF = 2
EMBEDDING_BATCH_SIZE = 100


# =============================================================================
# PDF TEXT EXTRACTION
# =============================================================================

def extract_pages(pdf_path: Path) -> list[dict]:
    """
    Extract text page-by-page using PyMuPDF.

    Returns:
        list[dict]: List of {page_num, text, is_toc, is_index, is_bibliography}
    """
    doc = fitz.open(str(pdf_path))
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")

        # Skip nearly empty pages
        if len(text.strip()) < 50:
            pages.append({
                "page_num": page_num + 1,  # 1-indexed
                "text": "",
                "is_empty": True,
                "is_toc": False,
                "is_index": False,
                "is_bibliography": False,
            })
            continue

        pages.append({
            "page_num": page_num + 1,
            "text": text,
            "is_empty": False,
            "is_toc": _is_toc_page(text),
            "is_index": _is_index_page(text),
            "is_bibliography": _is_bibliography_page(text),
        })

    doc.close()
    return pages


def _is_toc_page(text: str) -> bool:
    """Detect table of contents pages by patterns."""
    lines = text.strip().split('\n')
    if len(lines) < 5:
        return False

    # Count lines with TOC patterns: dots or dashes followed by numbers
    toc_pattern = re.compile(r'\.{3,}|\-{3,}|…+')
    page_num_pattern = re.compile(r'\d+\s*$')

    toc_line_count = 0
    for line in lines:
        line = line.strip()
        if toc_pattern.search(line) and page_num_pattern.search(line):
            toc_line_count += 1

    # If >40% of lines look like TOC entries
    return toc_line_count > len(lines) * 0.4


def _is_index_page(text: str) -> bool:
    """Detect index pages (alphabetically sorted short entries)."""
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    if len(lines) < 10:
        return False

    # Index pages have many short lines with page numbers
    short_lines_with_nums = 0
    for line in lines:
        if len(line) < 80 and re.search(r'\d+(?:,\s*\d+)*\s*$', line):
            short_lines_with_nums += 1

    return short_lines_with_nums > len(lines) * 0.5


def _is_bibliography_page(text: str) -> bool:
    """Detect bibliography/references pages."""
    text_lower = text.lower()

    # Check for bibliography headers
    bib_headers = ['bibliography', 'references', 'works cited', 'citations']
    has_header = any(h in text_lower[:500] for h in bib_headers)

    # Check for citation patterns (author-year, numbered refs)
    citation_pattern = re.compile(r'\(\d{4}\)|\[\d+\]|^\d+\.\s+[A-Z]', re.MULTILINE)
    citation_matches = len(citation_pattern.findall(text))

    return has_header or citation_matches > 5


def remove_repeated_headers_footers(pages: list[dict], threshold: float = 0.5) -> list[dict]:
    """
    Remove repeated headers/footers that appear on many pages.

    Strategy: Find lines that appear on >threshold fraction of pages.
    """
    if len(pages) < 5:
        return pages

    # Count first and last lines from each page
    first_lines = []
    last_lines = []

    for page in pages:
        if page.get("is_empty"):
            continue
        lines = page["text"].strip().split('\n')
        if lines:
            first_lines.append(lines[0].strip())
            last_lines.append(lines[-1].strip())

    # Find repeated lines
    first_line_counts = Counter(first_lines)
    last_line_counts = Counter(last_lines)

    total_pages = len([p for p in pages if not p.get("is_empty")])
    repeated_first = {line for line, count in first_line_counts.items()
                      if count > total_pages * threshold and len(line) < 100}
    repeated_last = {line for line, count in last_line_counts.items()
                     if count > total_pages * threshold and len(line) < 100}

    # Remove repeated lines
    for page in pages:
        if page.get("is_empty"):
            continue

        lines = page["text"].split('\n')
        cleaned_lines = []

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            # Skip if it's a repeated header (first few lines) or footer (last few lines)
            if i < 3 and line_stripped in repeated_first:
                continue
            if i >= len(lines) - 3 and line_stripped in repeated_last:
                continue
            cleaned_lines.append(line)

        page["text"] = '\n'.join(cleaned_lines)

    return pages


# =============================================================================
# CHAPTER DETECTION
# =============================================================================

def detect_chapters(pages: list[dict]) -> list[dict]:
    """
    Identify chapter boundaries and assign chapter names to pages.

    Detection strategies:
    - "Chapter X" or "CHAPTER X" patterns
    - All-caps lines at page start (likely section headers)
    """
    chapter_pattern = re.compile(
        r'^(?:CHAPTER|Chapter)\s+(\d+|[IVXLC]+)[\s:.\-]*(.*)$',
        re.MULTILINE
    )

    current_chapter = None

    for page in pages:
        if page.get("is_empty") or page.get("is_toc") or page.get("is_index"):
            page["chapter"] = current_chapter
            continue

        text = page["text"]

        # Look for chapter headers in first 500 chars
        match = chapter_pattern.search(text[:500])
        if match:
            chapter_num = match.group(1)
            chapter_title = match.group(2).strip() if match.group(2) else ""
            if chapter_title:
                current_chapter = f"Chapter {chapter_num}: {chapter_title}"
            else:
                current_chapter = f"Chapter {chapter_num}"

        page["chapter"] = current_chapter

    return pages


# =============================================================================
# CHUNKING
# =============================================================================

def chunk_pages(
    pages: list[dict],
    target_words: int = TARGET_CHUNK_WORDS,
    overlap_words: int = OVERLAP_WORDS
) -> list[dict]:
    """
    Chunk text with word-based targeting and page tracking.

    Args:
        pages: List of page dicts with text and metadata
        target_words: Target words per chunk (550 default)
        overlap_words: Words to overlap between chunks (85 default)

    Returns:
        list[dict]: Chunks with page_start, page_end, chapter, text
    """
    chunks = []
    current_text = []
    current_word_count = 0
    page_start = None
    page_end = None
    current_chapter = None

    for page in pages:
        # Skip non-content pages
        if page.get("is_empty") or page.get("is_toc") or page.get("is_index") or page.get("is_bibliography"):
            continue

        text = page["text"].strip()
        if not text:
            continue

        words = text.split()
        page_num = page["page_num"]
        chapter = page.get("chapter")

        if page_start is None:
            page_start = page_num
        page_end = page_num

        if chapter:
            current_chapter = chapter

        # Add words to current chunk
        current_text.extend(words)
        current_word_count += len(words)

        # Check if we've reached target size
        while current_word_count >= target_words:
            # Find a sentence boundary near the target
            chunk_words = current_text[:target_words]
            text_str = ' '.join(chunk_words)

            # Try to break at sentence boundary
            break_point = _find_sentence_boundary(text_str, target_words)

            if break_point < len(chunk_words) // 2:
                # No good boundary found, use target
                break_point = target_words

            # Create chunk
            chunk_text = ' '.join(current_text[:break_point])

            if len(chunk_text.split()) >= MIN_CHUNK_WORDS:
                chunks.append({
                    "text": chunk_text,
                    "page_start": page_start,
                    "page_end": page_end,
                    "chapter": current_chapter,
                })

            # Keep overlap words for next chunk
            current_text = current_text[max(0, break_point - overlap_words):]
            current_word_count = len(current_text)
            page_start = page_end  # New chunk starts from last page

    # Don't forget the last chunk
    if current_text and len(current_text) >= MIN_CHUNK_WORDS:
        chunks.append({
            "text": ' '.join(current_text),
            "page_start": page_start,
            "page_end": page_end,
            "chapter": current_chapter,
        })

    return chunks


def _find_sentence_boundary(text: str, target_words: int) -> int:
    """Find a sentence boundary near the target word count."""
    words = text.split()
    if len(words) <= target_words:
        return len(words)

    # Look for sentence-ending punctuation in last 20% of target
    search_start = int(target_words * 0.8)

    for i in range(target_words - 1, search_start - 1, -1):
        if i < len(words):
            word = words[i]
            if word.endswith(('.', '?', '!', '."', '?"', '!"')):
                return i + 1

    return target_words


# =============================================================================
# EMBEDDING & INDEXING
# =============================================================================

class TextbookIngestor:
    """
    Main class for ingesting PDF textbooks into the RAG system.

    Usage:
        ingestor = TextbookIngestor()
        ingestor.ingest(
            pdf_path="data/raw/Sleep_And_Health.pdf",
            book_title="Sleep and Health",
            version="v1"
        )
    """

    def __init__(self):
        self.client = None
        self._ensure_openai_client()

    def _ensure_openai_client(self):
        """Initialize OpenAI client."""
        from dotenv import load_dotenv
        load_dotenv()

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        self.client = OpenAI(api_key=api_key)

    def ingest(
        self,
        pdf_path: str | Path,
        book_title: str,
        version: str = "v1",
        rebuild: bool = False
    ) -> dict:
        """
        Main ingestion pipeline.

        Args:
            pdf_path: Path to the PDF file
            book_title: Human-readable title for the textbook
            version: Version string for idempotency
            rebuild: Force rebuild even if already indexed

        Returns:
            dict: Summary statistics
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        print("=" * 60)
        print(f"Textbook Ingestion: {book_title}")
        print("=" * 60)

        # Check idempotency
        doc_hash = self._compute_document_hash(pdf_path, version)

        if not rebuild and self._is_already_indexed(doc_hash):
            print(f"\nAlready indexed (hash: {doc_hash[:12]}...), skipping.")
            print("Use --rebuild to force re-processing.")
            return {"status": "skipped", "reason": "already_indexed"}

        # Step 1: Extract pages
        print(f"\n[1/6] Extracting pages from PDF...")
        pages = extract_pages(pdf_path)

        content_pages = [p for p in pages if not p.get("is_empty")]
        toc_pages = [p for p in pages if p.get("is_toc")]
        index_pages = [p for p in pages if p.get("is_index")]
        bib_pages = [p for p in pages if p.get("is_bibliography")]

        print(f"  Total pages: {len(pages)}")
        print(f"  Content pages: {len(content_pages)}")
        print(f"  TOC pages: {len(toc_pages)}")
        print(f"  Index pages: {len(index_pages)}")
        print(f"  Bibliography pages: {len(bib_pages)}")

        # Step 2: Remove repeated headers/footers
        print(f"\n[2/6] Cleaning headers/footers...")
        pages = remove_repeated_headers_footers(pages)

        # Step 3: Detect chapters
        print(f"\n[3/6] Detecting chapters...")
        pages = detect_chapters(pages)

        chapters = set(p.get("chapter") for p in pages if p.get("chapter"))
        print(f"  Chapters detected: {len(chapters)}")
        for ch in sorted(chapters, key=lambda x: x or ""):
            print(f"    - {ch}")

        # Step 4: Create chunks
        print(f"\n[4/6] Chunking text (target ~{TARGET_CHUNK_WORDS} words, {OVERLAP_WORDS} word overlap)...")
        raw_chunks = chunk_pages(pages)

        # Add metadata to chunks
        chunks = []
        for i, chunk in enumerate(raw_chunks):
            chunk_id = f"textbook_{doc_hash[:8]}_chunk_{i:05d}"

            # Build title from book + chapter
            title = book_title
            if chunk.get("chapter"):
                title = f"{book_title} - {chunk['chapter']}"

            # Create file:// URL with page anchor
            url = f"file://{pdf_path.resolve()}#page={chunk['page_start']}"

            chunks.append({
                "id": chunk_id,
                "title": title,
                "url": url,
                "chunk_index": i,
                "text": chunk["text"],
                "source": "textbook",
                "book_title": book_title,
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "chapter": chunk.get("chapter"),
                "document_hash": doc_hash,
            })

        print(f"  Chunks created: {len(chunks)}")

        if not chunks:
            print("Error: No chunks created from PDF")
            return {"status": "error", "reason": "no_chunks"}

        # Word count stats
        word_counts = [len(c["text"].split()) for c in chunks]
        print(f"  Word counts: min={min(word_counts)}, max={max(word_counts)}, avg={sum(word_counts)//len(word_counts)}")

        # Step 5: Generate embeddings
        print(f"\n[5/6] Generating embeddings...")
        embeddings = self._embed_chunks(chunks)

        if embeddings is None:
            print("Error: Failed to generate embeddings")
            return {"status": "error", "reason": "embedding_failed"}

        print(f"  Generated {len(embeddings)} embeddings")

        # Step 6: Update index
        print(f"\n[6/6] Updating FAISS index...")
        stats = self._update_index(chunks, embeddings, doc_hash, rebuild)

        # Update manifest
        self._update_manifest(doc_hash, book_title, version, pdf_path, len(chunks))

        print("\n" + "=" * 60)
        print("Ingestion complete!")
        print("=" * 60)
        print(f"  Book: {book_title}")
        print(f"  Chunks added: {len(chunks)}")
        print(f"  Index size: {stats['old_size']} -> {stats['new_size']}")

        return {
            "status": "success",
            "book_title": book_title,
            "chunks_created": len(chunks),
            "pages_extracted": len(pages),
            "chapters_detected": len(chapters),
            "index_old_size": stats["old_size"],
            "index_new_size": stats["new_size"],
        }

    def _compute_document_hash(self, pdf_path: Path, version: str) -> str:
        """Compute hash from PDF content + version for idempotency."""
        hasher = hashlib.sha256()

        # Hash file content
        with open(pdf_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)

        # Include version in hash
        hasher.update(version.encode())

        return hasher.hexdigest()

    def _is_already_indexed(self, doc_hash: str) -> bool:
        """Check if document is already in the manifest."""
        if not MANIFEST_PATH.exists():
            return False

        with open(MANIFEST_PATH, "r") as f:
            manifest = json.load(f)

        return doc_hash in manifest.get("documents", {})

    def _embed_chunks(self, chunks: list[dict]) -> Optional[np.ndarray]:
        """Generate embeddings for chunks."""
        all_embeddings = []
        total = len(chunks)

        for i in range(0, total, EMBEDDING_BATCH_SIZE):
            batch = chunks[i:i + EMBEDDING_BATCH_SIZE]
            batch_texts = [c["text"] for c in batch]

            batch_num = i // EMBEDDING_BATCH_SIZE + 1
            total_batches = (total + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE
            print(f"  Batch {batch_num}/{total_batches} (chunks {i+1}-{min(i+len(batch), total)})")

            embeddings = self._embed_batch_with_retry(batch_texts)
            if embeddings is None:
                return None

            all_embeddings.extend(embeddings)

            # Rate limiting
            if i + EMBEDDING_BATCH_SIZE < total:
                time.sleep(0.5)

        return np.array(all_embeddings, dtype=np.float32)

    def _embed_batch_with_retry(self, texts: list[str]) -> Optional[list[list[float]]]:
        """Embed a batch with retry logic."""
        backoff = RETRY_BACKOFF

        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=texts,
                )
                return [item.embedding for item in response.data]

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    print(f"    Warning: Embedding failed ({e}), retrying in {backoff}s...")
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    print(f"    Error: Embedding failed after {MAX_RETRIES} attempts: {e}")
                    return None

        return None

    def _update_index(
        self,
        new_chunks: list[dict],
        new_embeddings: np.ndarray,
        doc_hash: str,
        rebuild: bool
    ) -> dict:
        """
        Update the FAISS index with new chunks.

        Strategy:
        1. Load existing chunks and index
        2. If rebuild: remove old textbook chunks with same hash
        3. Append new chunks
        4. Rebuild index from combined embeddings
        """
        RAG_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

        # Load existing chunks
        existing_chunks = []
        if CHUNKS_PATH.exists():
            with open(CHUNKS_PATH, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        existing_chunks.append(json.loads(line))

        old_size = len(existing_chunks)

        # Filter out old textbook chunks if rebuilding
        if rebuild:
            # Remove chunks from the same document
            existing_chunks = [
                c for c in existing_chunks
                if c.get("source") != "textbook" or c.get("document_hash") != doc_hash
            ]
            print(f"  Removed old chunks for this document (rebuild mode)")

        # Also check if we need to remove duplicates by hash
        existing_hashes = {c.get("document_hash") for c in existing_chunks if c.get("source") == "textbook"}
        if doc_hash in existing_hashes and not rebuild:
            # This shouldn't happen if idempotency check works, but be safe
            existing_chunks = [
                c for c in existing_chunks
                if c.get("document_hash") != doc_hash
            ]

        # Combine chunks
        combined_chunks = existing_chunks + new_chunks

        # We need to regenerate embeddings for existing chunks if we're rebuilding
        # For simplicity, we'll store embeddings alongside chunks
        # But since we don't have stored embeddings, we need to rebuild the index

        # Load existing index to get embeddings
        existing_embeddings = None
        if FAISS_INDEX_PATH.exists() and existing_chunks:
            try:
                existing_index = faiss.read_index(str(FAISS_INDEX_PATH))
                # Extract vectors - this works for IndexFlatIP
                n_existing = len(existing_chunks)
                if existing_index.ntotal >= n_existing:
                    existing_embeddings = np.zeros((n_existing, EMBEDDING_DIMENSION), dtype=np.float32)
                    for i in range(n_existing):
                        existing_embeddings[i] = existing_index.reconstruct(i)
            except Exception as e:
                print(f"  Warning: Could not load existing embeddings: {e}")
                print(f"  Will only include new chunks")
                existing_embeddings = None
                combined_chunks = new_chunks

        # Combine embeddings
        if existing_embeddings is not None and len(existing_embeddings) > 0:
            combined_embeddings = np.vstack([existing_embeddings, new_embeddings])
        else:
            combined_embeddings = new_embeddings

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(combined_embeddings)

        # Build new index
        index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
        index.add(combined_embeddings)

        # Save index
        faiss.write_index(index, str(FAISS_INDEX_PATH))
        print(f"  Saved FAISS index: {FAISS_INDEX_PATH}")

        # Save chunks
        with open(CHUNKS_PATH, "w") as f:
            for chunk in combined_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        print(f"  Saved chunks: {CHUNKS_PATH}")

        return {
            "old_size": old_size,
            "new_size": len(combined_chunks),
        }

    def _update_manifest(
        self,
        doc_hash: str,
        book_title: str,
        version: str,
        pdf_path: Path,
        num_chunks: int
    ):
        """Update the textbook manifest for idempotency tracking."""
        manifest = {"documents": {}}

        if MANIFEST_PATH.exists():
            with open(MANIFEST_PATH, "r") as f:
                manifest = json.load(f)

        manifest["documents"][doc_hash] = {
            "book_title": book_title,
            "version": version,
            "pdf_path": str(pdf_path.resolve()),
            "num_chunks": num_chunks,
            "indexed_at": datetime.utcnow().isoformat() + "Z",
        }

        with open(MANIFEST_PATH, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"  Updated manifest: {MANIFEST_PATH}")

    def smoke_test(self, query: str) -> list[dict]:
        """
        Run a smoke test query against the index.

        Args:
            query: Test query string

        Returns:
            list[dict]: Top 3 matching chunks
        """
        print(f"\nSmoke Test: '{query}'")
        print("-" * 40)

        if not FAISS_INDEX_PATH.exists() or not CHUNKS_PATH.exists():
            print("Error: Index not found. Run ingestion first.")
            return []

        # Load index and chunks
        index = faiss.read_index(str(FAISS_INDEX_PATH))
        chunks = []
        with open(CHUNKS_PATH, "r") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))

        # Embed query
        response = self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=query,
        )
        query_embedding = np.array(response.data[0].embedding, dtype=np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.reshape(1, -1)

        # Search
        scores, indices = index.search(query_embedding, 3)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(chunks) and idx >= 0:
                chunk = chunks[idx]
                score = float(scores[0][i])

                print(f"\n[{i+1}] Score: {score:.4f}")
                print(f"    Source: {chunk.get('source', 'blog')}")
                if chunk.get("source") == "textbook":
                    print(f"    Book: {chunk.get('book_title')}")
                    print(f"    Chapter: {chunk.get('chapter')}")
                    print(f"    Pages: {chunk.get('page_start')}-{chunk.get('page_end')}")
                else:
                    print(f"    Title: {chunk.get('title')}")
                print(f"    Text: {chunk.get('text', '')[:200]}...")

                results.append({
                    "chunk": chunk,
                    "score": score,
                })

        return results
