"""
core/vectorstore.py - FAISS Vector Store with Metadata
=======================================================

This module handles all vector store operations:
- Loading and chunking documents with metadata
- Creating embeddings using sentence-transformers
- Building and saving FAISS index
- Similarity search with source tracking

Key Concepts:
- Embeddings: Dense vector representations of text
- FAISS: Facebook AI Similarity Search - efficient similarity lookup
- Chunks: Documents split into smaller pieces for better retrieval
- Metadata: Information about each chunk (source file, chunk ID)
"""

import json
import pickle
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Import configuration
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIMENSION,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RAW_DOCS_DIR,
    FAISS_INDEX_DIR,
    TOP_K_RESULTS,
)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class DocumentChunk:
    """
    Represents a chunk of text with its metadata.

    Attributes:
        text: The actual text content
        metadata: Dict with 'source' (filename) and 'chunk_id' (int)
    """

    def __init__(self, text: str, source: str, chunk_id: int):
        self.text = text
        self.metadata = {
            "source": source,
            "chunk_id": chunk_id,
        }

    def __repr__(self):
        preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"DocumentChunk(source={self.metadata['source']}, chunk_id={self.metadata['chunk_id']}, text='{preview}')"


class RetrievalResult:
    """
    Result from a similarity search.

    Attributes:
        chunks: List of DocumentChunk objects (most similar first)
        total_context_length: Combined character count of all chunks
        sources: Unique list of source filenames
    """

    def __init__(self, chunks: list[DocumentChunk]):
        self.chunks = chunks
        self.total_context_length = sum(len(c.text) for c in chunks)
        # Get unique sources while preserving order
        seen = set()
        self.sources = []
        for chunk in chunks:
            src = chunk.metadata["source"]
            if src not in seen:
                seen.add(src)
                self.sources.append(src)

    def get_context_string(self) -> str:
        """Combine all chunk texts into a single context string."""
        return "\n\n---\n\n".join(chunk.text for chunk in self.chunks)


# =============================================================================
# TEXT CHUNKING
# =============================================================================

def chunk_text(text: str, source: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[DocumentChunk]:
    """
    Split text into overlapping chunks with metadata.

    This uses a simple character-based splitting approach.
    For production, consider sentence-aware splitting.

    Args:
        text: The full document text
        source: Filename for metadata
        chunk_size: Target size of each chunk in characters
        overlap: Number of characters to overlap between chunks

    Returns:
        List of DocumentChunk objects
    """
    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        # Get chunk of text
        end = start + chunk_size

        # Try to break at a sentence boundary (period, newline)
        if end < len(text):
            # Look for natural break points near the end
            for break_char in ["\n\n", "\n", ". ", "! ", "? "]:
                break_pos = text.rfind(break_char, start + chunk_size // 2, end + 50)
                if break_pos != -1:
                    end = break_pos + len(break_char)
                    break

        chunk_text_content = text[start:end].strip()

        # Only add non-empty chunks
        if chunk_text_content:
            chunks.append(DocumentChunk(
                text=chunk_text_content,
                source=source,
                chunk_id=chunk_id
            ))
            chunk_id += 1

        # Move start position, accounting for overlap
        start = end - overlap

        # Prevent infinite loop
        if start >= len(text) - overlap:
            break

    return chunks


def load_documents(docs_dir: Path = RAW_DOCS_DIR) -> list[DocumentChunk]:
    """
    Load all .txt files from a directory and chunk them.

    Args:
        docs_dir: Directory containing .txt files

    Returns:
        List of all DocumentChunk objects from all files
    """
    all_chunks = []

    # Find all .txt files
    txt_files = list(docs_dir.glob("*.txt"))

    if not txt_files:
        print(f"⚠️  No .txt files found in {docs_dir}")
        return all_chunks

    print(f"📂 Found {len(txt_files)} document(s) to process")

    for filepath in txt_files:
        print(f"   Processing: {filepath.name}")

        # Read file content
        text = filepath.read_text(encoding="utf-8")

        # Chunk the document
        chunks = chunk_text(text, source=filepath.name)
        all_chunks.extend(chunks)

        print(f"   → Created {len(chunks)} chunks")

    print(f"📊 Total chunks: {len(all_chunks)}")
    return all_chunks


# =============================================================================
# EMBEDDING MODEL
# =============================================================================

# Global model instance (lazy loaded)
_embedding_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """
    Get or create the embedding model (singleton pattern).

    Uses sentence-transformers which runs locally - no API key needed.
    """
    global _embedding_model

    if _embedding_model is None:
        print(f"🔄 Loading embedding model: {EMBEDDING_MODEL_NAME}")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("✅ Embedding model loaded")

    return _embedding_model


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Convert texts to embedding vectors.

    Args:
        texts: List of strings to embed

    Returns:
        NumPy array of shape (len(texts), EMBEDDING_DIMENSION)
    """
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=True)
    return np.array(embeddings).astype("float32")


# =============================================================================
# FAISS INDEX OPERATIONS
# =============================================================================

def build_index(chunks: list[DocumentChunk]) -> tuple[faiss.IndexFlatL2, list[DocumentChunk]]:
    """
    Build a FAISS index from document chunks.

    Uses IndexFlatL2 which does exact L2 (Euclidean) distance search.
    For larger datasets, consider IndexIVFFlat for approximate search.

    Args:
        chunks: List of DocumentChunk objects to index

    Returns:
        Tuple of (FAISS index, chunks list for metadata lookup)
    """
    if not chunks:
        raise ValueError("No chunks provided to build index")

    # Extract texts and create embeddings
    texts = [chunk.text for chunk in chunks]
    print(f"🔢 Creating embeddings for {len(texts)} chunks...")
    embeddings = embed_texts(texts)

    # Create FAISS index
    print("🏗️  Building FAISS index...")
    index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
    index.add(embeddings)

    print(f"✅ Index built with {index.ntotal} vectors")
    return index, chunks


def save_index(index: faiss.IndexFlatL2, chunks: list[DocumentChunk],
               index_dir: Path = FAISS_INDEX_DIR) -> None:
    """
    Save FAISS index and metadata to disk.

    Creates two files:
    - index.faiss: The FAISS index
    - chunks.pkl: Pickled list of DocumentChunk objects

    Args:
        index: The FAISS index
        chunks: List of DocumentChunk objects
        index_dir: Directory to save to
    """
    # Ensure directory exists
    index_dir.mkdir(parents=True, exist_ok=True)

    # Save FAISS index
    index_path = index_dir / "index.faiss"
    faiss.write_index(index, str(index_path))
    print(f"💾 Saved index to {index_path}")

    # Save chunks metadata
    chunks_path = index_dir / "chunks.pkl"
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"💾 Saved metadata to {chunks_path}")


def load_index(index_dir: Path = FAISS_INDEX_DIR) -> tuple[faiss.IndexFlatL2, list[DocumentChunk]]:
    """
    Load FAISS index and metadata from disk.

    Args:
        index_dir: Directory containing saved index

    Returns:
        Tuple of (FAISS index, chunks list)

    Raises:
        FileNotFoundError: If index files don't exist
    """
    index_path = index_dir / "index.faiss"
    chunks_path = index_dir / "chunks.pkl"

    if not index_path.exists():
        raise FileNotFoundError(
            f"Index not found at {index_path}. "
            "Run 'python scripts/build_index.py' first."
        )

    # Load FAISS index
    index = faiss.read_index(str(index_path))

    # Load chunks metadata
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    print(f"✅ Loaded index with {index.ntotal} vectors and {len(chunks)} chunks")
    return index, chunks


# =============================================================================
# SIMILARITY SEARCH
# =============================================================================

# Global index and chunks (lazy loaded)
_index: Optional[faiss.IndexFlatL2] = None
_chunks: Optional[list[DocumentChunk]] = None


def _ensure_index_loaded() -> tuple[faiss.IndexFlatL2, list[DocumentChunk]]:
    """Load index if not already loaded."""
    global _index, _chunks

    if _index is None or _chunks is None:
        _index, _chunks = load_index()

    return _index, _chunks


def similarity_search(query: str, top_k: int = TOP_K_RESULTS) -> RetrievalResult:
    """
    Find the most similar chunks to a query.

    Args:
        query: The search query
        top_k: Number of results to return

    Returns:
        RetrievalResult with matching chunks and metadata
    """
    index, chunks = _ensure_index_loaded()

    # Embed the query
    query_embedding = embed_texts([query])

    # Search the index
    distances, indices = index.search(query_embedding, top_k)

    # Get the matching chunks
    results = []
    for idx in indices[0]:
        if idx >= 0 and idx < len(chunks):  # Valid index
            results.append(chunks[idx])

    return RetrievalResult(results)


def reload_index() -> None:
    """Force reload of the index (useful after rebuilding)."""
    global _index, _chunks
    _index = None
    _chunks = None
    _ensure_index_loaded()


# =============================================================================
# TESTING / DEMO
# =============================================================================

if __name__ == "__main__":
    # Demo: Load docs and show chunks
    print("=" * 60)
    print("VECTORSTORE DEMO")
    print("=" * 60)

    # Try to load existing index and do a search
    try:
        result = similarity_search("What is sleep hygiene?")
        print(f"\n🔍 Search results for 'What is sleep hygiene?':")
        print(f"   Found {len(result.chunks)} chunks")
        print(f"   Total context: {result.total_context_length} characters")
        print(f"   Sources: {result.sources}")
        print(f"\n   First chunk preview:")
        if result.chunks:
            print(f"   {result.chunks[0].text[:200]}...")
    except FileNotFoundError as e:
        print(f"\n⚠️  {e}")
        print("   Run 'python scripts/build_index.py' to create the index")
