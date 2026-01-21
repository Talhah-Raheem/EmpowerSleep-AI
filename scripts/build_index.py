#!/usr/bin/env python3
"""
scripts/build_index.py - Build FAISS Index from Documents
==========================================================

This script reads all .txt files from data/raw/, chunks them,
creates embeddings, and saves a FAISS index to data/faiss_index/.

Usage:
    python scripts/build_index.py

Run this script whenever you:
- Add new documents to data/raw/
- Modify existing documents
- Change chunking parameters in config.py

The script will overwrite any existing index.
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.vectorstore import (
    load_documents,
    build_index,
    save_index,
)
from config import RAW_DOCS_DIR, FAISS_INDEX_DIR


def main():
    """Main entry point for building the index."""

    print("=" * 60)
    print("EMPOWERSLEEP - Building FAISS Index")
    print("=" * 60)
    print()

    # Step 1: Check for source documents
    print(f"📁 Looking for documents in: {RAW_DOCS_DIR}")

    if not RAW_DOCS_DIR.exists():
        print(f"❌ Directory does not exist: {RAW_DOCS_DIR}")
        print("   Please create the directory and add .txt files")
        sys.exit(1)

    txt_files = list(RAW_DOCS_DIR.glob("*.txt"))
    if not txt_files:
        print(f"❌ No .txt files found in {RAW_DOCS_DIR}")
        print("   Please add educational content files")
        sys.exit(1)

    print()

    # Step 2: Load and chunk documents
    print("📄 STEP 1: Loading and chunking documents")
    print("-" * 40)
    chunks = load_documents(RAW_DOCS_DIR)

    if not chunks:
        print("❌ No chunks created from documents")
        sys.exit(1)

    print()

    # Step 3: Build the index
    print("🔢 STEP 2: Creating embeddings and building index")
    print("-" * 40)
    index, chunks = build_index(chunks)

    print()

    # Step 4: Save the index
    print("💾 STEP 3: Saving index to disk")
    print("-" * 40)
    save_index(index, chunks, FAISS_INDEX_DIR)

    print()
    print("=" * 60)
    print("✅ INDEX BUILD COMPLETE")
    print("=" * 60)
    print()
    print(f"   Documents processed: {len(txt_files)}")
    print(f"   Total chunks: {len(chunks)}")
    print(f"   Index vectors: {index.ntotal}")
    print(f"   Index location: {FAISS_INDEX_DIR}")
    print()
    print("You can now run the chatbot with: streamlit run app.py")


if __name__ == "__main__":
    main()
