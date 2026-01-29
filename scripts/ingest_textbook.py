#!/usr/bin/env python3
"""
scripts/ingest_textbook.py
===========================

CLI for ingesting PDF textbooks into the EmpowerSleep RAG system.

Usage:
    # Ingest a textbook
    python scripts/ingest_textbook.py \\
        --pdf data/raw/Sleep_And_Health.pdf \\
        --book-title "Sleep and Health" \\
        --version v1

    # Force rebuild (re-process even if already indexed)
    python scripts/ingest_textbook.py \\
        --pdf data/raw/Sleep_And_Health.pdf \\
        --book-title "Sleep and Health" \\
        --version v1 \\
        --rebuild

    # Run a smoke test query
    python scripts/ingest_textbook.py --smoke-test "What is REM sleep?"

Requirements:
    pip install PyMuPDF openai faiss-cpu numpy python-dotenv
"""

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

# Load .env from project root
load_dotenv(PROJECT_ROOT / ".env")


def main():
    parser = argparse.ArgumentParser(
        description="Ingest PDF textbooks into EmpowerSleep RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest a textbook
  python scripts/ingest_textbook.py \\
      --pdf data/raw/Sleep_And_Health.pdf \\
      --book-title "Sleep and Health" \\
      --version v1

  # Force rebuild
  python scripts/ingest_textbook.py \\
      --pdf data/raw/Sleep_And_Health.pdf \\
      --book-title "Sleep and Health" \\
      --version v1 \\
      --rebuild

  # Smoke test
  python scripts/ingest_textbook.py --smoke-test "What is sleep architecture?"
        """
    )

    # Ingestion arguments
    parser.add_argument(
        "--pdf",
        type=str,
        help="Path to PDF file to ingest"
    )
    parser.add_argument(
        "--book-title",
        type=str,
        help="Human-readable title for the textbook"
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1",
        help="Version string for idempotency (default: v1)"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild even if already indexed"
    )

    # Smoke test argument
    parser.add_argument(
        "--smoke-test",
        type=str,
        metavar="QUERY",
        help="Run a smoke test query against the index"
    )

    args = parser.parse_args()

    # Import here to avoid import errors before dotenv is loaded
    from rag.ingestion import TextbookIngestor

    ingestor = TextbookIngestor()

    # Handle smoke test
    if args.smoke_test:
        results = ingestor.smoke_test(args.smoke_test)
        if results:
            print(f"\nFound {len(results)} results.")
        return

    # Validate ingestion arguments
    if not args.pdf:
        parser.error("--pdf is required for ingestion")
    if not args.book_title:
        parser.error("--book-title is required for ingestion")

    pdf_path = Path(args.pdf)

    # Handle relative paths
    if not pdf_path.is_absolute():
        pdf_path = PROJECT_ROOT / pdf_path

    if not pdf_path.exists():
        print(f"Error: PDF not found: {pdf_path}")
        sys.exit(1)

    # Run ingestion
    result = ingestor.ingest(
        pdf_path=pdf_path,
        book_title=args.book_title,
        version=args.version,
        rebuild=args.rebuild
    )

    if result.get("status") == "success":
        print("\nIngestion successful!")
        sys.exit(0)
    elif result.get("status") == "skipped":
        print("\nAlready indexed. Use --rebuild to force re-processing.")
        sys.exit(0)
    else:
        print(f"\nIngestion failed: {result.get('reason', 'unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
