"""
rag/ingestion - Document ingestion modules for EmpowerSleep RAG system.

This package contains modules for ingesting various document types
(textbooks, PDFs, etc.) into the FAISS vector index.
"""

from .textbook_ingestor import TextbookIngestor

__all__ = ["TextbookIngestor"]
