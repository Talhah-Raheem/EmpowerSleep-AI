"""
config.py - Configuration settings for EmpowerSleep RAG Chatbot
================================================================

This file centralizes all configuration values. For a production system,
sensitive values like API keys should come from environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Base directory (where this file lives)
BASE_DIR = Path(__file__).parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DOCS_DIR = DATA_DIR / "raw"           # Source .txt files
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"  # Persisted vector index

# =============================================================================
# EMBEDDING MODEL CONFIGURATION
# =============================================================================

# Using sentence-transformers for local embeddings (no API key needed)
# This model is fast, free, and works offline
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # Dimension for all-MiniLM-L6-v2

# =============================================================================
# CHUNKING CONFIGURATION
# =============================================================================

# How to split documents into smaller pieces for embedding
CHUNK_SIZE = 500          # Target characters per chunk
CHUNK_OVERLAP = 100       # Overlap between chunks (helps preserve context)

# =============================================================================
# RETRIEVAL CONFIGURATION
# =============================================================================

# Number of similar chunks to retrieve for each query
TOP_K_RESULTS = 4

# Minimum total context length (characters) to attempt an answer
# If we retrieve less than this, we return a safe fallback
MIN_CONTEXT_LENGTH = 400

# =============================================================================
# LLM CONFIGURATION
# =============================================================================

# OpenAI API key - MUST be set in environment or .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Model to use for response generation
# gpt-4o-mini is cost-effective for demos; use gpt-4o for better quality
LLM_MODEL = "gpt-4o-mini"

# Temperature controls randomness (0 = deterministic, 1 = creative)
# Lower is better for factual Q&A
LLM_TEMPERATURE = 0.3

# Maximum tokens in the response
LLM_MAX_TOKENS = 500

# =============================================================================
# SAFETY CONFIGURATION
# =============================================================================

# These are used by core/safety.py for triage detection
# In production, these would be more comprehensive and possibly ML-based

CRISIS_KEYWORDS = [
    "suicide", "suicidal", "kill myself", "end my life", "want to die",
    "self-harm", "hurt myself", "cutting myself", "overdose",
    "can't go on", "no reason to live"
]

URGENT_KEYWORDS = [
    "can't sleep for days", "haven't slept in days", "no sleep for 72",
    "chest pain", "can't breathe", "heart racing won't stop",
    "hallucinating", "seeing things", "hearing voices",
    "medication not working", "took too much", "mixed medications"
]

# =============================================================================
# UI CONFIGURATION
# =============================================================================

# Demo questions shown as buttons in the Streamlit UI
DEMO_QUESTIONS = [
    "What is sleep hygiene and why does it matter?",
    "How does blue light from screens affect my sleep?",
    "What are some common myths about sleep?",
]
