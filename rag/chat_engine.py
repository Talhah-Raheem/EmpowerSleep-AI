"""
rag/chat_engine.py
==================

Core RAG chat engine extracted from the Streamlit app.
This module provides a clean API for the RAG pipeline that can be
used by any frontend (FastAPI, CLI, etc.).

The RAG pipeline:
1. Embeds user query using OpenAI
2. Retrieves relevant chunks from FAISS index
3. Generates grounded answer using LLM
4. Returns answer with source citations

Usage:
    from rag.chat_engine import ChatEngine

    engine = ChatEngine()
    answer, sources = engine.ask_question("What is sleep hygiene?")
"""

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()


# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths to RAG artifacts
RAG_ARTIFACTS_DIR = Path(__file__).parent.parent / "rag_artifacts"
FAISS_INDEX_PATH = RAG_ARTIFACTS_DIR / "faiss.index"
CHUNKS_PATH = RAG_ARTIFACTS_DIR / "chunks.jsonl"

# Embedding configuration (must match build_blog_index.py)
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536

# Retrieval configuration
TOP_K_RESULTS = 4

# LLM configuration
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 600

# Disclaimer appended to all responses
DISCLAIMER_SUFFIX = "\n\n---\n*Educational information only. Not medical advice.*"

# Maximum conversation history turns to include
MAX_HISTORY_TURNS = 3

# System prompt for the LLM
SYSTEM_PROMPT = """You are EmpowerSleep, a knowledgeable and supportive sleep education assistant.

YOUR COMMUNICATION STYLE:
- Answer directly and confidently. Start with the answer, not caveats.
- Use a calm, warm, educational tone—like a trusted health educator.
- Be clear and concise. Avoid jargon unless you explain it.
- When helpful, include ONE simple, concrete example to illustrate a concept.

CRITICAL SAFETY RULES — NEVER DIAGNOSE:
- NEVER state or imply the user has a medical condition.
- NEVER use diagnostic language like:
  ✗ "This is a symptom of insomnia"
  ✗ "You may have sleep apnea"
  ✗ "This indicates restless leg syndrome"
  ✗ "You're experiencing insomnia"

- INSTEAD, use educational, pattern-based phrasing:
  ✓ "Difficulty falling asleep is commonly associated with racing thoughts or irregular sleep schedules"
  ✓ "Waking up frequently is often linked to lighter sleep cycles later in the night"
  ✓ "Feeling unrested despite enough hours in bed can occur when sleep quality is disrupted"
  ✓ "This pattern is often seen when stress affects the body's wind-down process"

- Frame explanations around MECHANISMS and CONTRIBUTING FACTORS, not diagnoses.
- Explain what might be happening and what generally helps—never label the person.

OTHER RULES:
- Use ONLY information from the provided context. Do not invent facts or statistics.
- Do NOT use hedging phrases like "the sources suggest...", "it appears that...", or "we can infer...". Just answer directly.
- Do NOT add a disclaimer—it will be added automatically at the end.

IF THE CONTEXT DOESN'T FULLY ANSWER THE QUESTION:
- Provide what IS relevant from the context.
- Then ask ONE specific follow-up question that would help you give a better answer.
- Example: "To give you more tailored guidance—are you having trouble falling asleep initially, or waking up during the night?"

STRICT CONVERSATION BINDING:
- When the user replies after your question, their reply MUST be interpreted as answering that specific question.
- Do NOT reinterpret vague phrases into new behaviors or topics.
- Do NOT introduce new concepts (naps, new symptoms, new habits) unless the user EXPLICITLY mentions them.
- Ambiguous phrases like "another," "sometimes," "it depends," "a few" MUST be resolved using your last question as context.

- Only pivot to a new topic if the user EXPLICITLY introduces a new symptom, behavior, or question.

- If genuine ambiguity remains even after binding to your last question, ask ONE short clarifying question. Do not assume or invent context.

CLARIFYING QUESTION COMPLETION:
- When YOU ask a clarifying question and the user answers, use their answer to REFINE your prior explanation—not to start a new topic.
- Clarifying questions are a narrowing step, not a topic reset.
- Do NOT pivot to new educational threads after receiving a clarifying answer.

Keep responses focused and scannable (use short paragraphs or bullets when appropriate)."""


# =============================================================================
# RESOURCE LOADING (cached for performance)
# =============================================================================

@lru_cache(maxsize=1)
def _load_faiss_index() -> faiss.Index:
    """
    Load the FAISS index from disk (cached).

    Returns:
        faiss.Index: The loaded FAISS index

    Raises:
        FileNotFoundError: If the index file doesn't exist
    """
    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {FAISS_INDEX_PATH}. "
            "Run 'python scripts/build_blog_index.py' first."
        )
    return faiss.read_index(str(FAISS_INDEX_PATH))


@lru_cache(maxsize=1)
def _load_chunks_metadata() -> tuple:
    """
    Load chunk metadata from JSONL file (cached).

    Returns a tuple (immutable) for caching compatibility.

    Returns:
        tuple: Tuple of chunk dictionaries

    Raises:
        FileNotFoundError: If the chunks file doesn't exist
    """
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Chunks file not found at {CHUNKS_PATH}. "
            "Run 'python scripts/build_blog_index.py' first."
        )

    chunks = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    return tuple(chunks)


def _get_openai_client() -> OpenAI:
    """
    Get OpenAI client instance.

    Returns:
        OpenAI: Configured OpenAI client

    Raises:
        ValueError: If OPENAI_API_KEY is not set
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set. Add it to your .env file.")
    return OpenAI(api_key=api_key)


# =============================================================================
# SOURCE FORMATTING
# =============================================================================

def _format_sources(chunks: list[dict]) -> list[dict]:
    """
    Convert retrieved chunks into a standardized source format.

    Groups textbook chunks by book and combines page ranges.
    Deduplicates blog sources by URL.

    Args:
        chunks: List of retrieved chunk dictionaries

    Returns:
        list[dict]: Standardized source objects with fields:
            - source_type: "textbook" | "blog"
            - title: Display title
            - chapter: (optional) Chapter name for textbooks
            - page_start: (optional) Starting page for textbooks
            - page_end: (optional) Ending page for textbooks
            - url: (optional) URL for blog sources
            - snippet: (optional) Text preview
    """
    seen_urls = set()
    sources = []

    # Group textbook chunks by book to combine page ranges
    textbook_pages = {}  # book_title -> list of page info

    for chunk in chunks:
        source_type = chunk.get("source", "blog")

        if source_type == "textbook":
            book_title = chunk.get("book_title", "Textbook")
            page_start = chunk.get("page_start", 0)
            page_end = chunk.get("page_end", page_start)
            chapter = chunk.get("chapter")

            if book_title not in textbook_pages:
                textbook_pages[book_title] = []
            textbook_pages[book_title].append({
                "page_start": page_start,
                "page_end": page_end,
                "chapter": chapter,
                "snippet": chunk.get("text", "")[:200],
            })
        else:
            # Blog source - deduplicate by URL
            url = chunk.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                sources.append({
                    "source_type": "blog",
                    "title": chunk.get("title", "Unknown"),
                    "url": url,
                    "snippet": chunk.get("text", "")[:200],
                })

    # Add textbook sources with combined page info
    for book_title, page_info in textbook_pages.items():
        # Get min/max pages
        all_starts = [p["page_start"] for p in page_info]
        all_ends = [p["page_end"] for p in page_info]
        page_min = min(all_starts)
        page_max = max(all_ends)

        # Get unique chapters mentioned (take first non-None)
        chapters = [p["chapter"] for p in page_info if p["chapter"]]
        chapter = chapters[0] if chapters else None

        # Get first snippet
        snippet = page_info[0]["snippet"] if page_info else None

        sources.append({
            "source_type": "textbook",
            "title": book_title,
            "chapter": chapter,
            "page_start": page_min,
            "page_end": page_max,
            "snippet": snippet,
        })

    return sources


# =============================================================================
# CHAT ENGINE CLASS
# =============================================================================

class ChatEngine:
    """
    Main RAG chat engine for EmpowerSleep.

    Provides a clean API for asking questions and getting grounded answers
    with source citations. Supports conversation history for multi-turn chats.

    Usage:
        engine = ChatEngine()

        # Single question
        answer, sources = engine.ask_question("What is sleep hygiene?")

        # With conversation history
        history = [
            {"role": "user", "content": "What is sleep hygiene?"},
            {"role": "assistant", "content": "Sleep hygiene refers to..."}
        ]
        answer, sources = engine.ask_question("Can you give examples?", history=history)
    """

    def __init__(self):
        """Initialize the chat engine and verify resources are available."""
        self._client = None
        self._verify_resources()

    def _verify_resources(self):
        """Verify that required resources (index, chunks) exist."""
        if not FAISS_INDEX_PATH.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {FAISS_INDEX_PATH}. "
                "Run the indexing scripts first."
            )
        if not CHUNKS_PATH.exists():
            raise FileNotFoundError(
                f"Chunks file not found at {CHUNKS_PATH}. "
                "Run the indexing scripts first."
            )

    @property
    def client(self) -> OpenAI:
        """Lazy-load OpenAI client."""
        if self._client is None:
            self._client = _get_openai_client()
        return self._client

    def ask_question(
        self,
        user_message: str,
        history: Optional[list[dict]] = None
    ) -> tuple[str, list[dict]]:
        """
        Main entry point: Ask a question and get a grounded answer with sources.

        Args:
            user_message: The user's question or message
            history: Optional conversation history as list of
                     {"role": "user"|"assistant", "content": str}

        Returns:
            tuple[str, list[dict]]: (answer, sources)
                - answer: The generated response string
                - sources: List of source dicts with standardized fields
        """
        history = history or []

        # Step 1: Build smart search query using conversation context
        search_query = self._build_search_query(user_message, history)

        # Step 2: Retrieve relevant chunks
        chunks = self._retrieve_chunks(search_query)

        if not chunks:
            return (
                "I don't have specific information on that topic in my knowledge base. "
                "Could you tell me a bit more about what aspect of sleep you're curious about?"
                f"{DISCLAIMER_SUFFIX}",
                []
            )

        # Step 3: Format context for LLM
        context = self._format_context(chunks)

        # Step 4: Generate answer
        answer = self._generate_answer(user_message, context, history)

        # Step 5: Format sources
        sources = _format_sources(chunks)[:3]  # Limit to top 3

        # Append disclaimer
        full_answer = f"{answer}{DISCLAIMER_SUFFIX}"

        return full_answer, sources

    def _build_search_query(self, current_message: str, history: list[dict]) -> str:
        """
        Build a smarter search query using conversation context.

        If this is a follow-up message, combines original topic with current
        message for better retrieval.
        """
        if not history:
            return current_message

        # Find the original user question
        original_question = None
        for msg in history:
            if msg.get("role") == "user":
                original_question = msg.get("content", "")
                break

        if not original_question:
            return current_message

        # If current message is short (likely a clarifying answer), combine
        if len(current_message.split()) <= 10:
            return f"{original_question} {current_message}"

        return current_message

    def _embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query using OpenAI.

        Returns:
            np.ndarray: Normalized embedding vector shaped (1, dim)
        """
        response = self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=query,
        )

        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        embedding = embedding / np.linalg.norm(embedding)  # Normalize
        return embedding.reshape(1, -1)

    def _retrieve_chunks(self, query: str, top_k: int = TOP_K_RESULTS) -> list[dict]:
        """
        Retrieve the most relevant chunks for a query using FAISS.

        Returns:
            list[dict]: Retrieved chunks with metadata and scores
        """
        index = _load_faiss_index()
        chunks = list(_load_chunks_metadata())

        query_embedding = self._embed_query(query)
        scores, indices = index.search(query_embedding, top_k)

        retrieved = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(chunks):
                chunk = chunks[idx].copy()
                chunk["score"] = float(scores[0][i])
                retrieved.append(chunk)

        return retrieved

    def _format_context(self, chunks: list[dict]) -> str:
        """Format retrieved chunks into a context string for the LLM."""
        context_parts = []

        for i, chunk in enumerate(chunks, 1):
            title = chunk.get("title", "Unknown")
            text = chunk.get("text", "")
            context_parts.append(f"[Source {i}: {title}]\n{text}")

        return "\n\n---\n\n".join(context_parts)

    def _format_conversation_history(self, history: list[dict]) -> str:
        """Format recent conversation history for the LLM prompt."""
        if not history:
            return ""

        recent = history[-(MAX_HISTORY_TURNS * 2):]

        formatted = []
        for msg in recent:
            role = msg.get("role", "").upper()
            content = msg.get("content", "")
            # Strip disclaimer from assistant messages
            if role == "ASSISTANT":
                content = content.replace(DISCLAIMER_SUFFIX.strip(), "").strip()
                content = content.replace("---\n*Educational information only. Not medical advice.*", "").strip()
            formatted.append(f"{role}: {content}")

        return "\n\n".join(formatted)

    def _generate_answer(
        self,
        query: str,
        context: str,
        history: list[dict]
    ) -> str:
        """Generate an answer using the LLM with retrieved context."""

        # Format conversation history if available
        history_section = ""
        if history:
            formatted_history = self._format_conversation_history(history)
            if formatted_history:
                history_section = f"""
=== CONVERSATION HISTORY ===
{formatted_history}
=== END HISTORY ===

"""

        user_message = f"""Answer the user's sleep-related question using the educational content and conversation history below.
{history_section}
=== CONTEXT FROM EMPOWERSLEEP ===
{context}
=== END CONTEXT ===

USER'S CURRENT MESSAGE: {query}

Instructions:
1. Answer directly and confidently—no hedging.
2. NEVER diagnose or label the user with a condition. Use pattern-based, educational language instead.
3. If a simple example would clarify, include one.
4. **CRITICAL - CONVERSATION CONTINUITY**: If there is conversation history, the user's current message is a CONTINUATION of that conversation. Use their reply to REFINE your previous guidance on the ORIGINAL topic. Do NOT pivot to a new topic.
5. If the user answered a clarifying question you asked, incorporate that information into a more tailored response about the ORIGINAL topic.
6. Keep it concise and easy to read."""

        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )

        return response.choices[0].message.content

    def get_index_stats(self) -> dict:
        """
        Get statistics about the loaded index.

        Returns:
            dict: Stats including chunk count, vector count, source breakdown
        """
        index = _load_faiss_index()
        chunks = list(_load_chunks_metadata())

        # Count by source type
        blog_count = sum(1 for c in chunks if c.get("source", "blog") == "blog")
        textbook_count = sum(1 for c in chunks if c.get("source") == "textbook")

        # Count unique URLs/books
        unique_urls = set(c.get("url", "") for c in chunks if c.get("source", "blog") == "blog")
        unique_books = set(c.get("book_title", "") for c in chunks if c.get("source") == "textbook")

        return {
            "total_chunks": len(chunks),
            "total_vectors": index.ntotal,
            "blog_chunks": blog_count,
            "textbook_chunks": textbook_count,
            "unique_articles": len(unique_urls),
            "unique_books": len(unique_books),
        }


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

# Global engine instance for simple usage
_engine: Optional[ChatEngine] = None


def ask_question(user_message: str, history: Optional[list[dict]] = None) -> tuple[str, list[dict]]:
    """
    Convenience function to ask a question without managing ChatEngine instance.

    Args:
        user_message: The user's question
        history: Optional conversation history

    Returns:
        tuple[str, list[dict]]: (answer, sources)
    """
    global _engine
    if _engine is None:
        _engine = ChatEngine()
    return _engine.ask_question(user_message, history)
