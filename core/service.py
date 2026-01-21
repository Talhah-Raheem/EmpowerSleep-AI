"""
core/service.py - Main Service Layer
=====================================

This module provides the main API for the chatbot. The Streamlit app
(or any other UI) should ONLY call functions from this module.

The key function is `answer_question()` which:
1. Checks for crisis/urgent triage
2. Retrieves relevant context from the vector store
3. Checks if we have enough context (guardrail)
4. Generates a grounded response using the LLM
5. Returns a structured response with answer, triage level, and sources

This separation keeps the UI thin and business logic testable.
"""

import sys
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI

from config import (
    OPENAI_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    MIN_CONTEXT_LENGTH,
    TOP_K_RESULTS,
)
from core.safety import check_triage, get_standard_disclaimer, TriageResult
from core.vectorstore import similarity_search, RetrievalResult


# =============================================================================
# RESPONSE DATA STRUCTURE
# =============================================================================

@dataclass
class ChatResponse:
    """
    Structured response from the chatbot.

    This is what the UI receives and displays.

    Attributes:
        answer: The text response to show the user
        triage_level: "CRISIS", "URGENT", or None
        sources: List of source filenames used (empty if triage triggered)
        context_sufficient: Whether we had enough context to answer
    """
    answer: str
    triage_level: Optional[str] = None
    sources: list[str] = field(default_factory=list)
    context_sufficient: bool = True


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

# System prompt that enforces grounded responses
SYSTEM_PROMPT = """You are a helpful sleep education assistant. Your role is to provide accurate, educational information about sleep based ONLY on the provided context.

CRITICAL RULES:
1. ONLY use information from the provided context to answer questions
2. If the context doesn't contain relevant information, say so clearly
3. Never make up facts or statistics that aren't in the context
4. Always maintain a supportive, educational tone
5. Remind users you are not a medical professional when appropriate
6. Keep responses concise and easy to understand

You are helping people learn about sleep health, not diagnosing or treating conditions."""

# Template for the user message with context
USER_PROMPT_TEMPLATE = """Based on the following educational content, please answer the user's question.

=== EDUCATIONAL CONTENT ===
{context}
=== END CONTENT ===

User's question: {question}

Remember: Only use information from the content above. If the content doesn't address the question, acknowledge that limitation."""


# Fallback message when context is insufficient
INSUFFICIENT_CONTEXT_MESSAGE = """I don't have enough information in my knowledge base to answer that question safely and accurately.

**What I can help with:**
- General sleep hygiene tips
- Understanding circadian rhythms
- Common sleep myths and facts

**For your question, I recommend:**
- Consulting a healthcare provider or sleep specialist
- Checking reputable sources like the Sleep Foundation or CDC

Is there something else about sleep education I can help you with?"""


# =============================================================================
# LLM INTEGRATION
# =============================================================================

def _get_openai_client() -> OpenAI:
    """Get OpenAI client instance."""
    if not OPENAI_API_KEY:
        raise ValueError(
            "OpenAI API key not set. "
            "Please set OPENAI_API_KEY in your .env file or environment."
        )
    return OpenAI(api_key=OPENAI_API_KEY)


def _generate_response(question: str, context: str) -> str:
    """
    Generate a response using the LLM.

    Args:
        question: The user's question
        context: Retrieved context to ground the response

    Returns:
        The LLM's response text
    """
    client = _get_openai_client()

    user_message = USER_PROMPT_TEMPLATE.format(
        context=context,
        question=question
    )

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )

    return response.choices[0].message.content


# =============================================================================
# MAIN API FUNCTION
# =============================================================================

def answer_question(user_text: str) -> ChatResponse:
    """
    Process a user question and return a structured response.

    This is the MAIN ENTRY POINT that the UI should call.

    Flow:
    1. Check for crisis/urgent triage → return immediately if triggered
    2. Search vector store for relevant context
    3. Check if context is sufficient (>= MIN_CONTEXT_LENGTH chars)
    4. If insufficient → return safe fallback
    5. If sufficient → generate LLM response grounded in context
    6. Return structured response with sources

    Args:
        user_text: The user's raw input/question

    Returns:
        ChatResponse with answer, triage_level, sources, and context_sufficient
    """
    # -------------------------------------------------------------------------
    # Step 1: Safety/Triage Check
    # -------------------------------------------------------------------------
    triage_result = check_triage(user_text)

    if triage_result is not None:
        # Return triage response immediately - don't proceed to RAG
        return ChatResponse(
            answer=triage_result.message,
            triage_level=triage_result.level,
            sources=[],  # No sources for triage responses
            context_sufficient=True  # N/A for triage
        )

    # -------------------------------------------------------------------------
    # Step 2: Retrieve Relevant Context
    # -------------------------------------------------------------------------
    try:
        retrieval_result = similarity_search(user_text, top_k=TOP_K_RESULTS)
    except FileNotFoundError as e:
        # Index not built yet
        return ChatResponse(
            answer=(
                "⚠️ The knowledge base hasn't been set up yet. "
                "Please run `python scripts/build_index.py` first."
            ),
            triage_level=None,
            sources=[],
            context_sufficient=False
        )

    # -------------------------------------------------------------------------
    # Step 3: Check Context Sufficiency (Guardrail)
    # -------------------------------------------------------------------------
    if retrieval_result.total_context_length < MIN_CONTEXT_LENGTH:
        # Not enough context to answer safely
        return ChatResponse(
            answer=INSUFFICIENT_CONTEXT_MESSAGE,
            triage_level=None,
            sources=retrieval_result.sources,  # Show what we found anyway
            context_sufficient=False
        )

    # -------------------------------------------------------------------------
    # Step 4: Generate Grounded Response
    # -------------------------------------------------------------------------
    context_string = retrieval_result.get_context_string()

    try:
        llm_response = _generate_response(user_text, context_string)
    except ValueError as e:
        # API key not set
        return ChatResponse(
            answer=f"⚠️ Configuration error: {str(e)}",
            triage_level=None,
            sources=[],
            context_sufficient=True
        )
    except Exception as e:
        # Other API errors
        return ChatResponse(
            answer=f"⚠️ Error generating response: {str(e)}",
            triage_level=None,
            sources=[],
            context_sufficient=True
        )

    # -------------------------------------------------------------------------
    # Step 5: Return Successful Response
    # -------------------------------------------------------------------------
    # Prepend the disclaimer to the response
    disclaimer = get_standard_disclaimer()
    full_response = f"{disclaimer}\n\n{llm_response}"

    return ChatResponse(
        answer=full_response,
        triage_level=None,
        sources=retrieval_result.sources,
        context_sufficient=True
    )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def check_api_key() -> bool:
    """Check if OpenAI API key is configured."""
    return bool(OPENAI_API_KEY)


def get_available_sources() -> list[str]:
    """Get list of all source documents in the index."""
    try:
        # Do a broad search to get sources
        result = similarity_search("sleep", top_k=100)
        return result.sources
    except FileNotFoundError:
        return []


# =============================================================================
# TESTING / DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SERVICE LAYER DEMO")
    print("=" * 60)

    # Test triage
    print("\n--- Testing Triage ---")
    response = answer_question("I'm thinking about suicide")
    print(f"Triage level: {response.triage_level}")
    print(f"Answer preview: {response.answer[:100]}...")

    # Test normal question (requires index and API key)
    print("\n--- Testing Normal Question ---")
    if not check_api_key():
        print("⚠️  OpenAI API key not set - skipping LLM test")
    else:
        response = answer_question("What is sleep hygiene?")
        print(f"Triage level: {response.triage_level}")
        print(f"Sources: {response.sources}")
        print(f"Context sufficient: {response.context_sufficient}")
        print(f"Answer preview: {response.answer[:200]}...")
