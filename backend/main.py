"""
backend/main.py
===============

FastAPI backend for EmpowerSleep RAG chatbot.

Provides REST API endpoints for the chat interface:
- POST /chat - Send a message and get a response with sources
- GET /health - Health check endpoint

Run with:
    uvicorn backend.main:app --reload --port 8000

Or from project root:
    python -m uvicorn backend.main:app --reload --port 8000
"""

import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# Import the chat engine
from rag.chat_engine import ChatEngine, ask_question


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class ChatMessage(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., min_length=1, max_length=2000, description="User's message")
    history: Optional[list[dict]] = Field(
        default=None,
        description="Conversation history as list of {role, content} dicts"
    )


class Source(BaseModel):
    """Source citation model."""
    source_type: str = Field(..., description="Type of source: 'textbook', 'blog', or 'web'")
    title: str = Field(..., description="Title of the source")
    chapter: Optional[str] = Field(None, description="Chapter name (textbooks only)")
    page_start: Optional[int] = Field(None, description="Starting page (textbooks only)")
    page_end: Optional[int] = Field(None, description="Ending page (textbooks only)")
    url: Optional[str] = Field(None, description="URL (blog/web sources only)")
    snippet: Optional[str] = Field(None, description="Text preview from source")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    answer: str = Field(..., description="Generated answer")
    sources: list[Source] = Field(default_factory=list, description="Source citations")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    index_loaded: bool
    total_chunks: Optional[int] = None
    total_vectors: Optional[int] = None


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="EmpowerSleep API",
    description="RAG-powered sleep education chatbot API",
    version="1.0.0",
)

# CORS configuration
# Allow requests from local development and production frontend
ALLOWED_ORIGINS = [
    "http://localhost:3000",      # Next.js dev server
    "http://127.0.0.1:3000",
    "https://empowersleep.com",   # Production placeholder
    "https://www.empowersleep.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# Global chat engine instance (initialized on first request)
_chat_engine: Optional[ChatEngine] = None


def get_chat_engine() -> ChatEngine:
    """Get or create the chat engine instance."""
    global _chat_engine
    if _chat_engine is None:
        _chat_engine = ChatEngine()
    return _chat_engine


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.

    Returns the status of the API and index loading state.
    """
    try:
        engine = get_chat_engine()
        stats = engine.get_index_stats()
        return HealthResponse(
            status="healthy",
            index_loaded=True,
            total_chunks=stats["total_chunks"],
            total_vectors=stats["total_vectors"],
        )
    except FileNotFoundError as e:
        return HealthResponse(
            status="degraded",
            index_loaded=False,
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            index_loaded=False,
        )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatMessage):
    """
    Send a message and get a response with sources.

    The endpoint:
    1. Retrieves relevant content from the knowledge base
    2. Generates a grounded answer using LLM
    3. Returns the answer with source citations

    For multi-turn conversations, include the conversation history
    in the request to maintain context.
    """
    try:
        engine = get_chat_engine()

        # Call the RAG pipeline
        answer, sources = engine.ask_question(
            user_message=request.message,
            history=request.history
        )

        # Convert sources to response model
        source_models = []
        for src in sources:
            source_models.append(Source(
                source_type=src.get("source_type", "blog"),
                title=src.get("title", "Unknown"),
                chapter=src.get("chapter"),
                page_start=src.get("page_start"),
                page_end=src.get("page_end"),
                url=src.get("url"),
                snippet=src.get("snippet"),
            ))

        return ChatResponse(answer=answer, sources=source_models)

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail="Knowledge base not loaded. Please run indexing scripts first."
        )
    except ValueError as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {str(e)}"
        )


@app.get("/stats", tags=["System"])
async def get_stats():
    """
    Get statistics about the knowledge base.

    Returns counts of chunks, vectors, and breakdown by source type.
    """
    try:
        engine = get_chat_engine()
        return engine.get_index_stats()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
