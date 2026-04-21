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

import base64
import json as json_module
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from supabase import create_client, Client

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# Import the chat engine
from rag.chat_engine import ChatEngine

logger = logging.getLogger(__name__)

# =============================================================================
# ENVIRONMENT
# =============================================================================

APP_ENV = os.getenv("APP_ENV", "production")
IS_PRODUCTION = APP_ENV == "production"

# Rate limit strings — relaxed in development so local testing isn't blocked
CHAT_RATE_LIMIT = "10/minute" if IS_PRODUCTION else "200/minute"
SUGGESTIONS_RATE_LIMIT = "20/minute" if IS_PRODUCTION else "400/minute"
FEEDBACK_RATE_LIMIT = "30/minute" if IS_PRODUCTION else "600/minute"


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class ChatMessage(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., min_length=1, max_length=2000)
    history: Optional[list[dict]] = Field(default=None)


class Source(BaseModel):
    """Source citation model."""
    source_type: str
    title: str
    chapter: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    url: Optional[str] = None
    snippet: Optional[str] = None


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    answer: str
    sources: list[Source] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    index_loaded: bool
    total_chunks: Optional[int] = None
    total_vectors: Optional[int] = None


class FeedbackRequest(BaseModel):
    """Request model for feedback endpoint."""
    session_id: str
    user_question: str
    ai_response: str
    rating: int  # 1 = thumbs up, -1 = thumbs down


class SuggestionsRequest(BaseModel):
    """Request model for follow-up suggestions endpoint."""
    message: str
    response: str
    history: list[dict] = []


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="EmpowerSleep API",
    description="RAG-powered sleep education chatbot API",
    version="1.0.0",
    # Disable interactive docs in production — they expose full API structure
    docs_url=None if IS_PRODUCTION else "/docs",
    redoc_url=None if IS_PRODUCTION else "/redoc",
    openapi_url=None if IS_PRODUCTION else "/openapi.json",
)

# Rate limiter (per-IP)
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS configuration
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://empowersleep.com",
    "https://www.empowersleep.com",
    "https://empowersleep.ai",
    "https://www.empowersleep.ai",
]

extra_origins = os.getenv("CORS_ORIGINS", "")
if extra_origins:
    ALLOWED_ORIGINS.extend([origin.strip() for origin in extra_origins.split(",")])

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# EXCEPTION HANDLERS
# =============================================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return a generic 422 without leaking Pydantic model structure."""
    return JSONResponse(status_code=422, content={"detail": "Invalid request"})


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Catch-all — log full details server-side, return a safe generic message."""
    logger.error(
        "Unhandled exception on %s %s: %s",
        request.method,
        request.url.path,
        exc,
        exc_info=True,
    )
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# =============================================================================
# GLOBAL INSTANCES
# =============================================================================

_chat_engine: Optional[ChatEngine] = None
_supabase: Optional[Client] = None


def get_supabase() -> Client:
    """Get or create the Supabase client instance."""
    global _supabase
    if _supabase is None:
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SECRET_KEY")
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_SECRET_KEY must be set")
        _supabase = create_client(url, key)
    return _supabase


def get_chat_engine() -> ChatEngine:
    """Get or create the chat engine instance."""
    global _chat_engine
    if _chat_engine is None:
        _chat_engine = ChatEngine()
    return _chat_engine


def _safe_history(raw: list[dict]) -> list[dict]:
    """Strip 'system' role messages from client-provided history.

    Clients can inject system-role messages to attempt prompt injection.
    Only 'user' and 'assistant' turns are legitimate conversation history.
    """
    return [m for m in raw if m.get("role") in ("user", "assistant")]


# =============================================================================
# FILE PROCESSING HELPERS
# =============================================================================

MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB per file
MAX_PDF_CHARS = 100_000                  # ~25k tokens


def extract_pdf_text(content: bytes) -> str:
    """Extract all text from a PDF given its raw bytes. Capped at MAX_PDF_CHARS."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=content, filetype="pdf")
        parts = [page.get_text() for page in doc]
        return "\n\n".join(parts)[:MAX_PDF_CHARS]
    except Exception:
        return ""


async def process_uploaded_files(files: List[UploadFile]) -> list[dict]:
    """
    Read uploaded files and return a list of file_context dicts:
      - PDF:   {"type": "pdf",   "filename": ..., "text": ...}
      - Image: {"type": "image", "filename": ..., "base64": ..., "mime_type": ...}
    Files that are too large or unreadable are silently skipped.
    """
    file_context: list[dict] = []
    for upload in files:
        if not upload.filename:
            continue
        content = await upload.read()
        if len(content) > MAX_FILE_SIZE_BYTES:
            continue  # skip files over the size limit
        ct = upload.content_type or ""
        if ct == "application/pdf" or upload.filename.lower().endswith(".pdf"):
            text = extract_pdf_text(content)
            if text.strip():
                file_context.append({"type": "pdf", "filename": upload.filename, "text": text})
        elif ct.startswith("image/"):
            b64 = base64.b64encode(content).decode("utf-8")
            file_context.append({
                "type": "image",
                "filename": upload.filename,
                "base64": b64,
                "mime_type": ct,
            })
    return file_context


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    try:
        engine = get_chat_engine()
        stats = engine.get_index_stats()
        # Avoid leaking internal counts in production
        if IS_PRODUCTION:
            return HealthResponse(status="healthy", index_loaded=True)
        return HealthResponse(
            status="healthy",
            index_loaded=True,
            total_chunks=stats["total_chunks"],
            total_vectors=stats["total_vectors"],
        )
    except FileNotFoundError:
        return HealthResponse(status="degraded", index_loaded=False)
    except Exception:
        return HealthResponse(status="unhealthy", index_loaded=False)


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
@limiter.limit(CHAT_RATE_LIMIT)
async def chat(request: Request, body: ChatMessage):
    """
    Send a message and get a response with sources (non-streaming).
    """
    try:
        engine = get_chat_engine()
        safe_hist = _safe_history(body.history or [])
        answer, sources = engine.ask_question(
            user_message=body.message,
            history=safe_hist,
        )
        source_models = [
            Source(
                source_type=src.get("source_type", "blog"),
                title=src.get("title", "Unknown"),
                chapter=src.get("chapter"),
                page_start=src.get("page_start"),
                page_end=src.get("page_end"),
                url=src.get("url"),
                snippet=src.get("snippet"),
            )
            for src in sources
        ]
        return ChatResponse(answer=answer, sources=source_models)
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Knowledge base not loaded. Please run indexing scripts first.",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/chat/stream", tags=["Chat"])
@limiter.limit(CHAT_RATE_LIMIT)
async def chat_stream(
    request: Request,
    message: str = Form(...),
    history: str = Form("[]"),
    files: Optional[List[UploadFile]] = File(None),
):
    """
    Send a message and get a streaming response via Server-Sent Events (SSE).

    Accepts multipart/form-data with optional file attachments (PDF + images).

    Events emitted:
    - ``{"type": "sources", "sources": [...]}`` — sent before generation starts
    - ``{"type": "token", "text": "..."}``       — one per streamed token
    - ``[DONE]``                                  — signals end of stream
    - ``{"type": "error", "message": "..."}``     — on failure
    """
    engine = get_chat_engine()

    # Parse and sanitize history
    try:
        history_list: list[dict] = json_module.loads(history)
    except Exception:
        history_list = []

    safe_hist = _safe_history(history_list)

    # Process any uploaded files
    file_context: Optional[list[dict]] = None
    if files:
        processed = await process_uploaded_files([f for f in files if f.filename])
        if processed:
            file_context = processed

    async def event_generator():
        try:
            async for event_type, data in engine.stream_question(
                user_message=message,
                history=safe_hist,
                file_context=file_context,
            ):
                if event_type == "sources":
                    payload = json_module.dumps({"type": "sources", "sources": data})
                    yield f"data: {payload}\n\n"
                elif event_type == "metrics":
                    payload = json_module.dumps({"type": "metrics", "data": data})
                    yield f"data: {payload}\n\n"
                elif event_type == "token":
                    payload = json_module.dumps({"type": "token", "text": data})
                    yield f"data: {payload}\n\n"
                elif event_type == "done":
                    yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error("Stream error: %s", e, exc_info=True)
            payload = json_module.dumps({"type": "error", "message": "An error occurred. Please try again."})
            yield f"data: {payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/feedback", tags=["Feedback"])
@limiter.limit(FEEDBACK_RATE_LIMIT)
async def submit_feedback(request: Request, body: FeedbackRequest):
    """
    Submit feedback (thumbs up/down) for an AI response.
    Rating: 1 = thumbs up, -1 = thumbs down.
    """
    try:
        db = get_supabase()
        db.table("feedback").insert({
            "session_id": body.session_id,
            "user_question": body.user_question,
            "ai_response": body.ai_response,
            "rating": body.rating,
            "environment": os.environ.get("APP_ENV", "production"),
        }).execute()
        return {"status": "ok"}
    except ValueError:
        raise HTTPException(status_code=503, detail="Feedback service unavailable.")
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to save feedback.")


@app.post("/suggestions", tags=["Chat"])
@limiter.limit(SUGGESTIONS_RATE_LIMIT)
async def get_suggestions(request: Request, body: SuggestionsRequest):
    """
    Generate 3 follow-up question suggestions based on the conversation.
    """
    try:
        engine = get_chat_engine()
        client = engine.async_client  # use the property so it initialises if needed

        history_text = ""
        for turn in body.history[-4:]:
            role = turn.get("role", "")
            content = turn.get("content", "")
            history_text += f"{role.capitalize()}: {content}\n\n"

        prompt = (
            "Based on this sleep-related conversation, suggest exactly 3 short follow-up questions "
            "the user might want to ask next. Questions should dig deeper into what was just discussed.\n\n"
            f"{history_text}"
            f"User: {body.message}\n"
            f"Assistant: {body.response}\n\n"
            "Return exactly 3 questions, one per line, no numbering, no bullet points. "
            "Keep each question under 12 words."
        )

        completion = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=120,
            timeout=10,
        )

        text = completion.choices[0].message.content or ""
        questions = [q.strip() for q in text.strip().split("\n") if q.strip()][:3]
        return {"suggestions": questions}
    except Exception:
        return {"suggestions": []}


@app.get("/stats", tags=["System"])
async def get_stats(
    request: Request,
    x_admin_key: Optional[str] = Header(default=None),
):
    """
    Get statistics about the knowledge base.
    Requires X-Admin-Key header in production.
    """
    if IS_PRODUCTION:
        admin_key = os.environ.get("ADMIN_STATS_KEY")
        if not admin_key or x_admin_key != admin_key:
            raise HTTPException(status_code=404, detail="Not found")
    try:
        engine = get_chat_engine()
        return engine.get_index_stats()
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
