"""
app.py - Streamlit UI for EmpowerSleep RAG Chatbot
===================================================

This app uses a FAISS vector index built from EmpowerSleep blog articles
to answer sleep-related questions with RAG (Retrieval-Augmented Generation).

RAG FLOW:
1. User enters a question
2. Question is embedded using OpenAI text-embedding-3-small
3. FAISS similarity search retrieves top-k relevant chunks
4. Retrieved chunks are used as context for the LLM
5. LLM generates a grounded answer
6. Sources (blog titles + URLs) are displayed

Run with: streamlit run app.py
"""

import json
import os
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# =============================================================================
# CONFIGURATION
# =============================================================================

# Load environment variables from .env file
load_dotenv()

# Paths to RAG artifacts (built by scripts/build_blog_index.py)
RAG_ARTIFACTS_DIR = Path(__file__).parent / "rag_artifacts"
FAISS_INDEX_PATH = RAG_ARTIFACTS_DIR / "faiss.index"
CHUNKS_PATH = RAG_ARTIFACTS_DIR / "chunks.jsonl"

# Embedding configuration (must match build_blog_index.py)
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536

# Retrieval configuration
TOP_K_RESULTS = 4  # Number of chunks to retrieve

# LLM configuration
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 600

# Demo questions for quick testing
DEMO_QUESTIONS = [
    "What is sleep hygiene?",
    "How does caffeine affect sleep?",
    "What causes insomnia?",
]


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="EmpowerSleep - Sleep Education Assistant",
    page_icon="🌙",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
    /* Style for demo buttons */
    .stButton > button {
        width: 100%;
        text-align: left;
        padding: 10px 15px;
    }

    /* Source link styling */
    .source-link {
        padding: 8px 12px;
        margin: 4px 0;
        background-color: #f0f2f6;
        border-radius: 4px;
        display: block;
    }

    .source-link a {
        text-decoration: none;
        color: #0066cc;
    }

    .source-link a:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# RAG COMPONENTS - LOADING (cached for performance)
# =============================================================================

@st.cache_resource
def load_faiss_index():
    """
    Load the FAISS index from disk.

    This is cached so the index is only loaded once per session.
    The index contains embeddings of all blog article chunks.

    Returns:
        faiss.Index: The loaded FAISS index
    """
    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {FAISS_INDEX_PATH}. "
            "Run 'python scripts/build_blog_index.py' first."
        )

    index = faiss.read_index(str(FAISS_INDEX_PATH))
    return index


@st.cache_data
def load_chunks_metadata():
    """
    Load chunk metadata from JSONL file.

    Each chunk has: id, title, url, chunk_index, text
    This metadata is used to retrieve the actual text and source info
    after FAISS returns matching indices.

    Returns:
        list[dict]: List of chunk metadata dictionaries
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

    return chunks


def get_openai_client() -> OpenAI:
    """
    Get OpenAI client instance.

    Reads API key from environment variable OPENAI_API_KEY.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not set. Add it to your .env file."
        )
    return OpenAI(api_key=api_key)


# =============================================================================
# RAG COMPONENTS - EMBEDDING
# =============================================================================

def embed_query(query: str) -> np.ndarray:
    """
    Generate an embedding for a query using OpenAI.

    Uses text-embedding-3-small (1536 dimensions) to match
    the embeddings used when building the index.

    Args:
        query: The user's question

    Returns:
        np.ndarray: Normalized embedding vector (1536 dims)
    """
    client = get_openai_client()

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )

    # Extract embedding and convert to numpy
    embedding = np.array(response.data[0].embedding, dtype=np.float32)

    # Normalize for cosine similarity (FAISS IndexFlatIP expects normalized vectors)
    embedding = embedding / np.linalg.norm(embedding)

    # Reshape to (1, dim) for FAISS search
    return embedding.reshape(1, -1)


# =============================================================================
# RAG COMPONENTS - RETRIEVAL
# =============================================================================

def retrieve_relevant_chunks(query: str, top_k: int = TOP_K_RESULTS) -> list[dict]:
    """
    Retrieve the most relevant chunks for a query using FAISS.

    RAG Retrieval Flow:
    1. Embed the query using OpenAI
    2. Search FAISS index for similar vectors
    3. Map indices back to chunk metadata
    4. Return chunks with text and source info

    Args:
        query: The user's question
        top_k: Number of chunks to retrieve

    Returns:
        list[dict]: Retrieved chunks with {title, url, text, score}
    """
    # Load index and metadata (cached)
    index = load_faiss_index()
    chunks = load_chunks_metadata()

    # Generate query embedding
    query_embedding = embed_query(query)

    # Search FAISS index
    # Returns (distances, indices) - for IndexFlatIP, higher score = more similar
    scores, indices = index.search(query_embedding, top_k)

    # Map indices back to chunks
    retrieved = []
    for i, idx in enumerate(indices[0]):
        if idx < len(chunks) and idx >= 0:  # Valid index
            chunk = chunks[idx].copy()
            chunk["score"] = float(scores[0][i])
            retrieved.append(chunk)

    return retrieved


def format_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a context string for the LLM.

    Each chunk is formatted with its source title for attribution.

    Args:
        chunks: List of retrieved chunk dictionaries

    Returns:
        str: Formatted context string
    """
    context_parts = []

    for i, chunk in enumerate(chunks, 1):
        title = chunk.get("title", "Unknown")
        text = chunk.get("text", "")
        context_parts.append(f"[Source {i}: {title}]\n{text}")

    return "\n\n---\n\n".join(context_parts)


def get_unique_sources(chunks: list[dict]) -> list[dict]:
    """
    Deduplicate sources from retrieved chunks.

    Multiple chunks may come from the same article, so we
    deduplicate by URL to show each source only once.

    For textbook sources, we group by book and show page ranges.

    Args:
        chunks: List of retrieved chunks

    Returns:
        list[dict]: Unique sources with metadata for display
    """
    seen_urls = set()
    sources = []

    # Group textbook chunks by book to combine page ranges
    textbook_pages = {}  # book_title -> list of (page_start, page_end, chapter)

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
            })
        else:
            # Blog source - deduplicate by URL
            url = chunk.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                sources.append({
                    "source": "blog",
                    "title": chunk.get("title", "Unknown"),
                    "url": url
                })

    # Add textbook sources with combined page info
    for book_title, page_info in textbook_pages.items():
        # Get min/max pages
        all_starts = [p["page_start"] for p in page_info]
        all_ends = [p["page_end"] for p in page_info]
        page_min = min(all_starts)
        page_max = max(all_ends)

        # Get unique chapters mentioned
        chapters = list(set(p["chapter"] for p in page_info if p["chapter"]))

        sources.append({
            "source": "textbook",
            "book_title": book_title,
            "page_start": page_min,
            "page_end": page_max,
            "chapters": chapters,
        })

    return sources


def format_source_display(source: dict) -> str:
    """
    Format a source for display in the UI.

    Args:
        source: Source dict with metadata

    Returns:
        str: Formatted markdown string for display
    """
    if source.get("source") == "textbook":
        book_title = source.get("book_title", "Textbook")
        page_start = source.get("page_start", 0)
        page_end = source.get("page_end", page_start)
        chapters = source.get("chapters", [])

        # Format page range
        if page_start == page_end:
            pages = f"p. {page_start}"
        else:
            pages = f"pp. {page_start}-{page_end}"

        # Format chapter if available
        chapter_str = ""
        if chapters and len(chapters) == 1:
            chapter_str = f" - {chapters[0]}"
        elif chapters:
            # Multiple chapters - just show the first
            chapter_str = f" - {chapters[0]}"

        return f"📖 **{book_title}**{chapter_str} ({pages})"
    else:
        # Blog source
        title = source.get("title", "Unknown")
        url = source.get("url", "#")
        return f"[{title}]({url})"


# =============================================================================
# RAG COMPONENTS - GENERATION
# =============================================================================

# System prompt: confident, clear, educational, non-diagnostic
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
  Example: If you asked "How many times do you wake up at night?" and user says "another 2-3 times," interpret as "2-3 more awakenings" — NOT as introducing naps or a new topic.

- Only pivot to a new topic if the user EXPLICITLY introduces a new symptom, behavior, or question.
  Example of explicit shift: "Actually, I also want to ask about snoring."
  Example of NOT a shift: "Sometimes" after you asked about caffeine timing → still about caffeine.

- If genuine ambiguity remains even after binding to your last question, ask ONE short clarifying question. Do not assume or invent context.

CLARIFYING QUESTION COMPLETION:
- When YOU ask a clarifying question and the user answers, use their answer to REFINE your prior explanation—not to start a new topic.
- Clarifying questions are a narrowing step, not a topic reset.
- Do NOT pivot to new educational threads after receiving a clarifying answer.
  Example:
    You asked: "What time do you usually go to bed?"
    User says: "Around 10 PM"
    CORRECT: Use "10 PM" to refine guidance about the ORIGINAL topic (e.g., caffeine timing relative to their bedtime).
    WRONG: Pivot to general "bedtime habits" or "sleep schedules" as a new topic.

Keep responses focused and scannable (use short paragraphs or bullets when appropriate)."""

# Disclaimer placed at END of response (not beginning)
DISCLAIMER_SUFFIX = "\n\n---\n*Educational information only. Not medical advice.*"

# Maximum conversation history to include (user + assistant pairs)
MAX_HISTORY_TURNS = 3


def build_search_query(current_message: str, history: list[dict]) -> str:
    """
    Build a smarter search query using conversation context.

    If this is a follow-up message (history exists), combine the original
    topic with the current message for better retrieval.

    Args:
        current_message: The user's current message
        history: Previous messages in the conversation

    Returns:
        str: Augmented search query
    """
    if not history:
        return current_message

    # Find the original user question (first user message)
    original_question = None
    for msg in history:
        if msg.get("role") == "user":
            original_question = msg.get("content", "")
            break

    if not original_question:
        return current_message

    # If current message is short (likely a clarifying answer), combine with original topic
    if len(current_message.split()) <= 10:
        # Combine original question + current response for better retrieval
        return f"{original_question} {current_message}"

    return current_message


def format_conversation_history(history: list[dict], max_turns: int = MAX_HISTORY_TURNS) -> str:
    """
    Format recent conversation history for the LLM prompt.

    Args:
        history: Full conversation history
        max_turns: Maximum number of exchange pairs to include

    Returns:
        str: Formatted conversation history
    """
    if not history:
        return ""

    # Get last N turns (each turn = user + assistant)
    recent = history[-(max_turns * 2):]

    formatted = []
    for msg in recent:
        role = msg.get("role", "").upper()
        content = msg.get("content", "")
        # Strip the disclaimer from assistant messages for cleaner history
        if role == "ASSISTANT":
            content = content.replace(DISCLAIMER_SUFFIX.strip(), "").strip()
            content = content.replace("---\n*Educational information only. Not medical advice.*", "").strip()
        formatted.append(f"{role}: {content}")

    return "\n\n".join(formatted)


def generate_answer(query: str, context: str, history: list[dict] = None) -> str:
    """
    Generate an answer using the LLM with retrieved context and conversation history.

    Style: confident, educational, non-diagnostic.
    - Answers directly without hedging
    - Includes one example when helpful
    - Asks one clarifying question if context is incomplete
    - Uses conversation history to maintain topic continuity

    Args:
        query: The user's current message
        context: Formatted context from retrieved chunks
        history: Previous conversation messages (optional)

    Returns:
        str: The LLM's response (disclaimer added separately)
    """
    client = get_openai_client()

    # Format conversation history if available
    history_section = ""
    if history:
        formatted_history = format_conversation_history(history)
        if formatted_history:
            history_section = f"""
=== CONVERSATION HISTORY ===
{formatted_history}
=== END HISTORY ===

"""

    user_message = f"""Answer the user's sleep-related question using the educational content and conversation history below.
{history_section}
=== CONTEXT FROM EMPOWERSLEEP BLOG ===
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
# MAIN RAG PIPELINE
# =============================================================================

def answer_question(query: str, history: list[dict] = None) -> dict:
    """
    Main RAG pipeline: Retrieve relevant content and generate an answer.

    Complete RAG Flow:
    1. Build smart search query using conversation context
    2. Retrieve relevant chunks from FAISS index
    3. Format chunks as context for LLM
    4. Generate grounded answer using LLM + conversation history
    5. Extract unique sources for citation

    Args:
        query: The user's current message
        history: Previous conversation messages (optional, for context)

    Returns:
        dict: {answer: str, sources: list[dict]}
    """
    # Step 1: Build smart search query using conversation context
    # If user is answering a follow-up, combine with original topic for better retrieval
    search_query = build_search_query(query, history or [])

    # Step 2: Retrieve relevant chunks
    chunks = retrieve_relevant_chunks(search_query, top_k=TOP_K_RESULTS)

    if not chunks:
        return {
            "answer": "I don't have specific information on that topic in my knowledge base. Could you tell me a bit more about what aspect of sleep you're curious about? For example, are you interested in sleep habits, sleep environment, or something else?\n\n---\n*Educational information only. Not medical advice.*",
            "sources": []
        }

    # Step 3: Format context for LLM
    context = format_context(chunks)

    # Step 4: Generate answer with conversation history for continuity
    answer = generate_answer(query, context, history)

    # Step 5: Get unique sources (limit to top 3 most relevant)
    sources = get_unique_sources(chunks)[:3]

    # Append disclaimer at END (not beginning)
    full_answer = f"{answer}{DISCLAIMER_SUFFIX}"

    return {
        "answer": full_answer,
        "sources": sources
    }


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize session state variables for chat history."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None


init_session_state()


def reset_conversation():
    """Reset the conversation to start fresh."""
    st.session_state.messages = []
    st.session_state.pending_question = None


# =============================================================================
# CHECK SYSTEM READINESS
# =============================================================================

def check_system_ready() -> tuple[bool, str]:
    """
    Check if the RAG system is ready to use.

    Returns:
        tuple: (is_ready: bool, error_message: str)
    """
    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        return False, "OPENAI_API_KEY not set. Add it to your .env file."

    # Check FAISS index
    if not FAISS_INDEX_PATH.exists():
        return False, f"FAISS index not found. Run 'python scripts/build_blog_index.py' first."

    # Check chunks file
    if not CHUNKS_PATH.exists():
        return False, f"Chunks file not found. Run 'python scripts/build_blog_index.py' first."

    return True, ""


# =============================================================================
# HEADER
# =============================================================================

st.title("🌙 EmpowerSleep")
st.markdown("*Your Sleep Education Assistant*")

# System readiness check
is_ready, error_msg = check_system_ready()
if not is_ready:
    st.error(f"⚠️ {error_msg}")

st.markdown("---")


# =============================================================================
# DEMO QUESTION BUTTONS
# =============================================================================

st.markdown("**Try a demo question:**")

cols = st.columns(len(DEMO_QUESTIONS))

for i, question in enumerate(DEMO_QUESTIONS):
    with cols[i]:
        display_text = question[:40] + "..." if len(question) > 40 else question
        if st.button(display_text, key=f"demo_{i}", help=question):
            st.session_state.pending_question = question

st.markdown("---")


# =============================================================================
# CHAT HISTORY DISPLAY
# =============================================================================

def display_message(message: dict):
    """Display a single chat message with sources if applicable."""
    role = message["role"]
    content = message["content"]
    sources = message.get("sources", [])

    with st.chat_message(role):
        st.markdown(content)

        # Show sources for assistant messages
        if role == "assistant" and sources:
            with st.expander("📚 **Sources** (click to expand)"):
                for source in sources:
                    st.markdown(f"- {format_source_display(source)}")


# Display all messages in history
for message in st.session_state.messages:
    display_message(message)


# =============================================================================
# CHAT INPUT HANDLING
# =============================================================================

def process_user_input(user_input: str):
    """
    Process user input through the RAG pipeline with conversation context.

    Flow:
    1. Capture conversation history BEFORE adding new message
    2. Add user message to history
    3. Call RAG pipeline with history for context-aware retrieval and generation
    4. Display and store assistant response
    """
    # Capture existing history BEFORE adding the new message
    # This gives the RAG pipeline context about the ongoing conversation
    conversation_history = list(st.session_state.messages)

    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
    })

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response using RAG pipeline with conversation history
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            try:
                # Pass conversation history for context-aware retrieval and generation
                result = answer_question(user_input, history=conversation_history)
                answer = result["answer"]
                sources = result["sources"]
            except Exception as e:
                answer = f"⚠️ Error: {str(e)}"
                sources = []

        # Display the answer
        st.markdown(answer)

        # Display sources
        if sources:
            with st.expander("📚 **Sources** (click to expand)"):
                for source in sources:
                    st.markdown(f"- {format_source_display(source)}")

    # Add assistant message to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })


# Check for pending question from demo button
if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None
    process_user_input(question)
    st.rerun()

# Chat input
user_input = st.chat_input("Ask me about sleep...")

if user_input:
    process_user_input(user_input)
    st.rerun()


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("### About")
    st.markdown("""
    **EmpowerSleep** is an educational chatbot that answers
    questions about sleep using content from the EmpowerSleep blog.

    ⚠️ **Not a medical tool** - This is for educational
    purposes only. Always consult healthcare professionals
    for medical advice.

    ---

    **How it works:**
    1. Your question is embedded using AI
    2. Similar content is retrieved from our blog
    3. An answer is generated from that content
    4. Sources are cited for transparency

    This RAG (Retrieval-Augmented Generation) approach
    ensures answers are grounded in real content.
    """)

    st.markdown("---")

    # Show index stats
    try:
        index = load_faiss_index()
        chunks = load_chunks_metadata()

        # Count unique articles
        unique_urls = set(c.get("url", "") for c in chunks)

        st.markdown("### Knowledge Base")
        st.markdown(f"- **Articles:** {len(unique_urls)}")
        st.markdown(f"- **Chunks:** {len(chunks)}")
        st.markdown(f"- **Vectors:** {index.ntotal}")
    except Exception:
        st.markdown("### Knowledge Base")
        st.markdown("*Not loaded*")

    st.markdown("---")

    # Reset conversation button
    if st.button("🔄 Start New Conversation"):
        reset_conversation()
        st.rerun()

    st.markdown("---")
    st.markdown("*Built with Streamlit + FAISS + OpenAI*")
