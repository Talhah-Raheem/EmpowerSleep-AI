"""
app.py - Streamlit UI for EmpowerSleep RAG Chatbot
===================================================

This is a THIN UI layer that delegates all logic to core/service.py.

Features:
- Chat interface with message history
- Demo question buttons for quick testing
- Sources expander showing retrieved documents
- Visual indicators for triage levels

Run with: streamlit run app.py
"""

import streamlit as st
from core.service import answer_question, check_api_key, ChatResponse
from config import DEMO_QUESTIONS


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

    /* Triage alert styling */
    .crisis-alert {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 15px;
        margin: 10px 0;
    }

    .urgent-alert {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize session state variables for chat history."""
    if "messages" not in st.session_state:
        # Chat history: list of {"role": "user"|"assistant", "content": str, "sources": list}
        st.session_state.messages = []

    if "pending_question" not in st.session_state:
        # Used to pass demo button clicks to the chat input
        st.session_state.pending_question = None


init_session_state()


# =============================================================================
# HEADER
# =============================================================================

st.title("🌙 EmpowerSleep")
st.markdown("*Your Sleep Education Assistant*")

# API key check
if not check_api_key():
    st.warning(
        "⚠️ OpenAI API key not configured. "
        "Please set `OPENAI_API_KEY` in your `.env` file or environment."
    )

st.markdown("---")


# =============================================================================
# DEMO QUESTION BUTTONS
# =============================================================================

st.markdown("**Try a demo question:**")

# Create columns for demo buttons
cols = st.columns(len(DEMO_QUESTIONS))

for i, question in enumerate(DEMO_QUESTIONS):
    with cols[i]:
        # Truncate long questions for button display
        display_text = question[:40] + "..." if len(question) > 40 else question
        if st.button(display_text, key=f"demo_{i}", help=question):
            st.session_state.pending_question = question

st.markdown("---")


# =============================================================================
# CHAT HISTORY DISPLAY
# =============================================================================

def display_message(message: dict):
    """Display a single chat message with appropriate styling."""
    role = message["role"]
    content = message["content"]
    sources = message.get("sources", [])
    triage_level = message.get("triage_level")

    with st.chat_message(role):
        # Show triage indicator if applicable
        if triage_level == "CRISIS":
            st.error("🚨 Crisis Support Resources")
        elif triage_level == "URGENT":
            st.warning("⚠️ Medical Attention Recommended")

        # Show the message content
        st.markdown(content)

        # Show sources in expander (only for assistant messages with sources)
        if role == "assistant" and sources:
            with st.expander("📚 Sources used"):
                for source in sources:
                    st.markdown(f"- `{source}`")


# Display all messages in history
for message in st.session_state.messages:
    display_message(message)


# =============================================================================
# CHAT INPUT HANDLING
# =============================================================================

def process_user_input(user_input: str):
    """Process user input and generate response."""
    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
    })

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response: ChatResponse = answer_question(user_input)

        # Show triage indicator if applicable
        if response.triage_level == "CRISIS":
            st.error("🚨 Crisis Support Resources")
        elif response.triage_level == "URGENT":
            st.warning("⚠️ Medical Attention Recommended")

        # Show the response
        st.markdown(response.answer)

        # Show sources
        if response.sources:
            with st.expander("📚 Sources used"):
                for source in response.sources:
                    st.markdown(f"- `{source}`")

    # Add assistant message to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": response.answer,
        "sources": response.sources,
        "triage_level": response.triage_level,
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
    questions about sleep using a curated knowledge base.

    ⚠️ **Not a medical tool** - This is for educational
    purposes only. Always consult healthcare professionals
    for medical advice.

    ---

    **How it works:**
    1. Your question is checked for safety concerns
    2. Relevant content is retrieved from the knowledge base
    3. An AI generates a grounded response
    4. Sources are shown for transparency
    """)

    st.markdown("---")

    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("*Built with Streamlit + FAISS + OpenAI*")
