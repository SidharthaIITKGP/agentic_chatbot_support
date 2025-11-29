# src/ui/streamlit_app.py
import streamlit as st
import sys
from pathlib import Path
import os

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.llm_agent import run_llm_agent_with_memory, checkpointer

# Function to clear checkpoint history
def clear_checkpoint_history(session_id: str):
    """Clear the checkpoint history for a session."""
    try:
        # Get all checkpoints for this thread
        config = {"configurable": {"thread_id": session_id}}
        # Clear by creating a new empty state
        # Note: LangGraph doesn't have a direct clear method, so we'll just start fresh
        return True
    except Exception as e:
        st.error(f"Error clearing history: {e}")
        return False

# Page config
st.set_page_config(
    page_title="Customer Support Chatbot - LLM Powered",
    page_icon="💬",
    layout="centered",
)

st.title("💬 Customer Support Chatbot")
st.markdown("**LLM-Powered ReAct Agent** - Ask me about orders, refunds, returns, policies, or product availability!")

# Initialize session ID for memory
if "session_id" not in st.session_state:
    st.session_state.session_id = "streamlit_user"

# Initialize messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar controls
with st.sidebar:
    st.title("Settings")
    
    # Toggle to show reasoning trace
    show_reasoning = st.checkbox("Show ReAct Reasoning", value=False, help="Display the agent's thought process")
    
    st.divider()
    clear_history = st.button("Clear Chat History")

# Display chat messages
for message in st.session_state.messages:
    role = message.get("role", "user")
    content = message.get("content", "")
    with st.chat_message(role):
        st.markdown(content)

# Handle clear actions
if clear_history:
    st.session_state.messages = []
    # Note: With checkpointing, old conversation state is persisted
    # To truly start fresh, user should use a new session or we'd need to clear the DB
    # For now, just clear the UI display
    st.success("Chat history cleared! (Note: Conversation context persists in memory)")
    st.rerun()

# Main chat input
prompt = st.chat_input("How can I help you today?")

if prompt:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Show the user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call LLM agent
    try:
        with st.spinner("Thinking..."):
            result = run_llm_agent_with_memory(prompt, session_id=st.session_state.session_id)
            
            # Extract response (checkpointing returns answer and messages)
            response = result.get("answer", "Sorry — I couldn't produce an answer.")
            messages = result.get("messages", [])
            
            # Extract reasoning trace if available
            scratchpad = ""
            if show_reasoning and messages:
                # Build reasoning trace from message history
                trace_parts = []
                for msg in messages:
                    if hasattr(msg, "content"):
                        content = str(msg.content)
                        # Check message type for reasoning
                        msg_name = getattr(msg, "name", "")
                        if msg_name:  # Tool calls
                            trace_parts.append(f"🔧 Tool: {msg_name}\n{content[:200]}...")
                        elif "thought" in content.lower() or "let me" in content.lower():
                            trace_parts.append(f"💭 Reasoning: {content[:200]}...")
                scratchpad = "\n\n".join(trace_parts) if trace_parts else "No detailed reasoning trace available"
            
    except Exception as e:
        response = f"Agent error: {e}"
        scratchpad = ""

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response)
        
        # Show reasoning trace if enabled
        if show_reasoning and scratchpad:
            with st.expander("🧠 ReAct Reasoning Trace"):
                st.text(scratchpad)

    # Append assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
