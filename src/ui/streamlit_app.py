# src/ui/streamlit_app.py
import streamlit as st
import sys
from pathlib import Path
import os

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.agent_graph import run_agent_with_memory
from src.agent.memory import load_memory, save_memory, memory_path

# Page config
st.set_page_config(
    page_title="Customer Support Chatbot",
    page_icon="💬",
    layout="centered",
)

st.title("💬 Customer Support Chatbot")
st.markdown("Ask me about orders, refunds, returns, policies, or product availability!")

# Initialize session ID for memory
if "session_id" not in st.session_state:
    st.session_state.session_id = "streamlit_user"

# If messages haven't been initialized, try to load from persistent memory
if "messages" not in st.session_state:
    # load persistent memory and convert to UI-friendly chat format
    mem = load_memory(st.session_state.session_id)
    ui_messages = []
    for entry in mem.get("messages", []):
        # memory entries were stored as {"user": "...", "assistant": "..."}
        if isinstance(entry, dict) and "user" in entry and "assistant" in entry:
            ui_messages.append({"role": "user", "content": entry["user"]})
            ui_messages.append({"role": "assistant", "content": entry["assistant"]})
        else:
            # fallback: store as a plain user entry if unexpected shape
            ui_messages.append({"role": "user", "content": str(entry)})
    st.session_state.messages = ui_messages

# Sidebar controls
with st.sidebar:
    st.title("Settings")
    clear_history = st.button("Clear Chat History (session only)")
    clear_all = st.button("Clear Chat + Memory (persistent)")

# Display chat messages (use st.chat_message if available; fallback to markdown)
for message in st.session_state.messages:
    role = message.get("role", "user")
    content = message.get("content", "")
    # prefer st.chat_message when available
    try:
        with st.chat_message(role):
            st.markdown(content)
    except Exception:
        # fallback rendering
        if role == "user":
            st.markdown(f"**You:** {content}")
        else:
            st.markdown(f"**Bot:** {content}")

# Handle clear actions
if clear_history:
    st.session_state.messages = []
    # attempt to rerun to reflect immediately
    try:
        st.experimental_rerun()
    except Exception:
        try:
            st.rerun()
        except Exception:
            pass

if clear_all:
    # clear in-memory UI state
    st.session_state.messages = []
    # clear persistent memory file
    mp = memory_path(st.session_state.session_id)
    try:
        if os.path.exists(mp):
            os.remove(mp)
    except Exception as e:
        st.warning(f"Failed to remove memory file: {e}")
    # attempt to rerun
    try:
        st.experimental_rerun()
    except Exception:
        try:
            st.rerun()
        except Exception:
            pass

# Main chat input
prompt = ""
try:
    prompt = st.chat_input("How can I help you today?")
except Exception:
    # fallback if st.chat_input isn't available
    prompt = st.text_input("How can I help you today?", key="fallback_input")

if prompt:
    # normalize and append user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # show the user message immediately using chat_message if available
    try:
        with st.chat_message("user"):
            st.markdown(prompt)
    except Exception:
        st.markdown(f"**You:** {prompt}")

    # Call agent (wrapped in try/except)
    try:
        with st.spinner("Thinking..."):
            state = run_agent_with_memory(prompt, session_id=st.session_state.session_id)
            response = state.final_answer or "Sorry — I couldn't produce an answer."
    except Exception as e:
        response = f"Agent error: {e}"

    # Display assistant response
    try:
        with st.chat_message("assistant"):
            st.markdown(response)
    except Exception:
        st.markdown(f"**Bot:** {response}")

    # Append assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Optionally persist UI messages into memory's message list as well (keeps UI and memory consistent)
    try:
        # load existing memory, append latest pair, save
        mem = load_memory(st.session_state.session_id)
        mem_msgs = mem.get("messages", [])
        mem_msgs.append({"user": prompt, "assistant": response})
        mem["messages"] = mem_msgs[-20:]
        save_memory(st.session_state.session_id, mem)
    except Exception:
        # not critical — continue
        pass
