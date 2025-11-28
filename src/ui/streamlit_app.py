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

# Show info banner about clearing memory if needed
if "error_shown" not in st.session_state:
    st.session_state.error_shown = False

# Initialize session ID for memory
if "session_id" not in st.session_state:
    st.session_state.session_id = "streamlit_user"

# If messages haven't been initialized, try to load from persistent memory
if "messages" not in st.session_state:
    # load persistent memory and convert to UI-friendly chat format
    try:
        mem = load_memory(st.session_state.session_id)
        ui_messages = []
        for entry in mem.get("messages", []):
            # memory entries were stored as {"user": "...", "assistant": "..."}
            if isinstance(entry, dict) and "user" in entry and "assistant" in entry:
                ui_messages.append({"role": "user", "content": entry["user"]})
                ui_messages.append({"role": "assistant", "content": entry["assistant"]})
            # Skip any malformed entries
        st.session_state.messages = ui_messages
    except Exception:
        # If memory loading fails, start fresh
        st.session_state.messages = []

# Sidebar controls
with st.sidebar:
    st.title("Settings")
    
    # Toggle to show reasoning trace
    show_reasoning = st.checkbox("Show ReAct Reasoning", value=False, help="Display the agent's thought process")
    
    st.divider()
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
            # Import agent_graph to get full state with scratchpad
            from src.agent.agent_graph import agent_graph
            from src.agent.memory import load_memory, save_memory
            
            # Load memory
            mem = load_memory(st.session_state.session_id)
            history_raw = mem.get("messages", []) or []
            history = [entry for entry in history_raw if isinstance(entry, dict)]
            
            slots = {}
            if mem.get("last_order_id"):
                slots["order_id"] = mem.get("last_order_id")
            if mem.get("last_product_id"):
                slots["product_id"] = mem.get("last_product_id")
            
            # Create initial state
            initial_state = {
                "user_query": prompt,
                "intent": None,
                "slots": slots,
                "thought": None,
                "action": None,
                "action_input": None,
                "observation": None,
                "tool_response": None,
                "rag_results": [],
                "final_answer": None,
                "history": history,
                "iteration": 0,
                "scratchpad": "",
                "errors": []
            }
            
            # Run graph and get full result
            result = agent_graph.invoke(initial_state)
            
            response = result.get("final_answer") or "Sorry — I couldn't produce an answer."
            scratchpad = result.get("scratchpad", "")
            
            # Update memory
            if result.get("slots", {}).get("order_id"):
                mem["last_order_id"] = result["slots"]["order_id"]
            if result.get("slots", {}).get("product_id"):
                mem["last_product_id"] = result["slots"]["product_id"]
            if result.get("intent"):
                mem["last_intent"] = result["intent"]
            mem_msgs = mem.get("messages", []) or []
            mem_msgs.append({"user": prompt, "assistant": response})
            mem["messages"] = mem_msgs[-20:]
            save_memory(st.session_state.session_id, mem)
            
    except AttributeError as e:
        # Specific handling for State object issues - likely stale memory
        if "'State' object has no attribute 'get'" in str(e):
            response = "⚠️ **Memory format error detected.** This usually happens after a code update. Please click '**Clear Chat + Memory (persistent)**' in the sidebar to reset and try again."
            st.session_state.error_shown = True
            scratchpad = ""
        else:
            response = f"Agent error: {e}"
            scratchpad = ""
    except Exception as e:
        response = f"Agent error: {e}"
        scratchpad = ""

    # Display assistant response
    try:
        with st.chat_message("assistant"):
            st.markdown(response)
            
            # Show reasoning trace if enabled
            if show_reasoning and scratchpad:
                with st.expander("🧠 ReAct Reasoning Trace"):
                    st.text(scratchpad)
    except Exception:
        st.markdown(f"**Bot:** {response}")
        if show_reasoning and scratchpad:
            st.text_area("Reasoning:", scratchpad, height=200)

    # Append assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
