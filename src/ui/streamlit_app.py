# src/ui/streamlit_app.py
import streamlit as st
import sys
from pathlib import Path
import os

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.llm_agent import run_llm_agent_with_memory

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
            
            # Extract response from messages
            messages = result.get("messages", [])
            if messages:
                # Get the last message (should be assistant's response)
                last_message = messages[-1]
                response = last_message.content if hasattr(last_message, "content") else str(last_message)
            else:
                response = "Sorry — I couldn't produce an answer."
            
            # Extract reasoning trace if available
            scratchpad = ""
            if show_reasoning and messages:
                # Build reasoning trace from message history
                trace_parts = []
                for msg in messages[1:]:  # Skip the first user message
                    if hasattr(msg, "content"):
                        content = msg.content
                        # Check if it's a tool message or reasoning
                        if hasattr(msg, "type"):
                            msg_type = msg.type
                            if msg_type == "ai" and any(kw in content.lower() for kw in ["thought:", "action:", "i will", "let me"]):
                                trace_parts.append(f"Thought: {content}")
                            elif msg_type == "tool":
                                trace_parts.append(f"Observation: {content}")
                scratchpad = "\n\n".join(trace_parts) if trace_parts else "No reasoning trace available"
            
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
