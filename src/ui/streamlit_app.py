import streamlit as st
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.agent_graph import run_agent

# Page config
st.set_page_config(
    page_title="Customer Support Chatbot",
    page_icon="💬",
    layout="centered"
)

# Title
st.title("💬 Customer Support Chatbot")
st.markdown("Ask me about orders, refunds, returns, policies, or product availability!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("How can I help you today?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Run the agent
            state = run_agent(prompt, history=st.session_state.messages)
            response = state.final_answer
            
            # Display response
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Add a clear chat button in sidebar
with st.sidebar:
    st.title("Settings")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
