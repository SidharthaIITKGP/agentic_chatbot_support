# src/agent/llm_agent.py
"""
LLM-Powered ReAct Agent using LangGraph and Groq.

This replaces the manual rule-based reasoning with TRUE ReAct:
- LLM generates thoughts (reasoning)
- LLM decides which actions to take (tool calls)
- LLM observes results
- LLM decides when it has enough information
- LLM generates natural language responses
"""
import os
from typing import Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from .llm_tools import ALL_TOOLS

# Load environment variables
load_dotenv()

# Initialize checkpointer for persistent memory
# Create the database connection
import sqlite3
_conn = sqlite3.connect("src/logs/agent_memory.db", check_same_thread=False)
checkpointer = SqliteSaver(_conn)

# Initialize Groq LLM
def get_llm():
    """Get the Groq LLM instance."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    
    return ChatGroq(
        api_key=api_key,
        model="llama-3.1-8b-instant",  # Fast and efficient model
        temperature=0.1,  # Low temperature for consistent behavior
        max_tokens=8000,
    )

# System prompt that guides the LLM's behavior
SYSTEM_PROMPT = """You are a helpful customer support agent for an e-commerce company.

Your role is to assist customers with:
- Order tracking and status
- Refund inquiries
- Product availability
- Company policies (returns, cancellations, delivery, etc.)

IMPORTANT GUIDELINES:
1. Always be polite, professional, and empathetic
2. If you need information (like an order ID), ask the customer
3. Use the available tools to get accurate information
4. When you have all the information needed, provide a clear, helpful answer
5. Reference relevant policies when applicable
6. Keep responses concise but complete

AVAILABLE TOOLS:
- get_order_status: Check order status and tracking
- get_refund_status: Check refund processing status
- check_product_availability: Check if products are in stock
- search_policy_documents: Search company policies

WORKFLOW:
1. Understand what the customer needs
2. If you need more details (like order ID), ask
3. Use tools to gather information
4. Provide a helpful, complete answer

Remember: You're representing the company, so be helpful and professional!
"""

# Create the ReAct agent
def create_llm_agent():
    """
    Create an LLM-powered ReAct agent using LangGraph with checkpointing.
    
    This agent:
    - Uses LLM for reasoning (Thought)
    - Calls tools as needed (Action)
    - Observes tool results (Observation)
    - Decides when to finish (iterates until confident)
    - Automatically saves conversation state via checkpointing
    """
    llm = get_llm()
    
    # Bind system message to LLM
    llm_with_system = llm.bind(system=SYSTEM_PROMPT)
    
    # Create ReAct agent with tools and checkpointing
    # LangGraph handles the ReAct loop and state persistence automatically!
    agent = create_agent(
        model=llm_with_system,
        tools=ALL_TOOLS,
        checkpointer=checkpointer,  # Enable automatic state persistence
    )
    
    return agent

# Singleton instance
llm_agent = create_llm_agent()

def run_llm_agent(user_query: str, conversation_history: list = None) -> dict:
    """
    Run the LLM-powered ReAct agent.
    
    Args:
        user_query: The user's question
        conversation_history: Previous messages (optional)
    
    Returns:
        Dictionary with 'messages' list containing the conversation
    """
    # Prepare messages
    messages = conversation_history or []
    messages.append({"role": "user", "content": user_query})
    
    # Run the agent (LangGraph handles the ReAct loop)
    # Use a temporary thread_id for stateless queries
    result = llm_agent.invoke(
        {"messages": messages},
        config={"configurable": {"thread_id": "temp"}}
    )
    
    return result

def run_llm_agent_simple(user_query: str) -> str:
    """
    Simple interface: query in, answer out.
    
    Args:
        user_query: The user's question
    
    Returns:
        The agent's response as a string
    """
    result = run_llm_agent(user_query)
    
    # Extract final message
    messages = result.get("messages", [])
    if messages:
        last_message = messages[-1]
        if hasattr(last_message, 'content'):
            return last_message.content
        elif isinstance(last_message, dict):
            return last_message.get("content", "No response generated")
    
    return "No response generated"

def run_llm_agent_with_memory(user_query: str, session_id: str = "default") -> dict:
    """
    Run agent with persistent conversation memory using LangGraph checkpointing.
    
    Args:
        user_query: The user's question
        session_id: Session identifier for memory persistence (thread_id)
    
    Returns:
        Dictionary with 'answer' (string) and 'messages' (list)
    """
    # Run agent with thread_id for automatic state persistence
    result = llm_agent.invoke(
        {"messages": [{"role": "user", "content": user_query}]},
        config={"configurable": {"thread_id": session_id}}
    )
    
    # Extract answer
    result_messages = result.get("messages", [])
    answer = ""
    if result_messages:
        last_message = result_messages[-1]
        if hasattr(last_message, 'content'):
            answer = last_message.content
        elif isinstance(last_message, dict):
            answer = last_message.get("content", "No response")
    
    return {
        "answer": answer,
        "messages": result_messages
    }

# CLI interface
if __name__ == "__main__":
    import sys
    
    query = " ".join(sys.argv[1:]) or input("User query: ")
    
    print("\n" + "="*70)
    print("LLM-POWERED REACT AGENT")
    print("="*70)
    print(f"\nQuery: {query}\n")
    
    answer = run_llm_agent_simple(query)
    
    print("="*70)
    print("RESPONSE:")
    print("="*70)
    print(answer)
    print()
