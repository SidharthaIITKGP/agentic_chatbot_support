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
import re
from typing import Literal, Annotated, TypedDict
from operator import add
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import create_react_agent, ToolNode
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import tools_condition
from .llm_tools import ALL_TOOLS

# Load environment variables
load_dotenv()

# Initialize checkpointer for persistent memory
# Create the database connection
import sqlite3
_conn = sqlite3.connect("src/logs/agent_memory.db", check_same_thread=False)
checkpointer = SqliteSaver(_conn)

# Define state for our agent with interrupts
class AgentState(TypedDict):
    """State that tracks conversation messages."""
    messages: Annotated[list[BaseMessage], add]

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

⚠️ CRITICAL RULES - FOLLOW STRICTLY:

1. REQUIRED INFORMATION:
   - get_order_status tool needs: order_id (e.g., "98762")
   - get_refund_status tool needs: order_id
   - check_product_availability tool needs: product_id (e.g., "P123")
   
2. IF CUSTOMER DOES NOT PROVIDE REQUIRED INFO:
   - DO NOT call the tool
   - DO NOT guess or make assumptions
   - POLITELY ask for the missing information
   - Example: "I can help you track your order. Could you please provide your order ID? (Example: 98762)"

3. FOR SENSITIVE ACTIONS (cancellations, refunds):
   - Always explain what will happen
   - Ask for explicit confirmation
   - Example: "I can help cancel order 98762. Please note this cannot be undone. Would you like me to proceed?"

4. CONVERSATION FLOW:
   Step 1: Identify what customer needs
   Step 2: Check if you have required information (order/product ID)
   Step 3a: If YES → Use appropriate tool
   Step 3b: If NO → Ask customer for it (STOP and wait for response)
   Step 4: Provide helpful answer with policy context

AVAILABLE TOOLS:
- get_order_status(order_id) - Requires order_id parameter
- get_refund_status(order_id) - Requires order_id parameter  
- check_product_availability(product_id) - Requires product_id parameter
- search_policy_documents(query) - No ID needed

Remember: Always confirm you have the required information BEFORE calling any tool!
"""

# Create the ReAct agent with interrupts
def create_llm_agent():
    """
    Create an LLM-powered ReAct agent using LangGraph with checkpointing and interrupts.
    
    This agent:
    - Uses LLM for reasoning (Thought)
    - Calls tools as needed (Action)
    - Observes tool results (Observation)
    - Interrupts before tool calls to check for missing info
    - Automatically saves conversation state via checkpointing
    """
    llm = get_llm()
    
    # Bind system message to LLM
    llm_with_system = llm.bind(system=SYSTEM_PROMPT)
    
    # Create ReAct agent with tools, checkpointing, and interrupts
    # interrupt_before=["tools"] means agent will pause before calling tools
    # This allows us to validate inputs and ask for missing information
    agent = create_react_agent(
        model=llm_with_system,
        tools=ALL_TOOLS,
        checkpointer=checkpointer,
        # Note: interrupt_before would require manual handling
        # For now, we rely on LLM + tool validation
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
    Each call is isolated (no conversation history).
    
    Args:
        user_query: The user's question
    
    Returns:
        The agent's response as a string
    """
    import uuid
    # Use unique thread_id for each query to avoid history accumulation
    result = llm_agent.invoke(
        {"messages": [{"role": "user", "content": user_query}]},
        config={"configurable": {"thread_id": f"simple_{uuid.uuid4()}"}}
    )
    
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
