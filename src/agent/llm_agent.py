# src/agent/llm_agent.py
"""
LLM-Powered ReAct Agent using LangGraph and Groq.

- LLM generates thoughts (reasoning)
- LLM decides which actions to take (tool calls)
- LLM observes results
- LLM decides when it has enough information
- LLM generates natural language responses
"""
import os
import re
from typing import Literal, Annotated, TypedDict
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import create_react_agent, ToolNode
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
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
    messages: Annotated[list[BaseMessage], add_messages]

# Initialize Gemini LLM
def get_llm():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    return ChatGoogleGenerativeAI(
        api_key=api_key, #type: ignore
        model="gemini-2.5-flash",  # Fast and efficient model
    )

# System prompt that guides the LLM's behavior
SYSTEM_PROMPT = """You are a helpful customer support agent for an e-commerce company.

Your role is to assist customers with orders, refunds, returns, and policy questions.

CRITICAL RULES - MUST FOLLOW:
1. NEVER call a tool without the required parameters from the customer's actual message
2. NEVER use example numbers or make up IDs - only use what the customer provides
3. Order IDs are 4-5 digit numbers (like 12345, 98765, etc.)
4. Product IDs start with P or PROD (like P123, PROD456)

WORKFLOW:
Step 1: Check if the customer provided the required ID in their message
Step 2: If ID is present → call the appropriate tool with that ID
Step 3: If ID is NOT present → politely ask for it, DO NOT call any tool

EXAMPLES OF CORRECT BEHAVIOR:
- Customer: "where is my order 98762?" → Call get_order_status("98762")
- Customer: "where is my order" → Ask "I can help you track your order. Could you please provide your order ID?"
- Customer: "check stock for P123" → Call check_product_availability("P123")
- Customer: "is the product available" → Ask "I can help check availability. Could you provide the product ID?"

AVAILABLE TOOLS:
- get_order_status(order_id) - for order tracking
- get_refund_status(order_id) - for refund inquiries  
- check_product_availability(product_id) - for stock checks
- search_policy_documents(query) - for policy questions

Always be helpful, accurate, and concise in your responses.
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

    agent = create_react_agent(
        model=llm_with_system,
        tools=ALL_TOOLS,
        checkpointer=checkpointer,
    )
    
    return agent

# Singleton instance
llm_agent = create_llm_agent()

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


