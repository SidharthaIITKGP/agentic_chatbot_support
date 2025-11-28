# src/agent/agent_graph.py
"""
LangGraph-based orchestrator for the customer support agent.
Flow:
  1. classify_intent_node
  2. ask_for_slot_node (may short-circuit to END)
  3. call_tool_node (if applicable)
  4. call_rag_node
  5. compose_node
"""
from typing import Literal
from langgraph.graph import StateGraph, END
from .state import AgentState, State
from .nodes import (
    classify_intent_node,
    ask_for_slot_node,
    call_tool_node,
    call_rag_node,
    compose_node,
)

# Define routing logic
def should_continue_after_slot_check(state: AgentState) -> Literal["continue", "end"]:
    """Route to END if we have a final_answer (clarification needed), otherwise continue."""
    if state.get("final_answer"):
        return "end"
    return "continue"

# Build the LangGraph
def create_agent_graph():
    """Create and compile the LangGraph workflow."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("ask_for_slot", ask_for_slot_node)
    workflow.add_node("call_tool", call_tool_node)
    workflow.add_node("call_rag", call_rag_node)
    workflow.add_node("compose", compose_node)
    
    # Define edges
    workflow.set_entry_point("classify_intent")
    workflow.add_edge("classify_intent", "ask_for_slot")
    
    # Conditional edge after slot check
    workflow.add_conditional_edges(
        "ask_for_slot",
        should_continue_after_slot_check,
        {
            "continue": "call_tool",
            "end": END
        }
    )
    
    workflow.add_edge("call_tool", "call_rag")
    workflow.add_edge("call_rag", "compose")
    workflow.add_edge("compose", END)
    
    return workflow.compile()

# Create the compiled graph (singleton)
agent_graph = create_agent_graph()

def run_agent(user_query: str, history=None) -> State:
    """
    Run the agent using LangGraph.
    Returns legacy State object for backward compatibility.
    """
    # Create initial state
    initial_state: AgentState = {
        "user_query": user_query,
        "intent": None,
        "slots": {},
        "tool_response": None,
        "rag_results": [],
        "final_answer": None,
        "history": history or [],
        "errors": []
    }
    
    # Run the graph
    result = agent_graph.invoke(initial_state)
    
    # Convert back to legacy State for compatibility
    return State(
        user_query=result.get("user_query", user_query),
        intent=result.get("intent"),
        slots=result.get("slots", {}),
        tool_response=result.get("tool_response"),
        rag_results=result.get("rag_results", []),
        final_answer=result.get("final_answer"),
        history=result.get("history", []),
        errors=result.get("errors", [])
    )

# ---- Persistent-memory enabled runner ----
def run_agent_with_memory(user_query: str, session_id: str = "default_session") -> State:
    """
    Run the agent pipeline with persistent memory using LangGraph.
    Memory fields used: last_order_id, last_product_id, last_intent, messages.
    """
    from .memory import load_memory, save_memory

    # Load persistent memory
    mem = load_memory(session_id)

    # Seed history from memory (ensure it's a list of dicts, not State objects)
    history_raw = mem.get("messages", []) or []
    history = []
    for entry in history_raw:
        # Convert any State objects or weird formats to simple dicts
        if isinstance(entry, dict):
            history.append(entry)
        else:
            # Skip non-dict entries (might be old State objects)
            continue
    
    # Seed slots from memory
    slots = {}
    if mem.get("last_order_id"):
        slots["order_id"] = mem.get("last_order_id")
    if mem.get("last_product_id"):
        slots["product_id"] = mem.get("last_product_id")

    # Create initial state
    initial_state: AgentState = {
        "user_query": user_query,
        "intent": None,
        "slots": slots,
        "tool_response": None,
        "rag_results": [],
        "final_answer": None,
        "history": history,
        "errors": []
    }
    
    # Run the graph
    result = agent_graph.invoke(initial_state)
    
    # Apply memory-based intent correction if needed
    if result.get("intent") == "policy_query" and mem.get("last_intent"):
        last_intent = mem.get("last_intent")
        if result.get("slots", {}).get("order_id") and last_intent in ("order_status", "refund_status", "delivery_delay"):
            result["intent"] = last_intent
            # Re-run from call_tool with corrected intent
            result = agent_graph.invoke(result)
        elif result.get("slots", {}).get("product_id") and last_intent in ("product_availability", "inventory_check"):
            result["intent"] = last_intent
            result = agent_graph.invoke(result)

    # Update persistent memory
    if result.get("slots", {}).get("order_id"):
        mem["last_order_id"] = result["slots"]["order_id"]
    if result.get("slots", {}).get("product_id"):
        mem["last_product_id"] = result["slots"]["product_id"]
    if result.get("intent"):
        mem["last_intent"] = result["intent"]
    
    # Append messages (keep last 20)
    mem_msgs = mem.get("messages", []) or []
    mem_msgs.append({"user": user_query, "assistant": result.get("final_answer", "")})
    mem["messages"] = mem_msgs[-20:]
    save_memory(session_id, mem)

    # Convert back to legacy State
    return State(
        user_query=result.get("user_query", user_query),
        intent=result.get("intent"),
        slots=result.get("slots", {}),
        tool_response=result.get("tool_response"),
        rag_results=result.get("rag_results", []),
        final_answer=result.get("final_answer"),
        history=result.get("history", []),
        errors=result.get("errors", [])
    )

# Quick CLI
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add project root to path for direct execution
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    
    from src.agent.agent_graph import run_agent
    
    q = " ".join(sys.argv[1:]) or input("User query: ")
    s = run_agent(q)
    print("=== FINAL ANSWER ===")
    print(s.final_answer)
    if s.rag_results:
        print("\n--- RAG snippets ---")
        for r in s.rag_results:
            print(r.get("metadata"), r.get("text")[:200])
