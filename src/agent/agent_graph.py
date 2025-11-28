# src/agent/agent_graph.py
"""
ReAct-based LangGraph orchestrator for the customer support agent.

ReAct Pattern Flow:
  1. classify_intent_node: Initial classification
  2. reasoning_node: Think about what to do (Thought)
  3. action_node: Execute the action (Action)
  4. Loop back to reasoning_node or proceed to compose_node based on action
  5. compose_node: Generate final answer

The agent iterates through Thought → Action → Observation cycles until
it has enough information to answer, with a maximum iteration limit.
"""
from typing import Literal
from langgraph.graph import StateGraph, END
from .state import AgentState, State
from .nodes import (
    classify_intent_node,
    reasoning_node,
    action_node,
    compose_node,
)

# Maximum iterations to prevent infinite loops
MAX_ITERATIONS = 5

# Define routing logic
def should_continue_reasoning(state: AgentState) -> Literal["action", "end"]:
    """
    After reasoning: proceed to action if we haven't hit max iterations,
    otherwise force to end.
    """
    iteration = state.get("iteration", 0)
    if iteration > MAX_ITERATIONS:
        return "end"
    return "action"


def route_after_action(state: AgentState) -> Literal["reasoning", "compose", "end"]:
    """
    After action: determine next step based on what action was taken.
    - If action was "ask_for_slot": end (need user input)
    - If action was "finish": go to compose
    - Otherwise: loop back to reasoning for next iteration
    """
    action = state.get("action")
    iteration = state.get("iteration", 0)
    
    # Safety: prevent infinite loops
    if iteration >= MAX_ITERATIONS:
        return "compose"
    
    # If we asked for user input, stop here
    if action == "ask_for_slot":
        return "end"
    
    # If reasoning decided to finish, compose answer
    if action == "finish":
        return "compose"
    
    # Otherwise continue reasoning loop
    return "reasoning"


# Build the ReAct LangGraph
def create_agent_graph():
    """Create and compile the ReAct LangGraph workflow."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("reasoning", reasoning_node)
    workflow.add_node("action", action_node)
    workflow.add_node("compose", compose_node)
    
    # Define edges
    workflow.set_entry_point("classify_intent")
    workflow.add_edge("classify_intent", "reasoning")
    
    # Conditional edge after reasoning
    workflow.add_conditional_edges(
        "reasoning",
        should_continue_reasoning,
        {
            "action": "action",
            "end": END
        }
    )
    
    # Conditional edge after action - creates the ReAct loop
    workflow.add_conditional_edges(
        "action",
        route_after_action,
        {
            "reasoning": "reasoning",  # Loop back for next iteration
            "compose": "compose",
            "end": END
        }
    )
    
    workflow.add_edge("compose", END)
    
    return workflow.compile()

# Create the compiled graph (singleton)
agent_graph = create_agent_graph()

def run_agent(user_query: str, history=None) -> State:
    """
    Run the ReAct agent using LangGraph.
    Returns legacy State object for backward compatibility.
    """
    # Create initial state
    initial_state: AgentState = {
        "user_query": user_query,
        "intent": None,
        "slots": {},
        "thought": None,
        "action": None,
        "action_input": None,
        "observation": None,
        "tool_response": None,
        "rag_results": [],
        "final_answer": None,
        "history": history or [],
        "iteration": 0,
        "scratchpad": "",
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
    Run the ReAct agent pipeline with persistent memory using LangGraph.
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
    
    # Seed slots from memory - ONLY for digit-only follow-ups
    # Don't auto-fill for regular queries to avoid using stale context
    slots = {}
    query_stripped = user_query.strip()
    
    # Only seed from memory if user just typed digits (clear follow-up response)
    if query_stripped.isdigit():
        # This is a follow-up with just an order/product ID
        # Don't seed anything - let the classifier handle it fresh
        pass
    # For all other queries, start fresh without memory slots
    # This prevents "where is my order" from auto-using old order ID

    # Create initial state
    initial_state: AgentState = {
        "user_query": user_query,
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
    
    # Run the graph
    result = agent_graph.invoke(initial_state)
    
    # Apply memory-based intent correction if needed
    if result.get("intent") == "policy_query" and mem.get("last_intent"):
        last_intent = mem.get("last_intent")
        if result.get("slots", {}).get("order_id") and last_intent in ("order_status", "refund_status", "delivery_delay"):
            result["intent"] = last_intent
            # Re-run from classify_intent with corrected intent
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
    
    # run_agent is already defined in this file, no need to import
    q = " ".join(sys.argv[1:]) or input("User query: ")
    s = run_agent(q)
    print("=== FINAL ANSWER ===")
    print(s.final_answer)
    if s.rag_results:
        print("\n--- RAG snippets ---")
        for r in s.rag_results:
            print(r.get("metadata"), r.get("text")[:200])
