# src/agent/agent_graph.py
"""
Simple orchestrator that runs the nodes in a deterministic order.
This is not LangGraph — just a clear reference implementation of the flow:
  1. classify_intent_node
  2. ask_for_slot_node (may short-circuit)
  3. call_tool_node (if applicable)
  4. call_rag_node
  5. compose_node
"""
from .state import State
from .nodes import (
    classify_intent_node,
    ask_for_slot_node,
    call_tool_node,
    call_rag_node,
    compose_node,
)

def run_agent(user_query: str, history=None) -> State:
    state = State(user_query=user_query, history=history or [])
    # 1
    state = classify_intent_node(state)

    # 2
    state = ask_for_slot_node(state)
    if state.final_answer:
        return state

    # 3
    state = call_tool_node(state)

    # 4
    state = call_rag_node(state)

    # 5
    state = compose_node(state)

    return state

# Quick CLI
if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or input("User query: ")
    s = run_agent(q)
    print("=== FINAL ANSWER ===")
    print(s.final_answer)
    if s.rag_results:
        print("\n--- RAG snippets ---")
        for r in s.rag_results:
            print(r.get("metadata"), r.get("text")[:200])
