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
# ---- Persistent-memory enabled runner ----
def run_agent_with_memory(user_query: str, session_id: str = "default_session") -> State:
    """
    Run the agent pipeline while loading and saving persistent memory for session_id.
    Memory fields used: last_order_id, last_product_id, last_intent, messages.
    """
    from .memory import load_memory, save_memory

    # Load persistent memory
    mem = load_memory(session_id)

    # initialize state (State requires user_query in constructor)
    # Seed history from memory so nodes can use previous turns/intents
    history = mem.get("messages", []) or []
    state = State(user_query=user_query, history=history)

    # seed slots from memory (if present)
    slots = state.slots or {}
    if mem.get("last_order_id"):
        slots.setdefault("order_id", mem.get("last_order_id"))
    if mem.get("last_product_id"):
        slots.setdefault("product_id", mem.get("last_product_id"))
    state.slots = slots

    # run nodes in existing pipeline
    state = classify_intent_node(state)

    # If intent is ambiguous (policy_query) but we have last_intent and matching slots, reuse it
    if state.intent == "policy_query" and mem.get("last_intent"):
        last_intent = mem.get("last_intent")
        # If we have an order_id slot and last intent was order-related, use it
        if state.slots.get("order_id") and last_intent in ("order_status", "refund_status", "delivery_delay"):
            state.intent = last_intent
        # If we have a product_id slot and last intent was product-related, use it
        elif state.slots.get("product_id") and last_intent in ("product_availability", "inventory_check"):
            state.intent = last_intent

    # Ask for missing slots (may set state.final_answer)
    state = ask_for_slot_node(state)
    # If we set a clarifying prompt, save it to memory and return early (same behavior as run_agent)
    if state.final_answer:
        # update memory messages with this assistant reply (so follow-ups can use it)
        mem_msgs = mem.get("messages", []) or []
        mem_msgs.append({"user": user_query, "assistant": state.final_answer})
        mem["messages"] = mem_msgs[-20:]
        # persist last known slots/intent as before
        if state.slots.get("order_id"):
            mem["last_order_id"] = state.slots.get("order_id")
        if state.slots.get("product_id"):
            mem["last_product_id"] = state.slots.get("product_id")
        if state.intent:
            mem["last_intent"] = state.intent
        save_memory(session_id, mem)
        return state

    # Continue pipeline
    state = call_tool_node(state)
    state = call_rag_node(state)
    state = compose_node(state)

    # After producing final_answer, update persistent memory
    if state.slots.get("order_id"):
        mem["last_order_id"] = state.slots.get("order_id")
    if state.slots.get("product_id"):
        mem["last_product_id"] = state.slots.get("product_id")
    if state.intent:
        mem["last_intent"] = state.intent
    # append messages (keep last 20)
    mem_msgs = mem.get("messages", []) or []
    mem_msgs.append({"user": user_query, "assistant": getattr(state, 'final_answer', '')})
    mem["messages"] = mem_msgs[-20:]
    save_memory(session_id, mem)

    return state

