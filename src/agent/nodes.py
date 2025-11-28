# src/agent/nodes.py
"""
Nodes used in a simple agent graph.
Each node is a callable that receives a State and returns the updated State.
"""

from typing import Tuple
from .state import State
from .intent_classifier import classify_intent
from .prompts import CLARIFY_FOR_ORDER_ID, CLARIFY_FOR_PRODUCT_ID
from src.tools.tools import get_order_status, get_refund_status, get_inventory
from src.rag.retriever import retrieve_policy

def classify_intent_node(state: State) -> State:
    res = classify_intent(state.user_query)
    state.intent = res.get("intent")
    state.slots.update(res.get("slots", {}))
    return state

def ask_for_slot_node(state: State) -> State:
    # If intent requires order_id but missing, set final_answer to clarifying prompt
    if state.intent in ("order_status", "refund_status", "delivery_delay") and "order_id" not in state.slots:
        state.final_answer = CLARIFY_FOR_ORDER_ID
        return state
    if state.intent == "product_availability" and "product_id" not in state.slots:
        state.final_answer = CLARIFY_FOR_PRODUCT_ID
        return state
    return state

def call_tool_node(state: State) -> State:
    try:
        if state.intent == "order_status":
            oid = state.slots.get("order_id")
            state.tool_response = get_order_status(oid)
        elif state.intent == "refund_status":
            oid = state.slots.get("order_id")
            state.tool_response = get_refund_status(oid)
        elif state.intent == "product_availability":
            pid = state.slots.get("product_id")
            state.tool_response = get_inventory(pid)
        # else: leave tool_response None
    except Exception as e:
        state.errors.append(str(e))
    return state

def call_rag_node(state: State) -> State:
    """
    For intents that want a policy lookup or when tool response is missing,
    call the retriever and store the top snippets in state.rag_results.
    """
    # Decide when to call RAG
    if state.intent in ("charges_query", "return_policy", "delivery_delay", "policy_query"):
        out = retrieve_policy(state.user_query, fetch_k=10, top_k=3, alpha=0.85)
        state.rag_results = out.get("final", [])
    else:
        # also call RAG when we got a tool response but want policy backing
        if state.tool_response is not None:
            # craft a short policy-focused query derived from intent
            query = state.user_query
            out = retrieve_policy(query, fetch_k=6, top_k=2, alpha=0.85)
            state.rag_results = out.get("final", [])
    return state

def compose_node(state: State) -> State:
    """
    Call composer to produce final_answer from tool_response + rag_results
    """
    from .composer import compose_final_answer
    try:
        state.final_answer = compose_final_answer(state)
    except Exception as e:
        state.errors.append(str(e))
        state.final_answer = "Sorry — something went wrong while composing the answer."
    return state
