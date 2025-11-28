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


def _normalize_rag_output(out):
    """
    Normalize retriever output to a list of docs.
    Accept either a dict with "final" or a list of docs.
    """
    if out is None:
        return []
    if isinstance(out, dict):
        return out.get("final", []) or out.get("results", []) or []
    if isinstance(out, list):
        return out
    return []


def classify_intent_node(state: State) -> State:
    """
    Determine intent and extract simple slots. Also handle the common follow-up case:
    when the user replies with only an order id (digits), treat it as slot-filling and
    inherit previous intent (if available) instead of mis-classifying.
    """
    raw = (state.user_query or "").strip()

    # --- 1) Quick follow-up: user replied with only digits -> treat as order_id slot ---
    if raw.isdigit():
        # assign the order_id slot
        state.slots["order_id"] = raw

        # if we don't have a strong intent yet, try to inherit the previous one from history
        if not state.intent and getattr(state, "history", None):
            prev = state.history[-1] if state.history else {}
            prev_intent = prev.get("intent")
            if prev_intent in ("order_status", "refund_status", "delivery_delay", None):
                state.intent = "order_status"
                return state

        # if the current intent looks harmless, prefer it to be order_status for digits
        if state.intent in (None, "", "policy_query"):
            state.intent = "order_status"
            return state

    # --- 2) Otherwise use normal classifier ---
    res = classify_intent(state.user_query)
    state.intent = res.get("intent")
    # update any extracted slots from classifier (merge with existing)
    slots = res.get("slots", {}) or {}
    state.slots.update(slots)
    return state


def ask_for_slot_node(state: State) -> State:
    """
    If intent requires a particular slot and it is missing, ask for it (set final_answer).
    Also handle the case where the user provided a numeric order_id in a previous turn:
    if order_id exists now, do not ask for it.
    """
    # Order-related intents that need order_id
    if state.intent in ("order_status", "refund_status", "delivery_delay"):
        if not state.slots.get("order_id"):
            state.final_answer = CLARIFY_FOR_ORDER_ID
            return state

    # Product availability intent requires product_id
    if state.intent in ("product_availability", "inventory_check"):
        if not state.slots.get("product_id"):
            state.final_answer = CLARIFY_FOR_PRODUCT_ID
            return state

    return state


def call_tool_node(state: State) -> State:
    """
    Call the appropriate mock tool based on intent & slots. Do not crash if slot is missing;
    instead record an error or leave tool_response as None.
    """
    try:
        if state.intent == "order_status":
            oid = state.slots.get("order_id")
            if oid:
                state.tool_response = get_order_status(oid)
            else:
                state.tool_response = {"error": "missing_order_id"}
        elif state.intent == "refund_status":
            oid = state.slots.get("order_id")
            if oid:
                state.tool_response = get_refund_status(oid)
            else:
                state.tool_response = {"error": "missing_order_id"}
        elif state.intent in ("product_availability", "inventory_check"):
            pid = state.slots.get("product_id")
            if pid:
                state.tool_response = get_inventory(pid)
            else:
                state.tool_response = {"error": "missing_product_id"}
        else:
            # no tool needed
            state.tool_response = None
    except Exception as e:
        state.errors.append(str(e))
        state.tool_response = None
    return state


def call_rag_node(state: State) -> State:
    """
    For intents that want a policy lookup or when tool response exists but policy backing is desired,
    call the retriever and store a normalized list into state.rag_results.
    """
    try:
        # Intents that primarily want policy text
        if state.intent in ("charges_query", "return_policy", "delivery_delay", "policy_query"):
            out = retrieve_policy(state.user_query, fetch_k=10, top_k=3, alpha=0.85)
            state.rag_results = _normalize_rag_output(out)

        else:
            # Also call RAG when we got a tool response but want policy backing
            if state.tool_response is not None:
                query = state.user_query or ""
                out = retrieve_policy(query, fetch_k=6, top_k=2, alpha=0.85)
                state.rag_results = _normalize_rag_output(out)
            else:
                state.rag_results = []
    except TypeError:
        # fallback in case retriever signature doesn't accept kwargs
        try:
            out = retrieve_policy(state.user_query, top_k=2)
            state.rag_results = _normalize_rag_output(out)
        except Exception as e:
            state.errors.append(f"RAG error: {e}")
            state.rag_results = []
    except Exception as e:
        state.errors.append(f"RAG error: {e}")
        state.rag_results = []

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
