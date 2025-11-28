# src/agent/nodes.py
"""
Nodes for LangGraph agent.
Each node is a callable that receives AgentState and returns updated state dict.
"""

from typing import Dict, Any
from .state import AgentState
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


def classify_intent_node(state: AgentState) -> Dict[str, Any]:
    """
    Determine intent and extract simple slots. Handle follow-up case where user
    replies with only digits (treat as order_id and inherit previous intent).
    """
    raw = (state.get("user_query") or "").strip()
    slots = state.get("slots", {}).copy()
    intent = state.get("intent")
    history = state.get("history", [])

    # --- 1) Quick follow-up: user replied with only digits -> treat as order_id slot ---
    if raw.isdigit():
        slots["order_id"] = raw

        # if we don't have a strong intent yet, try to inherit from history
        if not intent and history:
            prev = history[-1] if history else {}
            prev_intent = prev.get("intent")
            if prev_intent in ("order_status", "refund_status", "delivery_delay", None):
                return {"intent": "order_status", "slots": slots}

        # if current intent is ambiguous, prefer order_status for digits
        if intent in (None, "", "policy_query"):
            return {"intent": "order_status", "slots": slots}

    # --- 2) Otherwise use normal classifier ---
    res = classify_intent(state.get("user_query", ""))
    new_intent = res.get("intent")
    extracted_slots = res.get("slots", {}) or {}
    
    # Merge slots
    slots.update(extracted_slots)
    
    return {"intent": new_intent, "slots": slots}


def ask_for_slot_node(state: AgentState) -> Dict[str, Any]:
    """
    If intent requires a particular slot and it's missing, ask for it.
    """
    intent = state.get("intent")
    slots = state.get("slots", {})
    
    # Order-related intents that need order_id
    if intent in ("order_status", "refund_status", "delivery_delay"):
        if not slots.get("order_id"):
            return {"final_answer": CLARIFY_FOR_ORDER_ID}

    # Product availability intent requires product_id
    if intent in ("product_availability", "inventory_check"):
        if not slots.get("product_id"):
            return {"final_answer": CLARIFY_FOR_PRODUCT_ID}

    return {}


def call_tool_node(state: AgentState) -> Dict[str, Any]:
    """
    Call the appropriate tool based on intent & slots.
    """
    intent = state.get("intent")
    slots = state.get("slots", {})
    errors = list(state.get("errors", []))
    
    tool_response = None
    
    try:
        if intent == "order_status":
            oid = slots.get("order_id")
            if oid:
                tool_response = get_order_status(oid)
        elif intent == "refund_status":
            oid = slots.get("order_id")
            if oid:
                tool_response = get_refund_status(oid)
        elif intent == "product_availability":
            pid = slots.get("product_id")
            if pid:
                tool_response = get_inventory(pid)
    except Exception as e:
        errors.append(str(e))
    
    return {"tool_response": tool_response, "errors": errors}


def call_rag_node(state: AgentState) -> Dict[str, Any]:
    """
    For intents that want policy lookup or when tool response exists,
    call the retriever and store normalized results.
    """
    intent = state.get("intent")
    user_query = state.get("user_query", "")
    tool_response = state.get("tool_response")
    rag_results = []
    
    try:
        # Intents that primarily want policy text
        if intent in ("charges_query", "return_policy", "delivery_delay", "policy_query"):
            out = retrieve_policy(user_query, fetch_k=10, top_k=3, alpha=0.85)
            rag_results = _normalize_rag_output(out)
        else:
            # Also call RAG when we got a tool response but want policy backing
            if tool_response is not None:
                out = retrieve_policy(user_query, fetch_k=6, top_k=2, alpha=0.85)
                rag_results = _normalize_rag_output(out)
    except Exception as e:
        # Fallback - try simpler signature
        try:
            out = retrieve_policy(user_query)
            rag_results = _normalize_rag_output(out)
        except Exception:
            rag_results = []
    
    return {"rag_results": rag_results}


def compose_node(state: AgentState) -> Dict[str, Any]:
    """
    Call composer to produce final_answer from tool_response + rag_results.
    """
    from .composer import compose_final_answer
    from .state import State
    
    errors = list(state.get("errors", []))
    
    try:
        # Convert AgentState to legacy State for composer compatibility
        legacy_state = State(
            user_query=state.get("user_query", ""),
            intent=state.get("intent"),
            slots=state.get("slots", {}),
            tool_response=state.get("tool_response"),
            rag_results=state.get("rag_results", []),
            final_answer=state.get("final_answer"),
            history=state.get("history", []),
            errors=errors
        )
        
        final_answer = compose_final_answer(legacy_state)
    except Exception as e:
        errors.append(str(e))
        final_answer = "Sorry — something went wrong while composing the answer."
    
    return {"final_answer": final_answer, "errors": errors}
