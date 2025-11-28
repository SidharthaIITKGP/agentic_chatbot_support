# src/agent/composer.py
"""
Simple composer that formats tool results and RAG snippets into a user-facing reply.
"""

from typing import Dict, Any
from .state import State
from .prompts import COMPOSER_HEADER, COMPOSER_POLICY_FOOTER, COMPOSER_NO_INFO

def _format_tool_response(state: State) -> str:
    tr = state.tool_response
    if not tr:
        return ""
    # Order status formatting
    if state.intent == "order_status":
        if tr.get("error"):
            return f"I couldn't find the order {state.slots.get('order_id')}. Please check the order ID."
        status = tr.get("order_status") or tr.get("status") or tr.get("status")
        expected = tr.get("expected_delivery") or tr.get("order_expected")
        last = tr.get("last_updated")
        reason = tr.get("delay_reason")
        s = f"Order {tr.get('order_id')} is currently: {status}."
        if expected:
            s += f" Expected delivery: {expected}."
        if reason:
            s += f" Reason: {reason}."
        return s

    if state.intent == "refund_status":
        if tr.get("error"):
            return f"I couldn't find the refund for order {state.slots.get('order_id')}. Please check the order ID."
        rstat = tr.get("refund_status") or tr.get("status")
        amount = tr.get("refund_amount") or tr.get("amount")
        processed = tr.get("processed_at") or tr.get("processed")
        s = f"Refund status for order {tr.get('order_id')}: {rstat}."
        if amount:
            s += f" Amount: {amount}."
        if processed:
            s += f" Processed at: {processed}."
        return s

    if state.intent == "product_availability":
        if tr.get("error"):
            return f"I couldn't find product {state.slots.get('product_id')}. Please check the product ID."
        in_stock = tr.get("in_stock")
        qty = tr.get("quantity_available") or tr.get("quantity")
        s = f"Product {tr.get('product_id')}: {'In stock' if in_stock else 'Out of stock'}."
        if qty is not None:
            s += f" Quantity available: {qty}."
        restock = tr.get("restock_date") or tr.get("restock")
        if restock:
            s += f" Restock expected: {restock}."
        return s

    # Fallback
    return str(tr)

def compose_final_answer(state: State) -> str:
    # If composer was short-circuited earlier (e.g., missing slot) return that
    if state.final_answer:
        return state.final_answer

    pieces = []

    # Tool-based response first
    tool_part = _format_tool_response(state)
    if tool_part:
        pieces.append(tool_part)

    # RAG-based policy snippets
    if state.rag_results:
        # For policy queries without tool response, provide actual answer from RAG
        if not tool_part and state.intent in ("charges_query", "return_policy", "delivery_delay", "policy_query", "cancellation_policy"):
            # Extract and summarize the most relevant policy information
            policy_texts = []
            for s in state.rag_results[:2]:  # Use top 2 most relevant results
                text = s.get("text", "")
                if text:
                    # Get first few sentences (approximately 200 chars)
                    snippet = text[:300].strip()
                    if len(text) > 300:
                        snippet += "..."
                    policy_texts.append(snippet)
            
            if policy_texts:
                pieces.append("\n\n".join(policy_texts))
        
        # Add policy reference footer
        provenance = []
        for s in state.rag_results:
            meta = s.get("metadata", {}) or {}
            doc_id = meta.get("doc_id") or "policy"
            if doc_id not in provenance:
                provenance.append(doc_id)
        if provenance:
            pieces.append(f"\n_Policy reference: {', '.join(provenance)}_")
    elif not tool_part:
        # nothing found
        return COMPOSER_NO_INFO

    return "\n\n".join(pieces)
