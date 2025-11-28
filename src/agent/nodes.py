# src/agent/nodes.py
"""
ReAct Nodes for LangGraph agent.
Each node implements part of the Reasoning and Acting cycle:
  1. reasoning_node: Thinks about what to do next
  2. action_node: Executes the decided action (tool/RAG/finish)
  3. observation_node: Processes results and updates scratchpad
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
    Initial step: Determine intent and extract simple slots.
    Handle follow-up case where user replies with only digits.
    """
    raw = (state.get("user_query") or "").strip()
    slots = state.get("slots", {}).copy()
    intent = state.get("intent")
    history = state.get("history", [])

    # Quick follow-up: user replied with only digits -> treat as order_id slot
    if raw.isdigit():
        slots["order_id"] = raw
        if not intent and history:
            prev = history[-1] if history else {}
            prev_intent = prev.get("intent")
            if prev_intent in ("order_status", "refund_status", "delivery_delay", None):
                return {"intent": "order_status", "slots": slots, "iteration": 0, "scratchpad": ""}
        if intent in (None, "", "policy_query"):
            return {"intent": "order_status", "slots": slots, "iteration": 0, "scratchpad": ""}

    # Otherwise use normal classifier
    res = classify_intent(state.get("user_query", ""))
    new_intent = res.get("intent")
    extracted_slots = res.get("slots", {}) or {}
    slots.update(extracted_slots)
    
    return {"intent": new_intent, "slots": slots, "iteration": 0, "scratchpad": ""}


def reasoning_node(state: AgentState) -> Dict[str, Any]:
    """
    ReAct Reasoning Step: Agent thinks about what to do next.
    Generates a thought and decides on the next action.
    
    Actions:
      - ask_for_slot: Need more information from user
      - call_tool: Execute a tool (order lookup, refund check, inventory)
      - call_rag: Query policy knowledge base
      - finish: Have enough information to answer
    """
    intent = state.get("intent")
    slots = state.get("slots", {})
    iteration = state.get("iteration", 0)
    scratchpad = state.get("scratchpad", "")
    tool_response = state.get("tool_response")
    rag_results = state.get("rag_results", [])
    
    # Build thought process
    thought = f"Iteration {iteration + 1}: "
    
    # Check if we need slots
    if intent in ("order_status", "refund_status", "delivery_delay"):
        if not slots.get("order_id"):
            thought += "User needs order tracking but hasn't provided order ID. I should ask for it."
            action = "ask_for_slot"
            action_input = {"slot_type": "order_id"}
            
            updated_scratchpad = scratchpad + f"\nThought: {thought}\nAction: {action}"
            return {
                "thought": thought,
                "action": action,
                "action_input": action_input,
                "scratchpad": updated_scratchpad,
                "iteration": iteration + 1
            }
    
    if intent in ("product_availability", "inventory_check"):
        if not slots.get("product_id"):
            thought += "User asking about product but no product ID. Need to ask for it."
            action = "ask_for_slot"
            action_input = {"slot_type": "product_id"}
            
            updated_scratchpad = scratchpad + f"\nThought: {thought}\nAction: {action}"
            return {
                "thought": thought,
                "action": action,
                "action_input": action_input,
                "scratchpad": updated_scratchpad,
                "iteration": iteration + 1
            }
    
    # Decide if we need to call tools
    if intent in ("order_status", "refund_status") and not tool_response:
        thought += f"I have order ID {slots.get('order_id')}. I should call the {intent} tool to get current status."
        action = "call_tool"
        action_input = {"intent": intent, "slots": slots}
        
        updated_scratchpad = scratchpad + f"\nThought: {thought}\nAction: {action}"
        return {
            "thought": thought,
            "action": action,
            "action_input": action_input,
            "scratchpad": updated_scratchpad,
            "iteration": iteration + 1
        }
    
    if intent in ("product_availability", "inventory_check") and not tool_response:
        thought += f"I have product ID {slots.get('product_id')}. Calling inventory tool to check availability."
        action = "call_tool"
        action_input = {"intent": intent, "slots": slots}
        
        updated_scratchpad = scratchpad + f"\nThought: {thought}\nAction: {action}"
        return {
            "thought": thought,
            "action": action,
            "action_input": action_input,
            "scratchpad": updated_scratchpad,
            "iteration": iteration + 1
        }
    
    # Decide if we need RAG
    if intent in ("charges_query", "return_policy", "delivery_delay", "policy_query"):
        if not rag_results:
            thought += "This is a policy question. I should search the knowledge base for relevant information."
            action = "call_rag"
            action_input = {"query": state.get("user_query", "")}
            
            updated_scratchpad = scratchpad + f"\nThought: {thought}\nAction: {action}"
            return {
                "thought": thought,
                "action": action,
                "action_input": action_input,
                "scratchpad": updated_scratchpad,
                "iteration": iteration + 1
            }
    else:
        # If we have tool response but no RAG yet, get supporting policy info
        if tool_response and not rag_results:
            thought += "I have the tool result. Let me get relevant policy information to provide complete context."
            action = "call_rag"
            action_input = {"query": state.get("user_query", "")}
            
            updated_scratchpad = scratchpad + f"\nThought: {thought}\nAction: {action}"
            return {
                "thought": thought,
                "action": action,
                "action_input": action_input,
                "scratchpad": updated_scratchpad,
                "iteration": iteration + 1
            }
    
    # We have everything we need - finish
    thought += "I have all the information needed. Time to compose the final answer."
    action = "finish"
    action_input = {}
    
    updated_scratchpad = scratchpad + f"\nThought: {thought}\nAction: {action}"
    return {
        "thought": thought,
        "action": action,
        "action_input": action_input,
        "scratchpad": updated_scratchpad,
        "iteration": iteration + 1
    }


def action_node(state: AgentState) -> Dict[str, Any]:
    """
    ReAct Action Step: Execute the action decided by reasoning_node.
    """
    action = state.get("action")
    action_input = state.get("action_input", {})
    scratchpad = state.get("scratchpad", "")
    errors = list(state.get("errors", []))
    
    observation = ""
    updates = {}
    
    try:
        if action == "ask_for_slot":
            slot_type = action_input.get("slot_type")
            if slot_type == "order_id":
                updates["final_answer"] = CLARIFY_FOR_ORDER_ID
                observation = "Asked user for order ID. Waiting for response."
            elif slot_type == "product_id":
                updates["final_answer"] = CLARIFY_FOR_PRODUCT_ID
                observation = "Asked user for product ID. Waiting for response."
        
        elif action == "call_tool":
            intent = action_input.get("intent")
            slots = action_input.get("slots", {})
            
            if intent == "order_status":
                oid = slots.get("order_id")
                if oid:
                    tool_response = get_order_status(oid)
                    updates["tool_response"] = tool_response
                    observation = f"Retrieved order status: {tool_response.get('status', 'N/A')}"
            
            elif intent == "refund_status":
                oid = slots.get("order_id")
                if oid:
                    tool_response = get_refund_status(oid)
                    updates["tool_response"] = tool_response
                    observation = f"Retrieved refund info: {tool_response.get('refund_status', 'N/A')}"
            
            elif intent in ("product_availability", "inventory_check"):
                pid = slots.get("product_id")
                if pid:
                    tool_response = get_inventory(pid)
                    updates["tool_response"] = tool_response
                    observation = f"Retrieved inventory: {tool_response.get('availability', 'N/A')}"
        
        elif action == "call_rag":
            query = action_input.get("query", state.get("user_query", ""))
            intent = state.get("intent")
            
            # Adjust RAG parameters based on intent
            if intent in ("charges_query", "return_policy", "delivery_delay", "policy_query"):
                out = retrieve_policy(query, fetch_k=10, top_k=3, alpha=0.85)
            else:
                out = retrieve_policy(query, fetch_k=6, top_k=2, alpha=0.85)
            
            rag_results = _normalize_rag_output(out)
            updates["rag_results"] = rag_results
            observation = f"Retrieved {len(rag_results)} relevant policy documents."
        
        elif action == "finish":
            observation = "Ready to compose final answer."
    
    except Exception as e:
        errors.append(str(e))
        observation = f"Error during action: {str(e)}"
    
    # Update scratchpad with observation
    updated_scratchpad = scratchpad + f"\nObservation: {observation}"
    
    updates["observation"] = observation
    updates["scratchpad"] = updated_scratchpad
    updates["errors"] = errors
    
    return updates


def compose_node(state: AgentState) -> Dict[str, Any]:
    """
    Final step: Compose the answer using all gathered information.
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
        
        # Add final thought to scratchpad
        scratchpad = state.get("scratchpad", "")
        scratchpad += f"\nFinal Answer: {final_answer[:100]}..."
        
    except Exception as e:
        errors.append(str(e))
        final_answer = "Sorry — something went wrong while composing the answer."
        scratchpad = state.get("scratchpad", "") + f"\nError: {str(e)}"
    
    return {"final_answer": final_answer, "scratchpad": scratchpad, "errors": errors}
