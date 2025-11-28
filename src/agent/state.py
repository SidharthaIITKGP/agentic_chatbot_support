# src/agent/state.py
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

@dataclass
class State:
    """
    Simple state container for an agent turn.
    Fields:
      - user_query: original user text
      - intent: e.g. "order_status", "refund_status", "product_availability", "policy_query"
      - slots: dict for extracted slots (order_id, product_id, policy_type, etc.)
      - tool_response: result from tools (if called)
      - rag_results: list of retrieved policy snippets (from retriever)
      - final_answer: text produced by composer
      - history: optional conversation history
      - errors: list of errors encountered
    """
    user_query: str
    intent: Optional[str] = None
    slots: Dict[str, Any] = field(default_factory=dict)
    tool_response: Optional[Dict[str, Any]] = None
    rag_results: List[Dict[str, Any]] = field(default_factory=list)
    final_answer: Optional[str] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
