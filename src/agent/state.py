# src/agent/state.py
from typing import TypedDict, Optional, Dict, Any, List, Annotated
from langgraph.graph import MessagesState
from operator import add

class AgentState(TypedDict):
    """
    State container for LangGraph agent.
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
    intent: Optional[str]
    slots: Dict[str, Any]
    tool_response: Optional[Dict[str, Any]]
    rag_results: List[Dict[str, Any]]
    final_answer: Optional[str]
    history: List[Dict[str, Any]]
    errors: Annotated[List[str], add]  # Use operator.add to append errors

# Keep the old State class for backward compatibility during transition
from dataclasses import dataclass, field

@dataclass
class State:
    """
    Legacy state container (for backward compatibility).
    Use AgentState for new LangGraph implementation.
    """
    user_query: str
    intent: Optional[str] = None
    slots: Dict[str, Any] = field(default_factory=dict)
    tool_response: Optional[Dict[str, Any]] = None
    rag_results: List[Dict[str, Any]] = field(default_factory=list)
    final_answer: Optional[str] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
