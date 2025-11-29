# src/agent/llm_tools.py
"""
LangChain tool wrappers for LLM-powered ReAct agent.
Converts existing functions into tools the LLM can call.
"""
from langchain.tools import tool
from typing import Optional
from src.tools.tools import get_order_status as _get_order_status
from src.tools.tools import get_refund_status as _get_refund_status
from src.tools.tools import get_inventory as _get_inventory
from src.rag.retriever import retrieve_policy as _retrieve_policy


@tool
def get_order_status(order_id: str) -> dict:
    """
    Get the current status of a customer order.
    
    Use this tool when the customer asks about:
    - Order tracking
    - Where their order is
    - Order status
    - Delivery status
    - "Where is my order"
    
    Args:
        order_id: The order ID (e.g., "98762", "54321")
    
    Returns:
        Dictionary with order details including status, expected delivery, and reason for any delays.
    """
    return _get_order_status(order_id)


@tool
def get_refund_status(order_id: str) -> dict:
    """
    Check the refund status for an order.
    
    Use this tool when the customer asks about:
    - Refund status
    - "Has my refund been processed"
    - "When will I get my refund"
    - Refund timeline
    
    Args:
        order_id: The order ID for which to check refund status
    
    Returns:
        Dictionary with refund status, amount, and processed date if applicable.
    """
    return _get_refund_status(order_id)


@tool
def check_product_availability(product_id: str) -> dict:
    """
    Check if a product is in stock and available for purchase.
    
    Use this tool when the customer asks about:
    - Product availability
    - "Is this product in stock"
    - Stock levels
    - Product inventory
    - "Can I buy product X"
    
    Args:
        product_id: The product ID (e.g., "P123", "PROD456")
    
    Returns:
        Dictionary with availability status, quantity, and restock date if out of stock.
    """
    return _get_inventory(product_id)


@tool
def search_policy_documents(query: str) -> str:
    """
    Search the company policy knowledge base for relevant information.
    
    Use this tool when the customer asks about:
    - Return policy
    - Refund policy
    - Cancellation policy
    - Delivery policy
    - Charges and fees
    - Replacement policy
    - Any company policies or rules
    
    Args:
        query: The user's question about policies (e.g., "what is the return policy", "can I cancel my order")
    
    Returns:
        Relevant policy information from the knowledge base.
    """
    try:
        results = _retrieve_policy(query, fetch_k=10, top_k=3, alpha=0.85)
        
        # Normalize output
        if results is None:
            return "No relevant policy information found."
        
        if isinstance(results, dict):
            docs = results.get("final", []) or results.get("results", []) or []
        elif isinstance(results, list):
            docs = results
        else:
            return "No relevant policy information found."
        
        if not docs:
            return "No relevant policy information found."
        
        # Format the results
        formatted = []
        for doc in docs[:3]:  # Top 3 results
            content = doc.get("text", "") or doc.get("content", "")
            metadata = doc.get("metadata", {})
            source = metadata.get("doc_id", "unknown")
            
            if content:
                formatted.append(f"Source: {source}\n{content[:500]}")
        
        if not formatted:
            return "No relevant policy information found."
        
        return "\n\n---\n\n".join(formatted)
    
    except Exception as e:
        return f"Error searching policies: {str(e)}"


# List of all tools for the agent
ALL_TOOLS = [
    get_order_status,
    get_refund_status,
    check_product_availability,
    search_policy_documents,
]
