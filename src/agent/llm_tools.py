# src/agent/llm_tools.py
"""
LangChain tool wrappers for LLM-powered ReAct agent.
Converts existing functions into tools the LLM can call.
Includes validation to ensure required parameters are provided.
"""
from langchain.tools import tool
from typing import Optional
from src.tools.tools import get_order_status as _get_order_status
from src.tools.tools import get_refund_status as _get_refund_status
from src.tools.tools import get_inventory as _get_inventory
from src.rag.retriever import retrieve_policy as _retrieve_policy


@tool
def get_order_status(order_id: str = "") -> dict:
    """
    Get the current status of a customer order.
    
    IMPORTANT: This tool REQUIRES an order ID (4-5 digit number like "98762").
    
    DO NOT call this tool if:
    - Customer hasn't mentioned an order ID yet
    - You're not certain what the order ID is
    - Order ID is empty or None
    
    Instead, respond directly: "I can help you track your order. Could you please provide your order ID? (Example: 98762)"
    
    Only use this tool when you have a valid order ID from the customer.
    
    Args:
        order_id: The order ID (e.g., "98762", "54321"). MUST be a 4-5 digit number.
    
    Returns:
        Dictionary with order details including status, expected delivery, and reason for any delays.
    """
    # Strict validation - no tolerance for missing or invalid order IDs
    if not order_id or len(order_id.strip()) < 4:
        # Return a response that makes the LLM ask the user
        raise ValueError(
            "Order ID is required to check order status. Please ask the customer: "
            "'I can help you track your order. Could you please provide your order ID? (Example: 98762)'"
        )
    
    return _get_order_status(order_id)


@tool
def get_refund_status(order_id: str = "") -> dict:
    """
    Check the refund status for an order.
    
    IMPORTANT: This tool REQUIRES an order ID (4-5 digit number like "98762").
    
    DO NOT call this tool if:
    - Customer hasn't mentioned an order ID yet
    - You're not certain what the order ID is
    - Order ID is empty or None
    
    Instead, respond directly: "I can help you check your refund status. Could you please provide your order ID?"
    
    Only use this tool when you have a valid order ID from the customer.
    
    Args:
        order_id: The order ID for which to check refund status. MUST be a 4-5 digit number.
    
    Returns:
        Dictionary with refund status, amount, and processed date if applicable.
    """
    # Strict validation
    if not order_id or len(order_id.strip()) < 4:
        raise ValueError(
            "Order ID is required to check refund status. Please ask the customer: "
            "'I can help you check your refund status. Could you please provide your order ID?'"
        )
    
    return _get_refund_status(order_id)


@tool
def check_product_availability(product_id: str = "") -> dict:
    """
    Check if a product is in stock and available for purchase.
    
    IMPORTANT: This tool REQUIRES a product ID (e.g., "P123", "P456").
    
    DO NOT call this tool if:
    - Customer hasn't mentioned a product ID yet
    - You're not certain what the product ID is
    - Product ID is empty or None
    
    Instead, respond directly: "I can help you check product availability. Could you please provide the product ID? (Example: P123)"
    
    Only use this tool when you have a valid product ID from the customer.
    
    Args:
        product_id: The product ID (e.g., "P123", "PROD456"). MUST be provided.
    
    Returns:
        Dictionary with availability status, quantity, and restock date if out of stock.
    """
    # Strict validation
    if not product_id or len(product_id.strip()) < 2:
        raise ValueError(
            "Product ID is required to check availability. Please ask the customer: "
            "'I can help you check product availability. Could you please provide the product ID? (Example: P123)'"
        )
    
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
