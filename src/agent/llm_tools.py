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
def get_order_status(order_id: str) -> dict:
    """
    Get the current status of a customer order.
    
    Use this tool when the customer asks about:
    - Order tracking ("Where is my order 98762?")
    - Order status ("What's the status of order 98762?")
    - Delivery information
    
    The order_id should be extracted from the customer's query.
    If the query contains a 4-5 digit number, that's likely the order ID.
    
    Args:
        order_id: The order ID number (e.g., "98762", "54321")
    
    Returns:
        Dictionary with order details including status, expected delivery, and reason for any delays.
    
    Example queries:
    - "Where is my order 98762?" → order_id="98762"
    - "Track order 54321" → order_id="54321"
    """
    # Validation
    if not order_id or len(order_id.strip()) < 4:
        raise ValueError(
            "Order ID is required to check order status. Please ask the customer: "
            "'I can help you track your order. Could you please provide your order ID? (Example: 98762)'"
        )
    
    return _get_order_status(order_id.strip())


@tool
def get_refund_status(order_id: str) -> dict:
    """
    Check the refund status for an order.
    
    Use this tool when the customer asks about:
    - Refund status ("What's the status of my refund for order 98762?")
    - Refund processing ("Has my refund been processed?")
    - When they'll receive their refund
    
    The order_id should be extracted from the customer's query.
    If the query contains a 4-5 digit number, that's likely the order ID.
    
    Args:
        order_id: The order ID number (e.g., "98762", "54321")
    
    Returns:
        Dictionary with refund status, amount, and processed date if applicable.
    
    Example queries:
    - "What's the status of my refund for order 98762?" → order_id="98762"
    - "Refund status for 54321" → order_id="54321"
    """
    # Validation
    if not order_id or len(order_id.strip()) < 4:
        raise ValueError(
            "Order ID is required to check refund status. Please ask the customer: "
            "'I can help you check your refund status. Could you please provide your order ID?'"
        )
    
    return _get_refund_status(order_id.strip())


@tool
def check_product_availability(product_id: str) -> dict:
    """
    Check if a product is in stock and available for purchase.
    
    Use this tool when the customer asks about:
    - Product availability ("Is product PROD123 available?")
    - Stock levels ("Do you have PROD123 in stock?")
    - Whether they can buy a product
    
    The product_id should be extracted from the customer's query.
    Look for product codes/IDs in the query (often start with "PROD" or "P").
    
    Args:
        product_id: The product ID (e.g., "PROD123", "P456")
    
    Returns:
        Dictionary with availability status, quantity, and restock date if out of stock.
    
    Example queries:
    - "Is product PROD123 available?" → product_id="PROD123"
    - "Check stock for P456" → product_id="P456"
    """
    # Validation
    if not product_id or len(product_id.strip()) < 2:
        raise ValueError(
            "Product ID is required to check availability. Please ask the customer: "
            "'I can help you check product availability. Could you please provide the product ID? (Example: PROD123)'"
        )
    
    return _get_inventory(product_id.strip())


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
