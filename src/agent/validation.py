# src/agent/validation.py
"""
Input validation layer for agent tools.
Extracts required parameters from user messages and validates before tool execution.
"""
import re
from typing import Optional, Dict, Any


def extract_order_id(text: str) -> Optional[str]:
    """Extract order ID from text. Returns None if not found."""
    # Look for 4-5 digit numbers
    match = re.search(r'\b(\d{4,5})\b', text)
    return match.group(1) if match else None


def extract_product_id(text: str) -> Optional[str]:
    """Extract product ID from text (format: P followed by digits)."""
    match = re.search(r'\b(P\d+)\b', text, re.IGNORECASE)
    return match.group(1).upper() if match else None


def validate_order_request(messages: list) -> Dict[str, Any]:
    """
    Check if we have order ID from conversation history.
    Returns: {"has_order_id": bool, "order_id": str or None, "should_ask": bool}
    """
    # Check recent messages for order ID
    for msg in reversed(messages[-5:]):  # Check last 5 messages
        if hasattr(msg, 'content'):
            content = str(msg.content)
            order_id = extract_order_id(content)
            if order_id:
                return {
                    "has_order_id": True,
                    "order_id": order_id,
                    "should_ask": False
                }
    
    return {
        "has_order_id": False,
        "order_id": None,
        "should_ask": True
    }


def validate_product_request(messages: list) -> Dict[str, Any]:
    """
    Check if we have product ID from conversation history.
    Returns: {"has_product_id": bool, "product_id": str or None, "should_ask": bool}
    """
    for msg in reversed(messages[-5:]):
        if hasattr(msg, 'content'):
            content = str(msg.content)
            product_id = extract_product_id(content)
            if product_id:
                return {
                    "has_product_id": True,
                    "product_id": product_id,
                    "should_ask": False
                }
    
    return {
        "has_product_id": False,
        "product_id": None,
        "should_ask": True
    }


def should_ask_for_confirmation(messages: list) -> bool:
    """
    Check if user wants to perform a sensitive action (cancel, refund, etc.)
    Returns True if we should ask for confirmation.
    """
    if not messages:
        return False
    
    last_message = messages[-1]
    if hasattr(last_message, 'content'):
        content = str(last_message.content).lower()
        
        # Check for action keywords without confirmation
        action_keywords = ['cancel', 'refund', 'return', 'replace']
        confirmation_keywords = ['yes', 'confirm', 'proceed', 'go ahead']
        
        has_action = any(keyword in content for keyword in action_keywords)
        has_confirmation = any(keyword in content for keyword in confirmation_keywords)
        
        # If action requested but no confirmation, ask for it
        return has_action and not has_confirmation
    
    return False
