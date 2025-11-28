# src/agent/intent_classifier.py
"""
Improved rule-based intent & slot extractor.

Product ID regex now requires a 'P' followed by a digit to avoid matching
words like 'product' or 'purchase'.
"""

import re
from typing import Dict, Any

# Patterns
ORDER_RE = re.compile(r"\b(\d{4,})\b")
# Require P followed by a digit (e.g. P123); prevents matching 'product' or 'purchase'
PRODUCT_RE = re.compile(r"\bP(?=\d)[0-9A-Za-z_-]{1,12}\b", re.IGNORECASE)

def _has_any(tokens: list, text: str) -> bool:
    t = text.lower()
    return any(tok in t for tok in tokens)

def classify_intent(text: str) -> Dict[str, Any]:
    text_l = text.lower()
    slots = {}

    # Extract slots first
    order_match = ORDER_RE.search(text)
    if order_match:
        slots["order_id"] = order_match.group(1)

    product_match = PRODUCT_RE.search(text)
    if product_match:
        # Uppercase canonical product id
        slots["product_id"] = product_match.group(0).upper()

    # Helpful phrase checks (exact phrasings)
    if "how long" in text_l or "how many days" in text_l:
        if "refund" in text_l or "return" in text_l:
            return {"intent": "return_policy", "slots": slots}

    # Refund-focused
    if "refund" in text_l or "refund status" in text_l:
        if "status" in text_l or "check" in text_l or "what is" in text_l:
            return {"intent": "refund_status", "slots": slots}
        if "how long" in text_l or "policy" in text_l or "take" in text_l:
            return {"intent": "return_policy", "slots": slots}
        return {"intent": "refund_status", "slots": slots}

    # Tracking/order phrases -> order status
    if "where is my order" in text_l or "track" in text_l or ("where" in text_l and "order" in text_l):
        return {"intent": "order_status", "slots": slots}

    # If there is an order ID present, prefer order_status (unless refund keywords present)
    if "order" in text_l and order_match:
        return {"intent": "order_status", "slots": slots}

    # Delivery/lateness specific
    if "out for delivery" in text_l or ("delivery" in text_l and ("late" in text_l or "delay" in text_l)):
        if order_match:
            return {"intent": "order_status", "slots": slots}
        return {"intent": "delivery_delay", "slots": slots}

    # Charges / billing
    if _has_any(["charged", "charge", "extra charge", "why was i charged", "fees", "convenience fee"], text):
        return {"intent": "charges_query", "slots": slots}

    # Return / policy
    if _has_any(["return", "return window", "refund policy", "how long do refunds take"], text):
        return {"intent": "return_policy", "slots": slots}

    # Product availability: explicit product id or stock keywords
    if product_match or _has_any(["in stock", "stock", "available"], text):
        return {"intent": "product_availability", "slots": slots}

    # Fallback to policy lookup
    return {"intent": "policy_query", "slots": slots}
