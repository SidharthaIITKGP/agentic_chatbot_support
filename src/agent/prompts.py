# src/agent/prompts.py
"""
Prompt templates and messages.

These are used by the composer and for clarifying-slot messages.
"""

CLARIFY_FOR_ORDER_ID = "I can check that for you — could you share the order ID (e.g., 98762)?"
CLARIFY_FOR_PRODUCT_ID = "Please provide the product ID (e.g., P123) so I can check availability."

# Composer templates
COMPOSER_HEADER = "Here’s what I found:\n\n"
COMPOSER_POLICY_FOOTER = "\n\nPolicy reference (for your records):\n{provenance}\n\nIf you want me to do an action (refund/replace/cancel), tell me and I can open a request."

COMPOSER_NO_INFO = "I couldn't find any information matching your request. Could you share more details (order ID or product ID)?"
