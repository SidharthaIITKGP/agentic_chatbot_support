# src/agent/memory.py
"""
Simple persistent memory for the agent.
Stores a lightweight JSON file per session_id containing:
- last_order_id
- last_product_id
- last_intent
- messages (short history)
"""
import os
import json
from typing import Dict, Any

DEFAULT_MEM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
os.makedirs(DEFAULT_MEM_DIR, exist_ok=True)

def memory_path(session_id: str) -> str:
    safe = session_id.replace("/", "_")
    return os.path.join(DEFAULT_MEM_DIR, f"agent_memory_{safe}.json")

def load_memory(session_id: str) -> Dict[str, Any]:
    path = memory_path(session_id)
    if not os.path.exists(path):
        return {"last_order_id": None, "last_product_id": None, "last_intent": None, "messages": []}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {"last_order_id": None, "last_product_id": None, "last_intent": None, "messages": []}

def save_memory(session_id: str, memory: Dict[str, Any]) -> None:
    path = memory_path(session_id)
    with open(path, "w") as f:
        json.dump(memory, f, indent=2)
