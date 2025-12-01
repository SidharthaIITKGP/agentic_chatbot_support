# src/tools/tools.py
"""
Mock API tools.

Functions:
- get_order_status(order_id: str) -> dict
- get_refund_status(order_id: str) -> dict
- get_inventory(product_id: str) -> dict

Also:
- load_all_mock_data(): loads JSON files from the local folder (orders.json, refunds.json, inventory.json)
- get_langchain_tools(): returns a list of LangChain Tool wrappers
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List

PACKAGE_DIR = Path(__file__).parent
ORDERS_PATH = PACKAGE_DIR / "orders.json"
REFUNDS_PATH = PACKAGE_DIR / "refunds.json"
INVENTORY_PATH = PACKAGE_DIR / "inventory.json"

# load on first use (lazy)
_DATA_CACHE: Dict[str, Optional[Dict[str, Any]]] = {
    "orders": None,
    "refunds": None,
    "inventory": None,
}

def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return {}

def load_all_mock_data():
    """Loads and caches all mock JSON data and returns a tuple (orders, refunds, inventory)."""
    if _DATA_CACHE["orders"] is None:
        _DATA_CACHE["orders"] = _load_json(ORDERS_PATH)
    if _DATA_CACHE["refunds"] is None:
        _DATA_CACHE["refunds"] = _load_json(REFUNDS_PATH)
    if _DATA_CACHE["inventory"] is None:
        _DATA_CACHE["inventory"] = _load_json(INVENTORY_PATH)
    return _DATA_CACHE["orders"], _DATA_CACHE["refunds"], _DATA_CACHE["inventory"]


def get_order_status(order_id: str) -> Dict[str, Any]:
    """
    Return mock order status for a given order_id.
    If order not found, returns {"error": "Order not found"}.
    """
    orders, _, _ = load_all_mock_data()
    if not orders:
        return {"error": "Orders DB empty or not found."}
    entry = orders.get(order_id)
    if not entry:
        return {"error": "Order not found", "order_id": order_id}
    return entry


def get_refund_status(order_id: str) -> Dict[str, Any]:
    """
    Return mock refund status for a given order_id.
    If not found, return {"error": "Order not found"}.
    """
    _, refunds, _ = load_all_mock_data()
    if not refunds:
        return {"error": "Refunds DB empty or not found."}
    entry = refunds.get(order_id)
    if not entry:
        return {"error": "Order not found", "order_id": order_id}
    return entry


def get_inventory(product_id: str) -> Dict[str, Any]:
    """
    Return mock inventory info for a given product_id.
    If not found, return {"error": "Product not found"}.
    """
    _, _, inventory = load_all_mock_data()
    if not inventory:
        return {"error": "Inventory DB empty or not found."}
    entry = inventory.get(product_id)
    if not entry:
        return {"error": "Product not found", "product_id": product_id}
    return entry


# Optional: LangChain Tool wrappers factory
def get_langchain_tools() -> List[Any]:
    """
    Return a list of LangChain Tool objects wrapping the above functions.

    Tools:
    - get_order_status (name: "get_order_status")
    - get_refund_status (name: "get_refund_status")
    - get_inventory (name: "get_inventory")
    """
    from langchain_core.tools import Tool

    tools = [
        Tool.from_function(get_order_status, name="get_order_status", description="Get order status by order ID"),
        Tool.from_function(get_refund_status, name="get_refund_status", description="Get refund status by order ID"),
        Tool.from_function(get_inventory, name="get_inventory", description="Get inventory by product ID"),
    ]
    return tools


