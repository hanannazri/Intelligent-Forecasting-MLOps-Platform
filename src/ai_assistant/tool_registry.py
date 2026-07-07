from src.ai_assistant.tools import (
    get_stockout_products,
    get_reorder_recommendation,
    get_inventory_summary,
)

TOOL_REGISTRY = {
    "get_stockout_products": {
        "function": get_stockout_products,
        "description": "Returns products with stockout risk from verified inventory data.",
    },
    "get_reorder_recommendation": {
        "function": get_reorder_recommendation,
        "description": "Returns reorder recommendation for a specific item.",
    },
    "get_inventory_summary": {
        "function": get_inventory_summary,
        "description": "Returns an inventory health summary.",
    },
}

FALLBACK_MESSAGE = (
    "I can only answer questions about inventory, demand forecasting, "
    "stockout risks, reorder recommendations, and inventory summaries."
)


def execute_tool(intent: str, arguments: dict | None = None):
    arguments = arguments or {}

    # Unknown or unsupported intent
    if intent not in TOOL_REGISTRY:
        return {"error": FALLBACK_MESSAGE}

    tool_function = TOOL_REGISTRY[intent]["function"]

    # Reorder recommendation requires an item_id
    if intent == "get_reorder_recommendation":
        item_id = arguments.get("item_id")

        if not item_id:
            return {
                "error": (
                    "Please specify an item ID. "
                    "Example: FOODS_3_090."
                )
            }

        return tool_function(item_id)

    # Inventory summary and stockout products require no arguments
    return tool_function()