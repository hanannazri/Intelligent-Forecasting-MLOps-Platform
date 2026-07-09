from src.ai_assistant.tool_registry import execute_tool, FALLBACK_MESSAGE

try:
    from src.ai_assistant.gemini_router import route_with_gemini
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

def format_answer(intent: str, data):
    if intent == "fallback":
        return FALLBACK_MESSAGE

    if isinstance(data, dict) and "error" in data:
        return data["error"]

    if intent == "get_inventory_summary":
        return (
            "Inventory Summary:\n\n"
            f"- Total products analyzed: {data.get('total_products', 'N/A')}\n"
            f"- Total inventory records: {data.get('total_rows', 'N/A')}\n"
            f"- High-risk products: {data.get('high_risk_products', 'Not available')}\n"
            f"- Total reorder quantity: {data.get('total_reorder_quantity', 'Not available')}"
        )

    if intent == "get_stockout_products":
        if not data:
            return "No high-risk stockout products were found in the current inventory data."

        lines = ["Top Stockout-Risk Products:\n"]

        for i, row in enumerate(data[:10], start=1):
            item_id = row.get("item_id", "Unknown item")
            reorder_qty = row.get("reorder_qty", row.get("recommended_order_qty", "N/A"))
            risk = row.get("stockout_risk", row.get("risk_level", row.get("risk", "N/A")))

            lines.append(f"{i}. {item_id} | Risk: {risk} | Reorder Qty: {reorder_qty}")

        return "\n".join(lines)

    if intent == "get_reorder_recommendation":
        item_id = data.get("item_id", "Unknown item")
        reorder_qty = data.get("reorder_qty", data.get("recommended_order_qty", "N/A"))
        forecast = data.get("forecast_demand", data.get("total_28_day_forecast", data.get("forecast", "N/A")))
        risk = data.get("stockout_risk", data.get("risk_level", data.get("risk", "N/A")))

        return (
            f"Reorder Recommendation for {item_id}:\n\n"
            f"- 28-day forecast demand: {forecast}\n"
            f"- Recommended reorder quantity: {reorder_qty}\n"
            f"- Stockout risk: {risk}"
        )

    return FALLBACK_MESSAGE


def answer_question(question: str):
    if GEMINI_AVAILABLE:
        route = route_with_gemini(question)
    else:
        from src.ai_assistant.intent_router import route_question

        intent = route_question(question)

        # Build the same structure Gemini returns
        route = {
            "intent": intent,
            "arguments": {}
        }

    intent = route.get("intent", "fallback")
    arguments = route.get("arguments", {})

    data = execute_tool(intent, arguments)

    return format_answer(intent, data)