import json
import os
import re

from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def extract_item_id(question: str):
    match = re.search(r"FOODS_\d+_\d+", question.upper())
    return match.group(0) if match else None


def rule_based_route(question: str):
    q = question.lower().strip()
    item_id = extract_item_id(question)

    if item_id and any(word in q for word in ["reorder", "restock", "replenishment", "replenish", "buy", "restocking"]):
        return {
            "intent": "get_reorder_recommendation",
            "arguments": {"item_id": item_id},
            "routing_method": "rule_based"
        }

    if any(word in q for word in ["summary", "overview", "inventory health", "inventory status", "overall inventory", "business summary"]):
        return {
            "intent": "get_inventory_summary",
            "arguments": {},
            "routing_method": "rule_based"
        }

    if any(word in q for word in ["stockout", "stock out", "run out", "shortage", "low stock", "high risk", "risk products", "need attention", "at risk", "restocking"]):
        return {
            "intent": "get_stockout_products",
            "arguments": {},
            "routing_method": "rule_based"
        }

    return None


def gemini_route(question: str):
    item_id = extract_item_id(question)

    prompt = f"""
You are an intent router for a retail inventory assistant.

Classify the user question into exactly one of these intents:
- get_inventory_summary
- get_stockout_products
- get_reorder_recommendation
- fallback

Rules:
- get_inventory_summary: overall inventory summary, overview, status, health, total products, business inventory risk.
- get_stockout_products: products at risk, stockout risk, low stock, shortage, urgent products, products needing attention.
- get_reorder_recommendation: reorder/restock/replenishment/buy quantity for one specific item_id only.
- fallback: outside this retail inventory forecasting project.

Return only valid JSON:
{{
  "intent": "intent_name",
  "arguments": {{
    "item_id": "ITEM_ID_OR_NULL"
  }}
}}

Question: {question}
Detected item_id: {item_id}
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        text = response.text.strip()
        text = text.replace("```json", "").replace("```", "").strip()

        route = json.loads(text)

        intent = route.get("intent", "fallback")
        arguments = route.get("arguments", {})

        if intent == "get_reorder_recommendation" and not item_id:
            intent = "get_stockout_products"
            arguments = {}

        elif intent == "get_reorder_recommendation":
            arguments["item_id"] = item_id or arguments.get("item_id")

        else:
            arguments = {}

        return {
            "intent": intent,
            "arguments": arguments,
            "routing_method": "gemini"
        }

    except Exception as e:
        return {
            "intent": "fallback",
            "arguments": {},
            "routing_method": "gemini_error",
            "error": str(e)
        }


def route_with_gemini(question: str) -> dict:
    rule_route = rule_based_route(question)

    if rule_route:
        return rule_route

    return gemini_route(question)