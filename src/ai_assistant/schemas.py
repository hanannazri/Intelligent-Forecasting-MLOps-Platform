from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel


AllowedIntent = Literal[
    "get_stockout_products",
    "get_reorder_recommendation",
    "get_inventory_summary",
    "fallback",
]


class ToolCall(BaseModel):
    intent: AllowedIntent
    arguments: Dict[str, Any] = {}


class AssistantResponse(BaseModel):
    question: str
    intent: AllowedIntent
    arguments: Dict[str, Any] = {}
    data: Optional[Any] = None
    answer: str
    source: str = "Verified forecast and inventory recommendation data"