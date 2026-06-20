from __future__ import annotations

import re


def route_request(messages: list[object], provider: str, model_override: str | None = None) -> dict:
    text = "\n".join(getattr(message, "content", "") if not isinstance(message, dict) else message.get("content", "") for message in messages)
    if model_override:
        return {"route_decision": "user_override", "reason": "model override supplied", "selected_model": model_override, "estimated_cost_difference": 0.0}
    risk = []
    if len(text.split()) > 1500:
        risk.append("long_context")
    if re.search(r"```|`[^`]+`|\b[a-zA-Z]\s*=", text):
        risk.append("code_or_math")
    if re.search(r"\b(legal|contract|clause|medical|finance|invoice|compliance|HIPAA)\b", text, flags=re.IGNORECASE):
        risk.append("regulated_domain")
    if re.search(r"\b(JSON only|strict JSON|reason step|prove|debug|root cause)\b", text, flags=re.IGNORECASE):
        risk.append("complex_reasoning_or_strict_output")
    if provider == "openai":
        cheap = "gpt-4o-mini"
        strong = "gpt-4o"
    else:
        cheap = "gemini-1.5-flash"
        strong = "gemini-1.5-pro"
    if risk:
        return {"route_decision": "stronger_model", "reason": ", ".join(risk), "selected_model": strong, "estimated_cost_difference": 0.0}
    return {"route_decision": "cheaper_model", "reason": "short low-risk prompt", "selected_model": cheap, "estimated_cost_difference": 0.0}
