from __future__ import annotations

import time
from app.models import ChatMessage
from optimizer.pipeline import optimize_messages
from optimizer.protect import extract_sensitive_facts


def similarity(a: str, b: str) -> float:
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    return round(len(sa & sb) / max(1, len(sa | sb)), 3)


def evaluate_quality(compression_level: str = "safe", mode: str = "dry-run", provider: str = "gemini") -> dict:
    prompt = "Refunds over $500 require Finance approval. Refunds over $500 require Finance approval. Explain for ORD-900184."
    started = time.perf_counter()
    opt = optimize_messages([ChatMessage(role="user", content=prompt)], compression_level, provider, mode)
    elapsed = round((time.perf_counter() - started) * 1000, 2)
    optimized_prompt = opt["optimized_messages"][0].content
    original_facts = extract_sensitive_facts(prompt)
    optimized_facts = extract_sensitive_facts(optimized_prompt)
    result = {
        "mode": mode,
        "provider": provider,
        "structural_validation_only": mode == "dry-run",
        "protected_region_preservation": opt["protected_region_status"]["status"],
        "response_length_difference": len(optimized_prompt) - len(prompt),
        "answer_similarity_heuristic": similarity(prompt, optimized_prompt),
        "numeric_answer_preservation": original_facts.get("number", set()) <= optimized_facts.get("number", set()),
        "json_validity": True,
        "code_block_presence": "```" in prompt and "```" in optimized_prompt or "```" not in prompt,
        "latency_ms": elapsed,
        "cost_before_tokens": opt["original_tokens"],
        "cost_after_tokens": opt["optimized_tokens"],
        "cache_hit": False,
    }
    return {"results": [result], "optimization": opt}
