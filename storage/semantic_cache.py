from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from optimizer.protect import extract_sensitive_facts
from optimizer.semantic_validator import SemanticValidator
from storage.db import cache_get, cache_set


TIME_SENSITIVE = re.compile(r"\b(today|now|current|latest|this week|this month|yesterday|tomorrow|price|weather|news)\b", re.IGNORECASE)


def exact_cache_key(provider: str, model: str, optimized_prompt: str, system_prompt: str, compression_level: str, temperature: float | None) -> str:
    payload = {
        "provider": provider,
        "model": model,
        "optimized_prompt_hash": hashlib.sha256(optimized_prompt.encode()).hexdigest(),
        "system_prompt_hash": hashlib.sha256(system_prompt.encode()).hexdigest(),
        "compression_level": compression_level,
        "temperature": temperature,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def semantic_cache_lookup(prompt: str, provider: str, model: str, threshold: float = 0.92) -> dict[str, Any]:
    trace = {"exact_hit": False, "semantic_hit": False, "similarity": 0.0, "rejection_reason": None}
    if TIME_SENSITIVE.search(prompt):
        trace["rejection_reason"] = "time_sensitive_prompt"
        return {"hit": None, "trace": trace}
    # Conservative MVP: expose semantic-cache decisioning and safety checks, but avoid broad reuse
    # unless a future embedding index is present. Exact cache handles production reuse today.
    trace["rejection_reason"] = "embedding_index_unavailable"
    return {"hit": None, "trace": trace}


def protected_facts_match(a: str, b: str) -> bool:
    facts_a = extract_sensitive_facts(a)
    facts_b = extract_sensitive_facts(b)
    for key in ["id", "date", "money", "percentage", "url", "email", "number"]:
        if facts_a.get(key, set()) != facts_b.get(key, set()):
            return False
    return True


def semantic_similarity(a: str, b: str) -> float:
    return SemanticValidator().similarity(a, b)["semantic_similarity"]
