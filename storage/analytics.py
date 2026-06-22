from __future__ import annotations

from storage.db import recent_traces, cache_stats


def analytics_summary() -> dict:
    traces = recent_traces(500)
    original = sum(t.get("original_tokens", 0) for t in traces)
    optimized = sum(t.get("optimized_tokens", 0) for t in traces)
    saved = max(0, original - optimized)
    protected_failures = sum(
        1 for t in traces for q in t.get("safety_checks", []) if not q.get("checks", {}).get("protected_regions_preserved", True)
    )
    quality_failures = sum(1 for t in traces if not t.get("accepted", False))
    savings_percent = round(saved / max(1, original) * 100, 2)
    cache = cache_stats()
    estimated_cost_saved = round(sum(float(t.get("total_estimated_savings", 0) or 0) for t in traces), 6)
    return {
        "total_requests": len(traces),
        "original_tokens": original,
        "optimized_tokens": optimized,
        "tokens_saved": saved,
        "savings_percent": savings_percent,
        "estimated_cost_saved": estimated_cost_saved,
        "cache_hit_rate": cache["hit_rate"],
        "quality_failures": quality_failures,
        "protected_failures": protected_failures,
        "cache": cache,
    }
