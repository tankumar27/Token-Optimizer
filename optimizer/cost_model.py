from __future__ import annotations

import os


DEFAULT_PRICES = {
    "dry-run": {"input_per_1k": 0.0, "output_per_1k": 0.0},
    # Gemini 2.5 Flash paid tier defaults, expressed per 1K tokens.
    # Override with env vars when Google pricing or billing tier changes.
    "gemini-flash": {"input_per_1k": float(os.getenv("GEMINI_FLASH_INPUT_PER_1K", "0.0003")), "output_per_1k": float(os.getenv("GEMINI_FLASH_OUTPUT_PER_1K", "0.0025"))},
    "openai-small": {"input_per_1k": float(os.getenv("OPENAI_SMALL_INPUT_PER_1K", "0.00015")), "output_per_1k": float(os.getenv("OPENAI_SMALL_OUTPUT_PER_1K", "0.0006"))},
    "openai-strong": {"input_per_1k": float(os.getenv("OPENAI_STRONG_INPUT_PER_1K", "0.005")), "output_per_1k": float(os.getenv("OPENAI_STRONG_OUTPUT_PER_1K", "0.015"))},
}


def estimate_cost(provider: str, model: str | None, input_tokens: int, output_tokens: int = 0) -> dict:
    key = _price_key(provider, model)
    price = DEFAULT_PRICES[key]
    input_cost = input_tokens / 1000 * price["input_per_1k"]
    output_cost = output_tokens / 1000 * price["output_per_1k"]
    return {
        "provider_price_key": key,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_input_cost": round(input_cost, 6),
        "estimated_output_cost": round(output_cost, 6),
        "estimated_total_cost": round(input_cost + output_cost, 6),
    }


def savings_report(provider: str, model: str | None, before_tokens: int, after_tokens: int, cache_hit: bool = False) -> dict:
    before = estimate_cost(provider, model, before_tokens)
    after = estimate_cost(provider, model, after_tokens)
    compression_savings = max(0.0, before["estimated_total_cost"] - after["estimated_total_cost"])
    cache_savings = after["estimated_total_cost"] if cache_hit else 0.0
    return {
        "estimated_cost_before": before["estimated_total_cost"],
        "estimated_cost_after": after["estimated_total_cost"],
        "estimated_input_cost_before": before["estimated_input_cost"],
        "estimated_input_cost_after": after["estimated_input_cost"],
        "estimated_output_cost": 0.0,
        "estimated_original_total_with_output": before["estimated_total_cost"],
        "estimated_optimized_total_with_output": after["estimated_total_cost"],
        "prompt_compression_savings": round(compression_savings, 6),
        "cache_savings": round(cache_savings, 6),
        "routing_savings": 0.0,
        "total_estimated_savings": round(compression_savings + cache_savings, 6),
        "pricing_basis": "estimated input tokens only",
        "provider_price_key": before["provider_price_key"],
    }


def live_cost_report(
    provider: str,
    model: str | None,
    original_input_tokens: int,
    optimized_input_tokens: int,
    output_tokens: int,
    provider_prompt_tokens: int | None = None,
) -> dict:
    original = estimate_cost(provider, model, original_input_tokens, output_tokens)
    optimized = estimate_cost(provider, model, optimized_input_tokens, output_tokens)
    live_prompt = estimate_cost(provider, model, provider_prompt_tokens or optimized_input_tokens, output_tokens)
    compression_savings = max(0.0, original["estimated_total_cost"] - optimized["estimated_total_cost"])
    return {
        "estimated_cost_before": original["estimated_total_cost"],
        "estimated_cost_after": optimized["estimated_total_cost"],
        "estimated_input_cost_before": original["estimated_input_cost"],
        "estimated_input_cost_after": optimized["estimated_input_cost"],
        "estimated_output_cost": optimized["estimated_output_cost"],
        "estimated_original_total_with_output": original["estimated_total_cost"],
        "estimated_optimized_total_with_output": optimized["estimated_total_cost"],
        "actual_provider_prompt_tokens": provider_prompt_tokens or optimized_input_tokens,
        "actual_provider_output_tokens": output_tokens,
        "actual_provider_estimated_total": live_prompt["estimated_total_cost"],
        "prompt_compression_savings": round(compression_savings, 6),
        "cache_savings": 0.0,
        "routing_savings": 0.0,
        "total_estimated_savings": round(compression_savings, 6),
        "pricing_basis": "live provider output tokens plus middleware input tokens",
        "provider_price_key": original["provider_price_key"],
    }


def _price_key(provider: str, model: str | None) -> str:
    if provider == "dry-run":
        return "dry-run"
    if provider == "openai":
        if model and any(word in model.lower() for word in ["strong", "gpt-4", "o3"]):
            return "openai-strong"
        return "openai-small"
    return "gemini-flash"
