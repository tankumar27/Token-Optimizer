from __future__ import annotations

from .prompt_type import PromptAnalysis


def plan_strategies(analysis: PromptAnalysis, provider: str, compression_level: str) -> dict:
    strategies: list[str] = ["exact_cache"]
    if analysis.cacheability != "low":
        strategies.append("semantic_cache")

    if analysis.prompt_type == "customer_support":
        strategies.extend(["customer_support_policy_dedupe", "instruction_dedupe"])
    elif analysis.prompt_type == "rag_context":
        strategies.extend(["rag_dedupe", "chunk_relevance_pruning"])
    elif analysis.prompt_type == "policy_instruction":
        strategies.append("instruction_dedupe")
    elif analysis.prompt_type == "chat_history":
        strategies.append("chat_history_compaction")
    elif analysis.prompt_type == "agent_tool_trace":
        strategies.append("tool_trace_cleanup")

    if analysis.risk_level == "low" and compression_level in {"balanced", "aggressive"}:
        strategies.append("semantic_compression")
    elif analysis.risk_level == "high":
        strategies.append("no_generated_summary")

    if analysis.input_tokens < 120 and analysis.repeated_instruction_ratio < 0.2:
        strategies = ["exact_cache", "no_change"]

    expected_savings = _expected_savings(analysis)
    return {
        "prompt_type": analysis.prompt_type,
        "risk_level": analysis.risk_level,
        "chosen_strategies": strategies,
        "expected_savings_percent": expected_savings,
        "expected_risk": "low" if analysis.risk_level != "high" else "medium",
        "reason": _reason(analysis, strategies),
        "provider": provider,
    }


def _expected_savings(analysis: PromptAnalysis) -> float:
    if analysis.prompt_type == "customer_support":
        return 25.0 if analysis.input_tokens >= 180 else 12.0
    if analysis.prompt_type == "rag_context":
        return 30.0 if analysis.rag_chunk_count >= 4 else 15.0
    if analysis.prompt_type == "policy_instruction":
        return 18.0
    if analysis.prompt_type in {"legal_compliance", "general_prompt"}:
        return 5.0
    return 10.0


def _reason(analysis: PromptAnalysis, strategies: list[str]) -> str:
    if "no_change" in strategies:
        return "tiny prompt with low repetition; avoid processing overhead"
    if analysis.prompt_type == "customer_support":
        return "support prompt contains repeated policy requirements and protected ticket facts"
    if analysis.prompt_type == "rag_context":
        return "retrieved context can contain duplicate chunks and overlapping evidence"
    if analysis.risk_level == "high":
        return "high-risk content; use conservative cleanup only"
    return "standard optimization strategy selected from prompt analysis"
