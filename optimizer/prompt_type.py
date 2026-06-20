from __future__ import annotations

import re
from dataclasses import dataclass, asdict

from .token_counter import count_tokens


@dataclass(frozen=True)
class PromptAnalysis:
    prompt_type: str
    input_tokens: int
    risk_level: str
    repeated_instruction_ratio: float
    rag_chunk_count: int
    cacheability: str
    sensitivity_flags: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def detect_prompt_type(text: str) -> PromptAnalysis:
    low = text.lower()
    flags: list[str] = []
    if re.search(r"\b(order id|ticket id|customer email|product sku|refund amount)\b", low):
        flags.append("customer_identifiers")
    if re.search(r"\$\d|\brefunds?\b|\bfinance approval\b", low):
        flags.append("money_or_approval")
    if re.search(r"\b(clause|agreement|contract|liability|indemnif)\b", low):
        flags.append("legal")
    if re.search(r"```|\{[\s\S]*?:[\s\S]*?\}|<[^>]+>", text):
        flags.append("structured_or_code")

    chunk_count = len(re.findall(r"(?im)^(?:Retrieved .*?chunk|Chunk|Source|Document section|Policy excerpt)\s*[A-Z0-9]*:", text))
    repeated_ratio = _repeated_sentence_ratio(text)

    if "customer_identifiers" in flags and re.search(r"\b(refund|support|ticket|customer-facing)\b", low):
        prompt_type = "customer_support"
    elif chunk_count >= 2:
        prompt_type = "rag_context"
    elif "legal" in flags:
        prompt_type = "legal_compliance"
    elif re.search(r"\b(policy|must|should|do not|agents?)\b", low) and repeated_ratio > 0.12:
        prompt_type = "policy_instruction"
    elif re.search(r"\b(user:|assistant:|tool:|observation:)\b", low):
        prompt_type = "chat_history"
    elif re.search(r"\b(error|trace|log|tool call|stack)\b", low):
        prompt_type = "agent_tool_trace"
    elif re.search(r"\b(product overview|user guide|faq|documentation)\b", low):
        prompt_type = "product_docs"
    elif re.search(r"\b(finance|quarterly|spending|margin|revenue|cost)\b", low):
        prompt_type = "finance_report"
    else:
        prompt_type = "general_prompt"

    risk = "high" if {"legal", "structured_or_code"} & set(flags) else "medium" if flags else "low"
    cacheability = "low" if re.search(r"\b(today|current|latest|now|real-time)\b", low) else "medium"
    if prompt_type in {"customer_support", "rag_context", "policy_instruction", "product_docs"}:
        cacheability = "high" if cacheability != "low" else "low"

    return PromptAnalysis(
        prompt_type=prompt_type,
        input_tokens=count_tokens(text),
        risk_level=risk,
        repeated_instruction_ratio=repeated_ratio,
        rag_chunk_count=chunk_count,
        cacheability=cacheability,
        sensitivity_flags=tuple(flags),
    )


def _repeated_sentence_ratio(text: str) -> float:
    sentences = [
        _normalize(match.group(0))
        for match in re.finditer(r"[^\s\n].*?(?:[.!?](?=\s+|\s*$)|$)", text)
        if len(match.group(0).split()) >= 3
    ]
    if not sentences:
        return 0.0
    unique = set(sentences)
    return round((len(sentences) - len(unique)) / max(1, len(sentences)), 3)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9$%]+", " ", text.lower())).strip()
