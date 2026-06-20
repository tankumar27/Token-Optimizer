from __future__ import annotations

import re

from .candidates import Candidate, normalize
from .safety import GrammarSafety
from .semantic_validator import SemanticValidator
from .token_counter import count_tokens


QUESTION_STARTERS = {"what", "why", "how", "when", "where", "who"}


class CompressionPlanner:
    def __init__(self, level: str, validator: SemanticValidator | None = None) -> None:
        self.level = level
        self.max_ratio = {"safe": 0.35, "balanced": 0.55, "aggressive": 0.7}.get(level, 0.35)
        self.threshold = {"safe": 0.58, "balanced": 0.45, "aggressive": 0.35}.get(level, 0.58)
        self.grammar = GrammarSafety()
        self.validator = validator or SemanticValidator()

    def select(self, candidates: list[Candidate], text: str) -> list[Candidate]:
        selected: list[Candidate] = []
        occupied: list[tuple[int, int]] = []
        token_budget = max(1, int(count_tokens(text) * self.max_ratio))
        saved = 0
        for candidate in sorted(candidates, key=lambda c: (c.utility_score, c.tokens_saved), reverse=True):
            if candidate.risk_flags:
                candidate.rejected_reason = candidate.rejected_reason or "safety gate rejected"
                continue
            if candidate.confidence < self.threshold:
                candidate.rejected_reason = "below confidence threshold"
                continue
            if any(_overlaps((candidate.start, candidate.end), used) for used in occupied):
                candidate.rejected_reason = "overlaps higher-utility candidate"
                continue
            if saved + candidate.tokens_saved > token_budget:
                candidate.rejected_reason = "max compression ratio reached"
                continue
            trial = self.apply(text, selected + [candidate])
            grammar = self.grammar.validate(trial)
            if not grammar["grammar_validity"]:
                candidate.grammar_risk = True
                candidate.risk_flags = (candidate.risk_flags or []) + grammar["grammar_flags"]
                candidate.rejected_reason = "grammar safety rejected"
                continue
            candidate.accepted = True
            selected.append(candidate)
            occupied.append((candidate.start, candidate.end))
            saved += candidate.tokens_saved
        return selected

    def apply(self, text: str, selected: list[Candidate]) -> str:
        result = text
        for candidate in sorted(selected, key=lambda c: c.start, reverse=True):
            start, end = _expand_removal_bounds(result, candidate.start, candidate.end)
            result = result[:start] + result[end:]
        return clean_text(result)


class TraceRecorder:
    def record(self, candidates: list[Candidate]) -> list[dict]:
        return [candidate.to_trace() for candidate in sorted(candidates, key=lambda c: (c.start, c.end, c.candidate_type))]


def clean_text(text: str) -> str:
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"([.!?])\s+(__PROTECTED_\d+__)", r"\1\n\2", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"([.!?]){2,}", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    stripped = text.strip(" ,;:")
    final_clause = re.split(r"[.!?]\s+", stripped)[-1]
    words = normalize(final_clause).split()
    if words and any(word in QUESTION_STARTERS for word in words[:3]) and not stripped.endswith(("?", ".", "!")):
        stripped += "?"
    return stripped


def _expand_removal_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    while end < len(text) and text[end] in " \t":
        end += 1
    if start > 0 and text[start - 1] in ",;:":
        start -= 1
        while start > 0 and text[start - 1] == " ":
            start -= 1
    return start, end


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]
