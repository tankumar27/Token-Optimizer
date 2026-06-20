from __future__ import annotations

from .candidates import Candidate, DISCOURSE_WORDS, normalize
from .token_counter import count_tokens


class CandidateScorer:
    def score(self, candidate: Candidate, text: str) -> Candidate:
        norm = normalize(candidate.span_text)
        words = norm.split()
        unique_words = set(words)
        candidate.tokens_saved = count_tokens(candidate.span_text)
        candidate.local_redundancy = 1.0 if candidate.candidate_type in {"repeated_token", "trailing_low_information"} else 0.75
        candidate.global_redundancy = min(1.0, max(0.0, normalize(text).count(norm) - 1))
        candidate.semantic_overlap = 1.0 if normalize(candidate.retained_span) == norm else 0.6
        candidate.information_density = len([w for w in unique_words if w not in DISCOURSE_WORDS]) / max(1, len(unique_words))
        candidate.position_importance = 0.8 if candidate.start < max(1, len(text) * 0.15) else 0.35
        candidate.task_relevance = 0.25 if candidate.information_density < 0.5 else 0.6
        risk_penalty = 0.22 * len(candidate.risk_flags or [])
        base = (
            0.2 * candidate.local_redundancy
            + 0.23 * candidate.global_redundancy
            + 0.18 * candidate.semantic_overlap
            + 0.14 * (1 - candidate.information_density)
            + 0.12 * min(1.0, candidate.tokens_saved / 8)
            + 0.08 * (1 - candidate.task_relevance)
            + 0.05 * (1 - candidate.position_importance)
        )
        if candidate.candidate_type in {"repeated_ngram", "repeated_sentence", "repeated_paragraph"}:
            base += 0.14
        candidate.confidence = round(max(0.0, min(0.99, base - risk_penalty)), 3)
        candidate.score = candidate.confidence
        candidate.utility_score = round(candidate.confidence * candidate.tokens_saved - risk_penalty, 3)
        return candidate
