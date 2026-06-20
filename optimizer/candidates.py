from __future__ import annotations

from dataclasses import dataclass, asdict
import re

from .segmenter import Segment


DOMAIN_PHRASES = {"new york", "x-chromosome x-chromosome", "had had", "not not"}
DISCOURSE_WORDS = {
    "hello", "hi", "hey", "please", "kindly", "thanks", "thank", "you",
    "really", "very", "just", "basically", "actually", "bro", "okay", "ok",
}


@dataclass
class Token:
    text: str
    start: int
    end: int
    norm: str


@dataclass
class Candidate:
    start: int
    end: int
    span_text: str
    retained_span: str
    candidate_type: str
    reason: str
    exact_repetition_count: int = 1
    local_redundancy: float = 0.0
    global_redundancy: float = 0.0
    semantic_overlap: float = 0.0
    information_density: float = 1.0
    task_relevance: float = 0.5
    position_importance: float = 0.5
    named_entity_presence: bool = False
    fact_presence: bool = False
    protected_risk: bool = False
    grammar_risk: bool = False
    contradiction_risk: bool = False
    risk_flags: list[str] | None = None
    tokens_saved: int = 0
    score: float = 0.0
    confidence: float = 0.0
    utility_score: float = 0.0
    accepted: bool = False
    rejected_reason: str | None = None

    def to_trace(self) -> dict:
        data = asdict(self)
        data["backend"] = "information_score_backend"
        data["removed_span"] = self.span_text if self.accepted else None
        data["risk_flags"] = self.risk_flags or []
        return data


class CandidateGenerator:
    def generate(self, segment: Segment) -> list[Candidate]:
        tokens = tokenize(segment.text, segment.start)
        candidates: list[Candidate] = []
        candidates.extend(self._repeated_tokens(tokens))
        candidates.extend(self._repeated_discourse_tokens(tokens))
        # Arbitrary repeated n-gram deletion is too risky for production prose:
        # repeated noun/verb phrases often carry grammar, entities, or relations.
        # Higher-level semantic passes handle meaningful phrase compression.
        candidates.extend(self._repeated_relation_diagnostics(tokens))
        candidates.extend(self._repeated_sentences(segment))
        candidates.extend(self._repeated_paragraphs(segment))
        candidates.extend(self._trailing_low_info(tokens))
        return dedupe_candidates(candidates)

    def _repeated_tokens(self, tokens: list[Token]) -> list[Candidate]:
        candidates: list[Candidate] = []
        i = 0
        while i < len(tokens):
            j = i + 1
            while j < len(tokens) and tokens[j].norm == tokens[i].norm:
                j += 1
            run = j - i
            if run > 1:
                for token in tokens[i + 1:j]:
                    candidates.append(Candidate(
                        token.start, token.end, token.text, tokens[i].text,
                        "repeated_token", "consecutive repeated token has retained copy", run,
                    ))
            i = j
        return candidates

    def _repeated_discourse_tokens(self, tokens: list[Token]) -> list[Candidate]:
        candidates: list[Candidate] = []
        first_seen: dict[str, Token] = {}
        for token in tokens:
            if token.norm not in DISCOURSE_WORDS:
                continue
            retained = first_seen.get(token.norm)
            if retained is None:
                first_seen[token.norm] = token
                continue
            candidates.append(Candidate(
                token.start, token.end, token.text, retained.text,
                "low_information_filler", "low-information discourse token already appears earlier", 2,
            ))
        return candidates

    def _repeated_ngrams(self, tokens: list[Token]) -> list[Candidate]:
        candidates: list[Candidate] = []
        seen: dict[tuple[str, ...], tuple[int, int, str]] = {}
        max_n = min(10, max(2, len(tokens) // 2))
        for n in range(max_n, 1, -1):
            for i in range(0, len(tokens) - n + 1):
                window = tokens[i:i + n]
                key = tuple(token.norm for token in window)
                norm = " ".join(key)
                if any(token.text.startswith("__PROTECTED_") for token in window) or is_domain_phrase(norm):
                    continue
                if "dup" in key and "chunk" in key:
                    continue
                span_text = span_text_from_tokens(window)
                if key in seen:
                    _, prior_end, retained_text = seen[key]
                    if window[0].start < prior_end:
                        continue
                    candidates.append(Candidate(
                        window[0].start, window[-1].end, span_text, retained_text,
                        "repeated_ngram", f"repeated {n}-token span has retained copy", 2,
                    ))
                else:
                    seen[key] = (window[0].start, window[-1].end, span_text)
        return candidates

    def _repeated_relation_diagnostics(self, tokens: list[Token]) -> list[Candidate]:
        candidates: list[Candidate] = []
        seen: dict[tuple[str, str], str] = {}
        for i in range(len(tokens) - 1):
            key = (tokens[i].norm, tokens[i + 1].norm)
            if key not in {("better", "than"), ("worse", "than"), ("easier", "than"), ("harder", "than")}:
                continue
            span = span_text_from_tokens(tokens[i:i + 2])
            if key in seen:
                candidates.append(Candidate(
                    tokens[i].start, tokens[i + 1].end, span, seen[key],
                    "repeated_ngram", "semantic relation phrase observed but protected from deletion", 2,
                ))
            else:
                seen[key] = span
        return candidates

    def _repeated_sentences(self, segment: Segment) -> list[Candidate]:
        candidates: list[Candidate] = []
        seen: dict[str, tuple[int, int, str]] = {}
        for match in re.finditer(r"[^\s\n].*?(?:[.!?](?=\s+|\s*$)|$)", segment.text):
            key = normalize(match.group(0))
            if len(key.split()) < 3:
                continue
            start = segment.start + match.start()
            end = segment.start + match.end()
            if key in seen:
                _, _, retained = seen[key]
                candidates.append(Candidate(start, end, match.group(0), retained, "repeated_sentence", "exact repeated sentence has retained copy", 2))
            else:
                seen[key] = (start, end, match.group(0))
        return candidates

    def _repeated_paragraphs(self, segment: Segment) -> list[Candidate]:
        candidates: list[Candidate] = []
        seen: dict[str, tuple[int, int, str]] = {}
        offset = 0
        for paragraph in re.split(r"(\n\s*\n)", segment.text):
            if not paragraph or paragraph.isspace():
                offset += len(paragraph)
                continue
            key = normalize(paragraph)
            start = segment.start + offset
            end = start + len(paragraph)
            if len(key.split()) >= 6:
                if key in seen:
                    _, _, retained = seen[key]
                    candidates.append(Candidate(start, end, paragraph, retained, "repeated_paragraph", "exact repeated paragraph has retained copy", 2))
                else:
                    seen[key] = (start, end, paragraph)
            offset += len(paragraph)
        return candidates

    def _trailing_low_info(self, tokens: list[Token]) -> list[Candidate]:
        if len(tokens) < 2:
            return []
        last = tokens[-1]
        prior_norms = {token.norm for token in tokens[:-1]}
        if last.norm in DISCOURSE_WORDS and last.norm in prior_norms:
            return [Candidate(last.start, last.end, last.text, last.text, "trailing_low_information", "trailing low-information token already appears earlier", 2)]
        return []


def tokenize(text: str, base: int = 0) -> list[Token]:
    tokens: list[Token] = []
    for match in re.finditer(r"__PROTECTED_\d+__|[A-Za-z0-9][A-Za-z0-9'-]*", text):
        value = match.group(0)
        tokens.append(Token(value, base + match.start(), base + match.end(), normalize_token(value)))
    return tokens


def normalize(text: str) -> str:
    return " ".join(normalize_token(token) for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", text))


def normalize_token(token: str) -> str:
    return token.strip(".,!?;:()[]{}\"'").lower()


def span_text_from_tokens(tokens: list[Token]) -> str:
    return " ".join(token.text for token in tokens)


def is_domain_phrase(norm: str) -> bool:
    return norm in DOMAIN_PHRASES or f"{norm} {norm}" in DOMAIN_PHRASES


def dedupe_candidates(candidates: list[Candidate]) -> list[Candidate]:
    by_key: dict[tuple[int, int, str], Candidate] = {}
    for candidate in candidates:
        key = (candidate.start, candidate.end, candidate.candidate_type)
        if key not in by_key or len(candidate.span_text) > len(by_key[key].span_text):
            by_key[key] = candidate
    return list(by_key.values())
