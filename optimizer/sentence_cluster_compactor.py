from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re

from .protect import extract_sensitive_facts
from .safety import GrammarSafety
from .semantic_validator import SemanticValidator
from .token_counter import count_tokens


STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "being", "been", "it",
    "this", "that", "because", "since", "really", "very", "extremely", "quite",
    "maybe", "probably", "possibly", "for", "some", "reason", "thing", "stuff",
    "more", "much", "less", "than", "compared", "to", "with", "and", "or", "but",
    "our", "without", "can", "could", "would", "should", "may", "might",
}

SYNONYMS = {
    "quick": "fast",
    "faster": "fast",
    "speedy": "fast",
    "stable": "reliable",
    "dependable": "reliable",
    "simple": "easy",
    "easier": "easy",
    "difficult": "hard",
    "difficulty": "easy",
    "harder": "hard",
    "costly": "expensive",
    "affordable": "cheap",
    "cheaper": "cheap",
    "supports": "can",
    "allows": "can",
    "requires": "must",
    "required": "must",
    "needs": "must",
    "should": "must",
}


@dataclass
class SentenceItem:
    text: str
    start: int
    end: int
    fingerprint: set[str]


class SemanticSentenceClusterCompactor:
    """General semantic redundancy compressor.

    Clusters semantically overlapping full sentences and keeps the best existing
    sentence. It avoids free rewriting, protected fact changes, and obvious
    contradictions.
    """

    def __init__(self) -> None:
        self.validator = SemanticValidator()
        self.grammar = GrammarSafety()

    def compact(self, text: str) -> tuple[str, list[dict]]:
        sentences = self._sentences(text)
        if len(sentences) < 2:
            return text, []
        groups = self._groups(sentences)
        traces: list[dict] = []
        replacements: list[tuple[int, int, str, list[SentenceItem], float]] = []
        occupied: list[tuple[int, int]] = []
        for group, score in groups:
            if len(group) < 2:
                continue
            start = min(item.start for item in group)
            end = max(item.end for item in group)
            if any(start < used_end and used_start < end for used_start, used_end in occupied):
                traces.append(self._trace(group, group[0].text, score, False, "overlaps stronger semantic cluster"))
                continue
            if self._unsafe_group(group):
                traces.append(self._trace(group, group[0].text, score, False, "protected facts or contradiction risk differ"))
                continue
            canonical = self._canonical(group)
            trial = text[:start] + canonical + text[end:]
            grammar = self.grammar.validate(trial)
            if not grammar["grammar_validity"]:
                traces.append(self._trace(group, canonical, score, False, "grammar validation rejected"))
                continue
            replacements.append((start, end, canonical, group, score))
            occupied.append((start, end))
            traces.append(self._trace(group, canonical, score, True, "semantic sentence cluster compacted"))

        if not replacements:
            return text, traces
        optimized = text
        for start, end, canonical, _, _ in sorted(replacements, reverse=True):
            optimized = optimized[:start] + canonical + optimized[end:]
        optimized = re.sub(r"\s{2,}", " ", optimized).strip()
        if count_tokens(optimized) >= count_tokens(text):
            for trace in traces:
                if trace["accepted"]:
                    trace["accepted"] = False
                    trace["rejected_reason"] = "final output was not shorter"
            return text, traces
        return optimized, traces

    def _sentences(self, text: str) -> list[SentenceItem]:
        items: list[SentenceItem] = []
        for match in re.finditer(r"[^\s\n].*?(?:[.!?](?=\s+|\s*$)|$)", text):
            sentence = match.group(0).strip()
            if "__PROTECTED_" in sentence or len(sentence.split()) < 4:
                continue
            items.append(SentenceItem(sentence, match.start(), match.end(), _fingerprint(sentence)))
        return items

    def _groups(self, sentences: list[SentenceItem]) -> list[tuple[list[SentenceItem], float]]:
        used: set[int] = set()
        groups: list[tuple[list[SentenceItem], float]] = []
        for i, sentence in enumerate(sentences):
            if i in used:
                continue
            group = [sentence]
            scores: list[float] = []
            for j in range(i + 1, len(sentences)):
                if j in used:
                    continue
                score = self._similarity(sentence, sentences[j])
                if score >= 0.68:
                    group.append(sentences[j])
                    scores.append(score)
                    used.add(j)
            if len(group) > 1:
                used.add(i)
                groups.append((group, round(sum(scores) / max(1, len(scores)), 3)))
        return sorted(groups, key=lambda pair: (len(pair[0]), pair[1]), reverse=True)

    def _similarity(self, a: SentenceItem, b: SentenceItem) -> float:
        if not a.fingerprint or not b.fingerprint:
            return 0.0
        overlap = len(a.fingerprint & b.fingerprint) / max(1, min(len(a.fingerprint), len(b.fingerprint)))
        jaccard = len(a.fingerprint & b.fingerprint) / max(1, len(a.fingerprint | b.fingerprint))
        sequence = SequenceMatcher(None, _normalized(a.text), _normalized(b.text)).ratio()
        model_score = self.validator.similarity(a.text, b.text)["semantic_similarity"]
        return max(overlap * 0.82, jaccard, sequence * 0.72, model_score if not self.validator.embedding_backend == "lexical" else 0)

    def _unsafe_group(self, group: list[SentenceItem]) -> bool:
        base_facts = _protected_fact_signature(group[0].text)
        base_entities = _entity_signature(group[0].text)
        base_polarity = _polarity(group[0].text)
        base_content = _content_signature(group[0].text)
        base_modality = _modality(group[0].text)
        for item in group[1:]:
            if _protected_fact_signature(item.text) != base_facts:
                return True
            if _entity_signature(item.text) != base_entities:
                return True
            if _polarity(item.text) != base_polarity:
                return True
            if _modality(item.text) != base_modality:
                return True
            item_content = _content_signature(item.text)
            if base_content and item_content:
                overlap = len(base_content & item_content) / max(1, min(len(base_content), len(item_content)))
                if overlap < 0.67:
                    return True
        return False

    def _canonical(self, group: list[SentenceItem]) -> str:
        return min((item.text for item in group), key=lambda text: (count_tokens(text), _filler_count(text), len(text)))

    def _trace(self, group: list[SentenceItem], canonical: str, score: float, accepted: bool, reason: str) -> dict:
        span = " ".join(item.text for item in group)
        return {
            "backend": "semantic_sentence_cluster",
            "candidate_type": "semantic_duplicate_sentence_cluster",
            "reason": reason,
            "span_text": span,
            "removed_span": " ".join(item.text for item in group if item.text != canonical) if accepted else None,
            "retained_span": canonical,
            "score": round(score, 3),
            "confidence": round(score, 3),
            "semantic_similarity": round(score, 3),
            "tokens_saved": max(0, count_tokens(span) - count_tokens(canonical)),
            "risk_flags": [] if accepted else ["semantic_cluster_rejected"],
            "accepted": accepted,
            "rejected_reason": None if accepted else reason,
        }


def _fingerprint(sentence: str) -> set[str]:
    sentence = _analysis_text(sentence)
    words = []
    for raw in re.findall(r"[A-Za-z0-9'-]+", sentence.lower()):
        if raw in STOPWORDS:
            continue
        word = SYNONYMS.get(raw, raw)
        if word.endswith("ing") and len(word) > 5:
            word = word[:-3]
        elif word.endswith("ed") and len(word) > 4:
            word = word[:-2]
        elif word.endswith("s") and len(word) > 4:
            word = word[:-1]
        words.append(word)
    return set(words)


def _normalized(sentence: str) -> str:
    return " ".join(sorted(_fingerprint(sentence)))


def _protected_fact_signature(text: str) -> tuple:
    facts = extract_sensitive_facts(text)
    return tuple(sorted((key, tuple(sorted(value))) for key, value in facts.items() if key in {"id", "date", "money", "percentage", "url", "email", "number"} and value))


def _entity_signature(text: str) -> tuple[str, ...]:
    text = _analysis_text(text)
    ignored = {"The", "A", "An", "Our", "Your", "Their", "This", "That", "New", "Knowledge", "Base", "Engineering", "Documentation", "Product", "Overview", "User", "Guide", "Training", "Material", "FAQ", "Question"}
    entities = []
    for value in re.findall(r"\b[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*){0,3}\b", text):
        if value in ignored:
            continue
        entities.append(value)
    return tuple(sorted(entities))


def _content_signature(text: str) -> set[str]:
    lowered = text.lower()
    for entity in _entity_signature(text):
        lowered = lowered.replace(entity.lower(), " ")
    return _fingerprint(lowered)


def _polarity(text: str) -> str:
    low = text.lower()
    if re.search(r"\b(not|never|cannot|can't|must not|prohibited|denied|unreliable|harder|worse|slower)\b", low):
        return "negative"
    return "positive"


def _modality(text: str) -> str:
    low = text.lower()
    if re.search(r"\b(will|must|shall|definitely|certainly)\b", low):
        return "certain"
    if re.search(r"\b(might|may|could|possible|possibly|potentially)\b", low):
        return "uncertain"
    return "neutral"


def _filler_count(text: str) -> int:
    return len(re.findall(r"\b(very|really|extremely|maybe|probably|possibly|basically|actually|for some reason)\b", text, flags=re.IGNORECASE))


def _analysis_text(text: str) -> str:
    return re.sub(r"(?m)^\s*[A-Z][A-Za-z &/-]{1,50}:\s*", " ", text)
