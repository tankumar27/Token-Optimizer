from __future__ import annotations

import re

from .candidates import Candidate, normalize, is_domain_phrase
from .semantic_validator import SemanticValidator


ORPHAN_STARTERS = {"is", "are", "was", "were", "has", "have", "do", "does", "did"}
ORPHAN_ENDERS = {"is", "are", "was", "were", "has", "have"}
DANGLING_WORDS = {"and", "but", "or", "because", "while", "with", "for", "to", "by", "of", "in", "on"}
CONTRADICTION_WORDS = {"must", "must not", "may", "required", "prohibited", "allowed", "denied", "approved", "shall", "shall not", "before", "after"}
RELATION_PHRASES = re.compile(
    r"\b("
    r"better than|worse than|more than|less than|greater than|fewer than|"
    r"is better|is worse|is easier|is harder|are better|are worse|"
    r"better|worse|easier|harder|superior|inferior|"
    r"equal to|same as|different from|not better|not worse|"
    r"plays? for|represents?|responded in|completed in|remained below|"
    r"proven effective|evidence that|causes?|helps?|reduces?|lowers?|increases?|"
    r"requires?|required|must|should|shall|may|might|could|will|cannot|can't|not"
    r")\b",
    re.IGNORECASE,
)


class SafetyGate:
    def __init__(self, validator: SemanticValidator | None = None) -> None:
        self.validator = validator or SemanticValidator()

    def check(self, candidate: Candidate, full_text: str) -> Candidate:
        flags: list[str] = []
        span = candidate.span_text
        norm = normalize(span)
        retained_norm = normalize(candidate.retained_span)
        duplicate_retained = bool(retained_norm and retained_norm == norm and normalize(full_text).count(norm) > 1)

        if re.search(r"__PROTECTED_\d+__", span):
            flags.append("protected_placeholder")
        if re.search(r"https?://|\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b", span):
            flags.append("url_or_email")
        if re.search(r"\$\d|\b\d{4}-\d{2}-\d{2}\b|\b\d+(?:\.\d+)?%|\b\d+(?:\.\d+)?\b", span):
            flags.append("number_date_money")
        if re.search(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)+\b|\bClause \d+(?:\.\d+)*\b", span):
            flags.append("id_or_legal_clause")
        if re.search(r"```|`[^`]+`|\{.*\}|\[.*\]|<[^>]+>", span, flags=re.DOTALL):
            flags.append("code_or_structured_data")
        if re.search(r"[a-zA-Z]\s*=\s*[-+*/^(). 0-9a-zA-Z]+", span):
            flags.append("math")
        if re.search(r"\"[^\"]+\"|'[^']+'", span):
            flags.append("quoted_exact_text")
        if is_domain_phrase(norm):
            flags.append("domain_term")
        if _has_named_entity(span, self.validator) and not duplicate_retained:
            flags.append("named_entity_without_identical_retained_copy")
        if _looks_contradiction_sensitive(span) and not duplicate_retained:
            flags.append("contradiction_sensitive")
        if candidate.candidate_type in {"repeated_ngram", "repeated_phrase", "low_information_filler"} and RELATION_PHRASES.search(span):
            flags.append("semantic_relation_phrase")
        if candidate.candidate_type == "repeated_ngram" and _looks_relation_bearing_partial(span):
            flags.append("relation_bearing_partial_span")
        if candidate.candidate_type == "repeated_ngram" and _looks_grammar_sensitive_partial(span):
            flags.append("grammar_sensitive_partial_span")
        if candidate.candidate_type == "repeated_ngram" and _looks_required_object_span(span, full_text, candidate.start, candidate.end):
            flags.append("required_object_phrase")
        if candidate.candidate_type == "repeated_ngram" and _looks_object_after_relation_span(full_text, candidate.start):
            flags.append("object_after_relation_phrase")

        candidate.named_entity_presence = _has_named_entity(span, self.validator)
        candidate.fact_presence = any(flag in flags for flag in ["number_date_money", "id_or_legal_clause", "url_or_email"])
        candidate.protected_risk = "protected_placeholder" in flags
        candidate.contradiction_risk = "contradiction_sensitive" in flags
        candidate.risk_flags = flags
        if flags:
            candidate.accepted = False
            candidate.rejected_reason = "risk_flags: " + ", ".join(flags)
        return candidate


class GrammarSafety:
    def validate(self, text: str) -> dict:
        flags: list[str] = []
        if "  " in text:
            flags.append("double_spaces")
        text = re.sub(r"```[\s\S]*?```", " ", text)
        text = re.sub(r"__PROTECTED_\d+__", "VALUE", text)
        if re.search(r"\s+[,.!?;:]", text):
            flags.append("broken_punctuation")
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
            raw_sentence = sentence.strip()
            if not raw_sentence:
                continue
            if raw_sentence.startswith(("-", "*", "•")):
                continue
            if raw_sentence.endswith(":"):
                continue
            if re.match(r"^[A-Z][A-Za-z ]{1,40}:\s+\S+", raw_sentence):
                continue
            words = re.findall(r"[A-Za-z']+", sentence.lower())
            if not words:
                continue
            if words[0] in ORPHAN_STARTERS and not (words[0] == "do" and len(words) > 1 and words[1] == "not"):
                flags.append("orphan_verb_start")
            if words[0] in {"can", "could", "would", "should", "may", "might"}:
                flags.append("missing_subject_before_modal")
            if len(words) > 1 and words[0] in {"the", "a", "an"} and words[1] in ORPHAN_STARTERS:
                flags.append("missing_subject_before_verb")
            starts_valid_by_gerund = len(words) > 1 and words[0] == "by" and words[1].endswith("ing")
            dangling_start = words[0] in DANGLING_WORDS and not starts_valid_by_gerund
            dangling_end = words[-1] in DANGLING_WORDS and not _ends_with_nonword_object(raw_sentence)
            if dangling_start or dangling_end:
                flags.append("dangling_connector_or_preposition")
            if len(words) > 1 and words[-1] in ORPHAN_ENDERS and not _ends_with_nonword_object(raw_sentence):
                flags.append("orphan_verb_end")
            if len(words) == 1 and words[0] not in {"hi", "hello", "thanks", "ok", "yes", "no"}:
                flags.append("empty_or_fragment_sentence")
        return {
            "grammar_validity": not flags,
            "grammar_flags": sorted(set(flags)),
            "validator_used": "rules",
            "fallback_used": True,
        }


def _ends_with_nonword_object(sentence: str) -> bool:
    """Allow prepositions whose object is a protected factual token.

    The fallback grammar checker tokenizes only alphabetic words. Without this
    guard, valid clauses like "opened on 2026-06-25" or "decreased to 1.6%"
    look as if they end with "on" or "to".
    """
    alpha_matches = list(re.finditer(r"[A-Za-z']+", sentence))
    if not alpha_matches:
        return False
    tail = sentence[alpha_matches[-1].end():].strip()
    if not tail:
        return False
    return bool(
        re.search(r"\d", tail)
        or re.search(r"[%$]", tail)
        or re.search(r"__PROTECTED_\d+__", tail)
        or re.search(r"https?://|[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}", tail)
        or re.search(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)+\b", tail)
    )


def _has_named_entity(text: str, validator: SemanticValidator) -> bool:
    if validator.entities(text):
        return True
    if re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text):
        return True
    if re.search(r"\bmy name is\s+[a-z][a-z'-]{2,}\b", text, flags=re.IGNORECASE):
        return True
    return False


def _looks_contradiction_sensitive(text: str) -> bool:
    low = text.lower()
    return any(word in low for word in CONTRADICTION_WORDS)


def _looks_grammar_sensitive_partial(text: str) -> bool:
    words = re.findall(r"[A-Za-z']+", text)
    if not words:
        return False
    if words[-1].lower() in ORPHAN_STARTERS:
        return True
    if any(word.isupper() and len(word) > 1 for word in words):
        return True
    title_words = [word for word in words if word[:1].isupper()]
    return len(title_words) >= 2


def _looks_relation_bearing_partial(text: str) -> bool:
    words = [word.lower() for word in re.findall(r"[A-Za-z']+", text)]
    if not words:
        return False
    if words[0] in {"the", "a", "an", "this", "that", "these", "those"}:
        return True
    relation_words = {
        "is", "are", "was", "were", "has", "have", "had", "be", "been",
        "for", "to", "from", "with", "by", "of", "in", "on", "than",
        "plays", "play", "represents", "represent", "proven", "evidence",
        "need", "needs", "required", "require", "requires", "cause", "causes",
        "reduce", "reduces", "lower", "lowers", "increase", "increases",
    }
    return any(word in relation_words for word in words)


def _looks_required_object_span(text: str, full_text: str, start: int, end: int) -> bool:
    words = re.findall(r"[A-Za-z']+", text)
    if not words or words[0].lower() not in {"the", "a", "an", "this", "that"}:
        return False
    sentence_start = max(full_text.rfind(".", 0, start), full_text.rfind("!", 0, start), full_text.rfind("?", 0, start), full_text.rfind("\n", 0, start)) + 1
    before = full_text[sentence_start:start].strip()
    after = full_text[end: full_text.find(".", end) if "." in full_text[end:] else len(full_text)].strip()
    before_words = re.findall(r"[A-Za-z']+", before.lower())
    if not before_words:
        return False
    if before_words[-1] in {"built", "tested", "used", "made", "created", "needs", "need", "requires", "require", "visited", "met", "called", "preserve", "verify"}:
        return True
    return not after


def _looks_object_after_relation_span(full_text: str, start: int) -> bool:
    sentence_start = max(full_text.rfind(".", 0, start), full_text.rfind("!", 0, start), full_text.rfind("?", 0, start), full_text.rfind("\n", 0, start)) + 1
    before = full_text[sentence_start:start].strip()
    before_words = re.findall(r"[A-Za-z']+", before.lower())
    if not before_words:
        return False
    relation_endings = {
        "need", "needs", "required", "require", "requires", "verify", "confirm", "check",
        "preserve", "remove", "delete", "process", "include", "contain", "costs", "cost",
        "reduces", "reduce", "lowers", "lower", "increases", "increase",
    }
    return any(word in relation_endings for word in before_words[-4:])
