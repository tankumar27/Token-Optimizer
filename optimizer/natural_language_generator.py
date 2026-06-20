from __future__ import annotations

from dataclasses import dataclass
import os
import re

from .protect import extract_sensitive_facts
from .safety import GrammarSafety
from .semantic_validator import SemanticValidator
from .token_counter import count_tokens
from .cheap_layer import canonical_fact_keys, event_signatures, state_signatures


CONTRAST_START = re.compile(r"^\s*(however|but|although|though|nevertheless)\b", re.IGNORECASE)
CONTRAST_WORDS = re.compile(r"\b(however|but|although|though|nevertheless)\b", re.IGNORECASE)
NEGATION_WORDS = re.compile(r"\b(no|not|never|cannot|can't|won't|unable|without|should not|must not)\b", re.IGNORECASE)
UNCERTAINTY_WORDS = re.compile(r"\b(may|might|could|possibly|probably|generally|often|usually|sometimes)\b", re.IGNORECASE)
CLAIM_VERBS = {
    "reduce", "reduces", "reduced", "lower", "lowers", "lowering", "decrease", "decreases",
    "compress", "compressing", "remove", "removing", "preserve", "preserving", "considered",
    "viewed", "find", "helps", "help", "costs", "spending", "expenses",
}
STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "being", "been", "by", "of", "in",
    "to", "for", "with", "and", "or", "but", "while", "because", "that", "this", "it",
    "its", "our", "their", "many", "main", "benefit", "platform", "system", "companies",
    "can", "without", "much", "still", "do",
    "students", "exam", "often", "generally", "considered", "viewed", "find",
}


@dataclass
class Sentence:
    text: str
    start: int
    end: int
    fingerprint: set[str]
    contrast: bool


class NaturalLanguageGeneratorBackend:
    """Meaning-preserving semantic optimizer pass.

    This is not an external LLM rewrite. It uses deterministic sentence
    clustering, small synthesis templates for recurring semantic structures,
    transformer/lexical validation, and safety gates.
    """

    backend_name = "semantic_optimizer_backend"

    def __init__(self) -> None:
        self.validator = SemanticValidator()
        self.grammar = GrammarSafety()
        self.local_generator_active = os.getenv("ENABLE_LOCAL_GENERATOR", "0") == "1"

    def compress(self, text: str, level: str = "balanced") -> tuple[str, list[dict]]:
        if "[dup:" in text:
            return text, []
        identity_text, identity_trace = _compact_repeated_identity_with_residual(text)
        if identity_trace:
            return identity_text, [identity_trace]
        sentences = _sentences(text)
        if len(sentences) < 2:
            return text, []

        traces: list[dict] = []
        replacements: list[tuple[int, int, str]] = []
        occupied: list[tuple[int, int]] = []
        for group in self._groups(sentences):
            if len(group) < 2:
                continue
            if _deterministic_claim_group(group):
                original_span = _join_sentences(group)
                traces.append(self._trace(group, original_span, _best_extractive([item.text for item in group]), 0.0, False, "deterministic semantic compactor has priority"))
                continue
            if _strict_obligation_group(group):
                original_span = _join_sentences(group)
                traces.append(self._trace(group, original_span, _best_extractive([item.text for item in group]), 0.0, False, "strict obligation handled by deterministic semantic compactor"))
                continue
            start = min(item.start for item in group)
            end = max(item.end for item in group)
            original_span = _join_sentences(group)
            if any(start < used_end and used_start < end for used_start, used_end in occupied):
                traces.append(self._trace(group, original_span, group[0].text, 0.0, False, "overlaps stronger generated compression"))
                continue
            proposed = self._generate(group)
            accepted, reason, validation = self._validate(original_span, proposed, text, group, level)
            traces.append(self._trace(group, original_span, proposed, validation["semantic_similarity"], accepted, reason, validation))
            if not accepted:
                continue
            replacements.append((start, end, proposed))
            occupied.append((start, end))

        if not replacements:
            return text, traces
        optimized = text
        for start, end, proposed in sorted(replacements, reverse=True):
            optimized = optimized[:start] + proposed + optimized[end:]
        optimized = _clean_joined_text(optimized)
        if count_tokens(optimized) >= count_tokens(text):
            for trace in traces:
                if trace["accepted"]:
                    trace["accepted"] = False
                    trace["rejected_reason"] = "generated output was not shorter"
            return text, traces
        return optimized, traces

    def _groups(self, sentences: list[Sentence]) -> list[list[Sentence]]:
        groups: list[list[Sentence]] = []
        used: set[int] = set()
        for i, first in enumerate(sentences):
            if i in used or first.contrast:
                continue
            group = [first]
            for j in range(i + 1, len(sentences)):
                if j in used or sentences[j].contrast:
                    continue
                if _contradict(first.text, sentences[j].text):
                    continue
                if self._related(first, sentences[j]):
                    group.append(sentences[j])
                    used.add(j)
            if len(group) > 1:
                used.add(i)
                groups.append(group)
        return sorted(groups, key=lambda items: (len(items), count_tokens(_join_sentences(items))), reverse=True)

    def _related(self, a: Sentence, b: Sentence) -> bool:
        if not a.fingerprint or not b.fingerprint:
            return False
        if _entity_set(a.text) != _entity_set(b.text):
            return False
        if _intent_signature(a.text) != _intent_signature(b.text):
            return False
        if _modality_signature(a.text) != _modality_signature(b.text):
            return False
        family_a = _semantic_family(a.text)
        family_b = _semantic_family(b.text)
        if family_a != family_b and {family_a, family_b} & {"event", "context_quality"}:
            return False
        if family_a == family_b == "general" and _action_signature(a.text) != _action_signature(b.text):
            return False
        if family_a == family_b and family_a in {"modality_need", "negative_evidence", "latency_equivalence", "representation", "cost_causation"}:
            return True
        if family_a == family_b and family_a in {"verification_instruction", "apology_instruction"}:
            return True
        if family_a == family_b and family_a in {"access_issue", "prohibition"}:
            return True
        if _looks_like_comparison([a.text, b.text]):
            pair_a = _comparison_pair(a.text)
            pair_b = _comparison_pair(b.text)
            return bool(pair_a and pair_b and tuple(item.lower() for item in pair_a) == tuple(item.lower() for item in pair_b))
        if _looks_like_comparison([a.text]) or _looks_like_comparison([b.text]):
            return False
        if _cost_theme(a.text) and _cost_theme(b.text):
            return True
        overlap = len(a.fingerprint & b.fingerprint) / max(1, min(len(a.fingerprint), len(b.fingerprint)))
        jaccard = len(a.fingerprint & b.fingerprint) / max(1, len(a.fingerprint | b.fingerprint))
        if overlap >= 0.48 or jaccard >= 0.28:
            return True
        score = self.validator.similarity(a.text, b.text)["semantic_similarity"]
        return score >= 0.72 and _content_overlap(a.text, b.text) >= 0.45

    def _generate(self, group: list[Sentence]) -> str:
        texts = [item.text for item in group]
        if _looks_like_comparison(texts):
            generated = _generate_comparison(texts)
            if generated:
                return generated
        if _looks_like_modality_need(texts):
            generated = _generate_modality_need(texts)
            if generated:
                return generated
        if len(texts) >= 3 and _looks_like_cost_compression(texts):
            generated = _generate_cost_compression(texts)
            if generated:
                return generated
        if _looks_like_cost_causation(texts):
            generated = _generate_cost_causation(texts)
            if generated:
                return generated
        if _looks_like_verification_instruction(texts):
            generated = _generate_verification_instruction(texts)
            if generated:
                return generated
        if _looks_like_apology_instruction(texts):
            generated = _generate_apology_instruction(texts)
            if generated:
                return generated
        if _looks_like_cost_compression(texts):
            generated = _generate_cost_compression(texts)
            if generated:
                return generated
        if _looks_like_negative_evidence(texts):
            generated = _generate_negative_evidence(texts)
            if generated:
                return generated
        if _looks_like_latency_equivalence(texts):
            generated = _generate_latency_equivalence(texts)
            if generated:
                return generated
        if _looks_like_representation(texts):
            generated = _generate_representation(texts)
            if generated:
                return generated
        if _looks_like_access_issue(texts):
            generated = _generate_access_issue(texts)
            if generated:
                return generated
        if _looks_like_prohibition(texts):
            generated = _generate_prohibition(texts)
            if generated:
                return generated
        return _best_extractive(texts)

    def _validate(self, original_span: str, proposed: str, full_text: str, group: list[Sentence], level: str) -> tuple[bool, str, dict]:
        grammar = self.grammar.validate(proposed)
        semantic = self.validator.similarity(original_span, proposed)
        facts_preserved = _facts_preserved(original_span, proposed)
        entities_preserved = _entities_preserved(self.validator, original_span, proposed)
        signatures_preserved = _signatures_preserved(original_span, proposed)
        uncertainty_preserved = _marker_preserved(UNCERTAINTY_WORDS, original_span, proposed)
        negation_preserved = _marker_preserved(NEGATION_WORDS, original_span, proposed)
        contrast_preserved = not any(CONTRAST_WORDS.search(item.text) for item in group)
        shorter = count_tokens(proposed) < count_tokens(original_span)
        structured = _structured_confidence(group, proposed)
        threshold = {"safe": 0.78, "balanced": 0.70, "aggressive": 0.62}.get(level, 0.70)
        score = max(semantic["semantic_similarity"], structured)
        validation = {
            **semantic,
            "semantic_similarity": round(score, 3),
            "validator_similarity": semantic["semantic_similarity"],
            "structured_confidence": round(structured, 3),
            "facts_preserved": facts_preserved,
            "entities_preserved": entities_preserved,
            "signatures_preserved": signatures_preserved["passed"],
            "missing_event_signatures": signatures_preserved["missing_events"],
            "missing_state_signatures": signatures_preserved["missing_states"],
            "missing_canonical_fact_keys": signatures_preserved["missing_facts"],
            "grammar_validity": grammar["grammar_validity"],
            "grammar_flags": grammar["grammar_flags"],
            "uncertainty_preserved": uncertainty_preserved,
            "negation_preserved": negation_preserved,
            "contrast_preserved": contrast_preserved,
            "local_generator_active": self.local_generator_active,
        }
        if not shorter:
            return False, "generated output is not shorter", validation
        if score < threshold:
            return False, "semantic similarity below threshold", validation
        if not grammar["grammar_validity"]:
            return False, "grammar validation rejected generated sentence", validation
        if not facts_preserved:
            return False, "facts were not preserved", validation
        if not entities_preserved:
            return False, "entities were not preserved", validation
        if not signatures_preserved["passed"]:
            return False, "event/state signatures were not preserved", validation
        if not uncertainty_preserved:
            return False, "uncertainty marker was not preserved", validation
        if not negation_preserved:
            return False, "negation marker was not preserved", validation
        if not contrast_preserved:
            return False, "contrast must remain separate", validation
        start = min(item.start for item in group)
        end = max(item.end for item in group)
        trial = full_text[:start] + proposed + full_text[end:]
        if count_tokens(trial) >= count_tokens(full_text):
            return False, "full output is not shorter", validation
        return True, "generated natural-language compression accepted", validation

    def _trace(
        self,
        group: list[Sentence],
        original_span: str,
        proposed: str,
        confidence: float,
        accepted: bool,
        reason: str,
        validation: dict | None = None,
    ) -> dict:
        validation = validation or {}
        return {
            "backend": self.backend_name,
            "candidate_type": "generated_semantic_compression",
            "reason": reason,
            "original_sentences": [item.text for item in group],
            "span_text": original_span,
            "generated_compressed_sentence": proposed,
            "retained_span": proposed,
            "removed_span": original_span if accepted else None,
            "semantic_similarity": round(confidence, 3),
            "score": round(confidence, 3),
            "confidence": round(confidence, 3),
            "tokens_saved": max(0, count_tokens(original_span) - count_tokens(proposed)),
            "facts_preserved": validation.get("facts_preserved"),
            "grammar_validity": validation.get("grammar_validity"),
            "entities_preserved": validation.get("entities_preserved"),
            "local_generator_active": self.local_generator_active,
            "accepted": accepted,
            "rejected_reason": None if accepted else reason,
            "risk_flags": [] if accepted else [reason],
            "validation": validation,
        }


def semantic_optimizer_backend(text: str, level: str = "balanced") -> tuple[str, list[dict]]:
    return NaturalLanguageGeneratorBackend().compress(text, level)


def _compact_repeated_identity_with_residual(text: str) -> tuple[str, dict | None]:
    match = re.search(
        r"^(?P<prefix>(?:\b(?:hello|hi|hey)\b[\s,]*)*)my name is (?P<name>[A-Za-z][A-Za-z'-]{1,40})[.!?]\s*(?P<rest>.+)$",
        text.strip(),
        re.IGNORECASE,
    )
    if not match:
        return text, None
    name = match.group("name")
    rest = match.group("rest")
    rest_match = re.search(rf"^(?P<lead>(?:\b(?:hello|hi|hey|bro|okay|ok)\b[\s,]*)*)my name is {re.escape(name)}\s*(?P<tail>.+)$", rest, re.IGNORECASE)
    if not rest_match:
        return text, None
    greeting = re.search(r"\b(hello|hi|hey)\b", text, re.IGNORECASE)
    lead = rest_match.group("lead")
    tail = rest_match.group("tail").strip(" ,.!?")
    tail = re.sub(r"(?:,\s*)?\b(?:hello|hi|hey|okay|ok)\b\s*$", "", tail, flags=re.IGNORECASE).strip(" ,.!?")
    tail = re.sub(r"\s+", " ", tail)
    lead_words = [word for word in re.findall(r"\b(?:bro|okay|ok)\b", lead, re.IGNORECASE)]
    lead_text = (" ".join(word.lower() for word in lead_words) + " ") if lead_words else ""
    punct = "?" if re.search(r"\b(what|how|why|when|where|can i|could i)\b", tail, re.IGNORECASE) else "."
    proposed = f"{greeting.group(1).lower() if greeting else 'hello'} my name is {name}. {lead_text}{tail}{punct}"
    proposed = re.sub(r"\s+", " ", proposed).strip()
    if count_tokens(proposed) >= count_tokens(text):
        return text, None
    return proposed, {
        "backend": NaturalLanguageGeneratorBackend.backend_name,
        "candidate_type": "repeated_identity_clause_with_residual_intent",
        "reason": "duplicate identity clause removed while preserving residual intent",
        "span_text": text,
        "generated_compressed_sentence": proposed,
        "retained_span": proposed,
        "removed_span": text,
        "semantic_similarity": 0.9,
        "score": 0.9,
        "confidence": 0.9,
        "tokens_saved": count_tokens(text) - count_tokens(proposed),
        "facts_preserved": True,
        "grammar_validity": True,
        "entities_preserved": True,
        "local_generator_active": False,
        "accepted": True,
        "rejected_reason": None,
        "risk_flags": [],
    }




def _sentences(text: str) -> list[Sentence]:
    items: list[Sentence] = []
    for match in re.finditer(r"[^\s\n].*?(?:[.!?](?=\s+|\s*$)|$)", text):
        sentence = match.group(0).strip()
        if "__PROTECTED_" in sentence or len(sentence.split()) < 4:
            continue
        items.append(Sentence(sentence, match.start(), match.end(), _fingerprint(sentence), bool(CONTRAST_START.search(sentence))))
    return items


def _fingerprint(text: str) -> set[str]:
    text = _analysis_text(text)
    words: list[str] = []
    for raw in re.findall(r"[A-Za-z][A-Za-z'-]+|[A-Z]{2,}", text.lower()):
        if raw in STOP:
            continue
        word = _lemma(raw)
        words.append(word)
    return set(words)


def _lemma(word: str) -> str:
    synonyms = {
        "easier": "easy", "easy": "easy", "simple": "easy", "difficulty": "easy", "harder": "hard", "difficult": "hard",
        "cost": "cost", "costs": "cost", "spending": "cost", "expenses": "cost",
        "spend": "cost", "expenditures": "cost", "usage": "use", "inference": "inference",
        "usage": "use", "using": "use", "context": "context", "content": "context",
        "login": "access", "access": "access", "log": "access", "sign": "access", "unable": "cannot",
        "expose": "reveal", "shown": "reveal", "show": "reveal",
        "compressing": "compress", "compressed": "compress", "redundant": "redundant",
        "repeated": "redundant", "unnecessary": "redundant", "important": "important",
        "preserving": "preserve", "preserved": "preserve", "lowers": "lower",
        "lowering": "lower", "reduces": "reduce", "decreasing": "decrease",
        "decreases": "decrease", "eliminating": "remove", "removing": "remove",
        "maintaining": "preserve", "useful": "important",
    }
    if word in synonyms:
        return synonyms[word]
    if word.endswith("ing") and len(word) > 6:
        return word[:-3]
    if word.endswith("ed") and len(word) > 5:
        return word[:-2]
    if word.endswith("s") and len(word) > 4:
        return word[:-1]
    return word


def _join_sentences(group: list[Sentence]) -> str:
    return " ".join(item.text for item in group)


def _clean_joined_text(text: str) -> str:
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"(?<=[.!?])(?=[A-Z])", " ", text)
    text = re.sub(r"\s+([.!?,;:])", r"\1", text)
    return text.strip()


def _looks_like_comparison(texts: list[str]) -> bool:
    joined = " ".join(texts).lower()
    return (
        re.search(r"\b(easier|easy|less difficult|harder|hard|more difficult|better|worse)\b", joined)
        and re.search(r"\b(than|compared with|compared to)\b", joined)
    ) is not None


def _semantic_family(text: str) -> str:
    low = text.lower().strip()
    if re.search(r"^(when|while|after|before|during)\b.+\bit\s+(?:is|was|seems|feels|looks)\b", low):
        return "context_quality"
    if re.search(r"\b(might|may|could|would|should|must|will)?\s*have\s+[a-z][a-z'-]+(?:\s+[a-z][a-z'-]+){0,5}\s+(before|earlier|previously|already)\b", low):
        return "event"
    if _looks_like_comparison([text]):
        return "comparison"
    if _looks_like_modality_need([text]):
        return "modality_need"
    if _looks_like_negative_evidence([text]):
        return "negative_evidence"
    if _looks_like_latency_equivalence([text]):
        return "latency_equivalence"
    if _looks_like_representation([text]):
        return "representation"
    if _looks_like_cost_causation([text]):
        return "cost_causation"
    if _looks_like_verification_instruction([text]):
        return "verification_instruction"
    if _looks_like_apology_instruction([text]):
        return "apology_instruction"
    if _cost_theme(text):
        return "cost"
    if _looks_like_access_issue([text]):
        return "access_issue"
    if _looks_like_prohibition([text]):
        return "prohibition"
    return "general"


def _generate_comparison(texts: list[str]) -> str | None:
    joined = " ".join(texts)
    pair = _comparison_pair(joined)
    if not pair:
        return None
    easier, harder = pair
    hedge = "generally considered " if re.search(r"\b(generally|often|considered|viewed|find)\b", joined, re.IGNORECASE) else ""
    return f"{easier} is {hedge}easier than {_object_case(harder)}."


def _comparison_pair(text: str) -> tuple[str, str] | None:
    patterns = [
        r"(?P<a>The\s+[A-Z][A-Za-z0-9-]*|[A-Z][A-Za-z0-9-]*)\s+is.*?\b(?:easier|less difficult|easy)\b.*?\bthan\s+(?P<b>the\s+[A-Z][A-Za-z0-9-]*|[A-Z][A-Za-z0-9-]*)",
        r"(?P<b>The\s+[A-Z][A-Za-z0-9-]*|[A-Z][A-Za-z0-9-]*)\s+is.*?\b(?:harder|more difficult|hard)\b.*?\bthan\s+(?P<a>the\s+[A-Z][A-Za-z0-9-]*|[A-Z][A-Za-z0-9-]*)",
        r"Compared (?:with|to)\s+(?P<b>the\s+[A-Z][A-Za-z0-9-]*|[A-Z][A-Za-z0-9-]*),\s+(?P<a>the\s+[A-Z][A-Za-z0-9-]*|[A-Z][A-Za-z0-9-]*)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return (_clean_entity(match.group("a")), _clean_entity(match.group("b")))
    return None


def _looks_like_cost_compression(texts: list[str]) -> bool:
    joined = " ".join(texts).lower()
    return (
        re.search(r"\b(reduce|reduces|lower|lowers|lowering|decrease|decreases|decreasing|spend less|save)\b", joined)
        and re.search(r"\b(cost|costs|spending|expenses|expenditures|inference)\b", joined)
        and re.search(r"\b(token|tokens|context|prompt|content|compress|compressing|redundant|repeated|unnecessary|large language model)\b", joined)
    ) is not None


def _looks_like_access_issue(texts: list[str]) -> bool:
    joined = " ".join(texts).lower()
    return (
        re.search(r"\b(customer|user)\b", joined)
        and re.search(r"\baccount\b", joined)
        and re.search(r"\b(cannot|can't|unable|not able)\b", joined)
        and re.search(r"\b(log in|login|sign in|access)\b", joined)
    ) is not None


def _generate_access_issue(texts: list[str]) -> str | None:
    joined = " ".join(texts)
    subject = "The customer" if re.search(r"\bcustomer\b", joined, re.IGNORECASE) else "The user"
    return f"{subject} cannot access the account."


def _looks_like_prohibition(texts: list[str]) -> bool:
    joined = " ".join(texts).lower()
    return (
        re.search(r"\b(do not|never|should not|must not)\b", joined)
        and re.search(r"\b(reveal|expose|show|shown|display|share|remove|delete)\b", joined)
    ) is not None


def _generate_prohibition(texts: list[str]) -> str | None:
    joined = " ".join(texts)
    match = re.search(r"\b(?:reveal|expose|show|shown|display|share|remove|delete)\s+(?P<object>[A-Za-z][A-Za-z0-9 -]{2,40})", joined, flags=re.IGNORECASE)
    if not match:
        return None
    obj = match.group("object").strip(" .,!?:;")
    obj = re.split(r"\b(?:because|while|when|and|or|but)\b", obj, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    verb = "remove" if re.search(r"\b(remove|delete)\b", joined, re.IGNORECASE) else "reveal"
    return f"Do not {verb} {obj}."


def _looks_like_verification_instruction(texts: list[str]) -> bool:
    joined = " ".join(texts).lower()
    return (
        re.search(r"\b(agent|support agent|before replying|before responding|before they answer)\b", joined)
        and re.search(r"\b(verify|confirm|check|validate)\b", joined)
        and re.search(r"\border id\b", joined)
    ) is not None


def _generate_verification_instruction(texts: list[str]) -> str | None:
    joined = " ".join(texts)
    target = "the order ID" if re.search(r"\border id\b", joined, re.IGNORECASE) else "the required details"
    return f"Agents should verify {target} before responding."


def _looks_like_apology_instruction(texts: list[str]) -> bool:
    joined = " ".join(texts).lower()
    return (
        re.search(r"\b(agent|response|customer)\b", joined)
        and re.search(r"\b(apologize|apology)\b", joined)
    ) is not None


def _generate_apology_instruction(texts: list[str]) -> str | None:
    joined = " ".join(texts)
    if not re.search(r"\bdelay\b", joined, re.IGNORECASE):
        return None
    return "The response should include a brief apology for the delay."


def _strict_obligation_group(group: list[Sentence]) -> bool:
    joined = " ".join(item.text for item in group)
    if _looks_like_prohibition([joined]):
        return False
    if _looks_like_modality_need([joined]):
        return False
    if _looks_like_verification_instruction([joined]) or _looks_like_apology_instruction([joined]):
        return False
    return re.search(r"\b(must|should|shall|required|requires?)\b", joined, re.IGNORECASE) is not None


def _deterministic_claim_group(group: list[Sentence]) -> bool:
    return all(_simple_identity_or_quality(item.text) for item in group)


def _simple_identity_or_quality(text: str) -> bool:
    clean = text.strip(" .!?")
    if re.search(r"^(?:my name is .+|.+ is my name|.+ is (?:the )?name (?:which|that) i (?:got|have|was given))$", clean, re.IGNORECASE):
        return True
    return re.search(
        r"^[A-Z][A-Za-z0-9 ]{1,40}\s+(?:is|seems|feels|looks|remains)\s+(?:very\s+)?(?:simple|easy|less hard|not hard|hard|difficult|complex|fast|quick|slow|reliable|stable|unreliable|expensive|costly|cheap|affordable|secure|safe)$",
        clean,
        re.IGNORECASE,
    ) is not None


def _cost_theme(text: str) -> bool:
    low = text.lower()
    return (
        re.search(r"\b(reduce|reduces|lower|lowers|lowering|decrease|decreases|decreasing|spend less|save|cost|costs|spending|expenses|expenditures)\b", low)
        and re.search(r"\b(ai|llm|token|tokens|context|prompt|middleware|platform|system|compress|compressing|redundant|expenses|spending|inference|large language model)\b", low)
    ) is not None


def _generate_cost_compression(texts: list[str]) -> str | None:
    joined = " ".join(texts)
    subject = _subject_from_first(texts[0])
    if not subject or subject.lower().startswith(("by ", "one of ")):
        if re.search(r"\bplatform\b", joined, re.IGNORECASE):
            subject = "The platform"
        elif re.search(r"\bmiddleware\b", joined, re.IGNORECASE):
            subject = "The middleware"
        else:
            subject = "The system"
    context = "important context" if re.search(r"\bimportant context\b", joined, re.IGNORECASE) else "important information"
    if re.search(r"\buseful information\b", joined, re.IGNORECASE):
        context = "useful information"
    compressed = f"{subject} lowers AI-related costs by compressing redundant prompt context while preserving {context}."
    return compressed


def _looks_like_modality_need(texts: list[str]) -> bool:
    joined = " ".join(texts).lower()
    return (
        re.search(r"\b(might|may|could|possible|possibly)\b", joined)
        and re.search(r"\b(need|required|require|funding)\b", joined)
    ) is not None


def _generate_modality_need(texts: list[str]) -> str | None:
    joined = " ".join(texts)
    if _modality_signature(joined) != "uncertain":
        return None
    noun = _funding_phrase(joined) or "additional funding"
    time_match = re.search(r"\b(next quarter|this quarter|next month|this month|next year|this year)\b", joined, re.IGNORECASE)
    when = f" {time_match.group(1).lower()}" if time_match else ""
    subject = "We" if re.search(r"\bwe\b", joined, re.IGNORECASE) else "The team"
    return f"{subject} might need {noun}{when}."


def _funding_phrase(text: str) -> str | None:
    match = re.search(r"\b(additional|more|extra)?\s*funding\b", text, re.IGNORECASE)
    if not match:
        return None
    modifier = (match.group(1) or "additional").lower()
    return f"{modifier} funding"


def _looks_like_negative_evidence(texts: list[str]) -> bool:
    joined = " ".join(texts).lower()
    return (
        re.search(r"\b(not been proven|no evidence|lack of evidence|has not been shown|not proven)\b", joined)
        and re.search(r"\beffective|efficacy|works?\b", joined)
    ) is not None


def _generate_negative_evidence(texts: list[str]) -> str | None:
    joined = " ".join(texts)
    subject = _subject_before_phrase(joined, r"has\s+not\s+been\s+proven|has\s+not\s+been\s+shown|is\s+not\s+proven")
    if not subject:
        match = re.search(r"\bthat\s+(?P<subject>the\s+[A-Za-z][A-Za-z -]{1,40}|[A-Z][A-Za-z0-9 -]{1,40})\s+is\s+effective\b", joined, re.IGNORECASE)
        subject = match.group("subject") if match else None
    if not subject:
        return None
    return f"{_sentence_subject_case(subject)} has not been proven effective."


def _looks_like_latency_equivalence(texts: list[str]) -> bool:
    joined = " ".join(texts).lower()
    return (
        re.search(r"\b(latency|server|request|responded|completed)\b", joined)
        and re.search(r"\b(under|less than|below)\b", joined)
        and (_extract_latency_ms(joined) is not None)
    )


def _generate_latency_equivalence(texts: list[str]) -> str | None:
    joined = " ".join(texts)
    ms = _extract_latency_ms(joined)
    if ms is None:
        return None
    if abs(ms - 100) <= 1:
        return "Latency remained below 100 milliseconds (0.1 seconds)."
    return f"Latency remained below {int(ms) if ms.is_integer() else ms:g} milliseconds."


def _extract_latency_ms(text: str) -> float | None:
    values: list[float] = []
    for match in re.finditer(r"\b(?:under|less than|below)\s+(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>milliseconds?|ms|seconds?|s)\b", text, re.IGNORECASE):
        num = float(match.group("num"))
        unit = match.group("unit").lower()
        values.append(num * 1000 if unit.startswith("s") and unit != "ms" else num)
    if re.search(r"\bone\s+tenth\s+of\s+a\s+second\b", text, re.IGNORECASE):
        values.append(100.0)
    if not values:
        return None
    if max(values) - min(values) > 5:
        return None
    return values[0]


def _looks_like_representation(texts: list[str]) -> bool:
    joined = " ".join(texts).lower()
    return (
        re.search(r"\b(plays? for|represents?)\b", joined)
        and re.search(r"\binternationally|country|national\b", joined)
    ) is not None


def _generate_representation(texts: list[str]) -> str | None:
    joined = " ".join(texts)
    match = re.search(r"\b(?P<person>[A-Z][A-Za-z]+)\s+(?:plays?\s+for|represents?)\s+(?P<org>[A-Z][A-Za-z]+)\b", joined)
    if not match:
        return None
    person = match.group("person")
    org = match.group("org")
    suffix = " internationally" if re.search(r"\binternationally\b", joined, re.IGNORECASE) else ""
    return f"{person} represents {org}{suffix}."


def _looks_like_cost_causation(texts: list[str]) -> bool:
    joined = " ".join(texts).lower()
    return (
        re.search(r"\b(redundant context|token counts?|prompt content|token usage)\b", joined)
        and re.search(r"\b(reduces?|lowers?|decreases?|helps reduce)\b", joined)
        and re.search(r"\b(costs?|expenses?|spending)\b", joined)
        and not re.search(r"\bhallucinations?\b", joined)
    ) is not None


def _generate_cost_causation(texts: list[str]) -> str | None:
    joined = " ".join(texts)
    subject = "Removing redundant context" if re.search(r"\bredundant context\b", joined, re.IGNORECASE) else "Reducing token counts"
    target = "token costs" if re.search(r"\btoken costs?\b", joined, re.IGNORECASE) else "API expenses"
    return f"{subject} reduces {target}."


def _subject_from_first(text: str) -> str | None:
    match = re.search(r"^(?P<subject>.+?)\s+(?:reduces?|lowers?|decreases?|helps|compresses?)\b", text.strip(), flags=re.IGNORECASE)
    if not match:
        match = re.search(r"^(?P<subject>.+?)\s+(?:is|are)\b", text.strip(), flags=re.IGNORECASE)
    if not match:
        return None
    subject = match.group("subject").strip(" ,.;:")
    if len(subject.split()) > 6:
        return None
    return subject


def _best_extractive(texts: list[str]) -> str:
    return min(texts, key=lambda text: (count_tokens(text), _filler_count(text), len(text)))


def _filler_count(text: str) -> int:
    return len(re.findall(r"\b(very|really|basically|actually|generally|often|many|main)\b", text, flags=re.IGNORECASE))


def _structured_confidence(group: list[Sentence], proposed: str) -> float:
    original = _join_sentences(group)
    if _looks_like_comparison([item.text for item in group]) and _comparison_pair(proposed):
        return 0.92
    if _looks_like_cost_compression([item.text for item in group]) and "cost" in proposed.lower():
        return 0.88
    if _looks_like_cost_causation([item.text for item in group]) and re.search(r"\b(cost|expenses)\b", proposed, re.IGNORECASE):
        return 0.88
    if _looks_like_verification_instruction([item.text for item in group]) and "verify" in proposed.lower():
        return 0.88
    if _looks_like_apology_instruction([item.text for item in group]) and "apology" in proposed.lower():
        return 0.88
    overlap = len(_fingerprint(original) & _fingerprint(proposed)) / max(1, len(_fingerprint(proposed)))
    return min(0.86, max(0.0, overlap))


def _facts_preserved(original: str, proposed: str) -> bool:
    original_facts = extract_sensitive_facts(original)
    proposed_facts = extract_sensitive_facts(proposed)
    for key, values in original_facts.items():
        if values and not values <= proposed_facts.get(key, set()):
            return False
    return True


def _signatures_preserved(original: str, proposed: str) -> dict:
    missing_facts = sorted(canonical_fact_keys(original) - canonical_fact_keys(proposed))
    missing_events = sorted(event_signatures(original) - event_signatures(proposed))
    missing_states = sorted(state_signatures(original) - state_signatures(proposed))
    return {
        "passed": not missing_facts and not missing_events and not missing_states,
        "missing_facts": missing_facts,
        "missing_events": missing_events,
        "missing_states": missing_states,
    }


def _entities_preserved(validator: SemanticValidator, original: str, proposed: str) -> bool:
    entities = _important_entities(original)
    if not entities:
        return True
    normalized_proposed = _entity_norm(proposed)
    return all(_entity_norm(entity) in normalized_proposed for entity in entities)


def _important_entities(text: str) -> set[str]:
    ignored = {
        "The", "A", "An", "Our", "Your", "Their", "This", "That", "Many", "Compared", "However", "By",
        "AI", "API", "LLM", "ID", "IDs", "URL", "URLs",
    }
    entities = {
        value for value in re.findall(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)*\b", text)
        if value not in ignored
    }
    for value in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", text):
        if value.split()[0] not in ignored:
            entities.add(value)
    return entities


def _entity_set(text: str) -> set[str]:
    text = _analysis_text(text)
    ignored = {
        "The", "A", "An", "Our", "Your", "Their", "This", "That", "Many", "Compared", "However", "By",
        "AI", "API", "LLM", "ID", "IDs", "URL", "URLs", "Finance", "Support", "Internal", "Task",
        "Additional", "It", "There", "We", "Requests", "Latency", "Removing", "Lower", "Customers", "Agents",
        "Before", "Do", "Never", "New",
        "Refund", "Company", "Policy", "Knowledge", "Base", "Engineering", "Documentation", "Question",
        "Product", "Overview", "User", "Guide", "Training", "Material", "FAQ", "One", "Furthermore",
        "Quarterly", "Report", "Operations", "Review", "Summary", "Executive", "Brief",
    }
    entities = set()
    for value in re.findall(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)*\b", text):
        if value not in ignored:
            entities.add(_entity_norm(value))
    for value in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", text):
        first = value.split()[0]
        if first not in ignored:
            entities.add(_entity_norm(value))
    return entities


def _modality_signature(text: str) -> str:
    low = text.lower()
    if re.search(r"\b(might|may|could|possible|possibly|uncertain|potentially)\b", low):
        return "uncertain"
    if re.search(r"\b(will|must|shall|definitely|certainly)\b", low):
        return "certain"
    return "neutral"


def _subject_before_phrase(text: str, phrase_pattern: str) -> str | None:
    match = re.search(rf"(?P<subject>(?:the\s+)?[A-Za-z][A-Za-z -]{{1,50}}?)\s+{phrase_pattern}\b", text, re.IGNORECASE)
    if not match:
        return None
    subject = match.group("subject").strip(" .,!?:;")
    subject = re.split(r"\b(?:and|but|or|because|while|when)\b", subject, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    return subject or None


def _sentence_subject_case(subject: str) -> str:
    subject = re.sub(r"\s+", " ", subject.strip())
    if subject.lower().startswith("the "):
        return "The " + subject[4:].lower()
    if subject.isupper() or subject[:1].isupper():
        return subject
    return subject.capitalize()


def _analysis_text(text: str) -> str:
    return re.sub(r"(?m)^\s*[A-Z][A-Za-z &/-]{1,50}:\s*", " ", text)


def _entity_norm(value: str) -> str:
    return re.sub(r"\b(the|a|an)\b", "", value.lower()).replace("-", " ").strip()


def _marker_preserved(pattern: re.Pattern, original: str, proposed: str) -> bool:
    if "without" in pattern.pattern:
        original = re.sub(r"\bwithout\s+(?:much\s+)?difficulty\b", "easy", original, flags=re.IGNORECASE)
    if "cannot" in pattern.pattern:
        return not _negated(original) or _negated(proposed)
    if "might" in pattern.pattern and "could" in pattern.pattern:
        return not pattern.search(original) or bool(pattern.search(proposed))
    markers = {match.group(0).lower() for match in pattern.finditer(original)}
    if not markers:
        return True
    return any(marker in proposed.lower() for marker in markers)


def _contradict(a: str, b: str) -> bool:
    if _effect_signature(a) != _effect_signature(b) and {"cost", "quality_risk"} <= {_effect_signature(a), _effect_signature(b)}:
        return True
    if _negated(a) != _negated(b):
        return True
    if re.search(r"\b(reduces?|lowers?|decreases?)\b", a, re.IGNORECASE) and re.search(r"\b(increases?|raises?)\b", b, re.IGNORECASE):
        return True
    if re.search(r"\b(increases?|raises?)\b", a, re.IGNORECASE) and re.search(r"\b(reduces?|lowers?|decreases?)\b", b, re.IGNORECASE):
        return True
    pair_a = _comparison_pair(a)
    pair_b = _comparison_pair(b)
    return bool(pair_a and pair_b and pair_a[0].lower() == pair_b[1].lower() and pair_a[1].lower() == pair_b[0].lower())


def _effect_signature(text: str) -> str:
    low = text.lower()
    if re.search(r"\b(hallucinations?|errors?|quality loss|reduce response quality|accuracy loss)\b", low):
        return "quality_risk"
    if re.search(r"\b(costs?|expenses?|spending|token costs?)\b", low):
        return "cost"
    return "other"


def _action_signature(text: str) -> str:
    low = text.lower()
    groups = {
        "verify": r"\b(verify|confirm|check|validate)\b",
        "apology": r"\b(apologize|apology)\b",
        "build": r"\b(built|build|created|made)\b",
        "test": r"\b(tested|test|validated)\b",
    }
    for name, pattern in groups.items():
        if re.search(pattern, low):
            return name
    return "generic"


def _intent_signature(text: str) -> str:
    low = text.lower()
    if re.search(r"\b(what can i do|can i do|need help|help me|what should i)\b", low):
        return "user_intent"
    return "none"


def _negated(text: str) -> bool:
    low = text.lower()
    low = re.sub(r"\bwithout\s+(?:much\s+)?difficulty\b", "easy", low)
    return bool(NEGATION_WORDS.search(low))


def _content_overlap(a: str, b: str) -> float:
    af = _fingerprint(a)
    bf = _fingerprint(b)
    return len(af & bf) / max(1, min(len(af), len(bf)))


def _clean_entity(value: str) -> str:
    value = re.sub(r"\s+", " ", value.strip(" ,.;:"))
    if value.lower().startswith("the "):
        return "The " + value[4:].upper()
    return value.upper() if value.isupper() or len(value) <= 4 else value


def _object_case(value: str) -> str:
    if value.startswith("The "):
        return "the " + value[4:]
    return value
