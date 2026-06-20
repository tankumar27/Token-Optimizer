from __future__ import annotations

from dataclasses import dataclass
import re

from .semantic_validator import SemanticValidator
from .token_counter import count_tokens


FILLER = {
    "very", "really", "extremely", "quite", "more", "much", "maybe", "probably",
    "possibly", "basically", "actually",
    "comparison", "relative", "relatively",
}

QUALITY_CANONICAL = {
    "simple": "easy",
    "easy": "easy",
    "easier": "easy",
    "less hard": "easy",
    "not hard": "easy",
    "better": "better",
    "hard": "hard",
    "harder": "hard",
    "difficult": "hard",
    "complex": "complex",
    "fast": "fast",
    "quick": "fast",
    "slow": "slow",
    "reliable": "reliable",
    "stable": "reliable",
    "unreliable": "unreliable",
    "expensive": "expensive",
    "not expensive": "cheap",
    "costly": "expensive",
    "cheap": "cheap",
    "cheaper": "cheap",
    "affordable": "cheap",
    "faster": "fast",
    "slower": "slow",
    "secure": "secure",
    "safe": "secure",
}

OPPOSITES = {
    "easy": "hard",
    "hard": "easy",
    "fast": "slow",
    "slow": "fast",
    "reliable": "unreliable",
    "unreliable": "reliable",
    "expensive": "cheap",
    "cheap": "expensive",
    "secure": "unsecure",
}

REQUIREMENT_WORDS = {"must", "required", "requires", "need", "needs", "should", "shall"}
CAPABILITY_WORDS = {"can", "supports", "allows", "able"}
NEGATION_WORDS = {"not", "never", "cannot", "can't", "must not", "prohibited"}
NAME = r"[A-Za-z0-9][A-Za-z0-9-]*(?:\s+[A-Za-z0-9][A-Za-z0-9-]*){0,3}"


@dataclass
class SemanticClaim:
    sentence: str
    claim_type: str
    subject: str
    relation: str
    target: str
    polarity: str
    canonical: str
    start: int
    end: int

    @property
    def key(self) -> tuple[str, str, str, str, str]:
        return (
            self.claim_type,
            self.subject.lower(),
            self.relation,
            self.target.lower(),
            self.polarity,
        )

    @property
    def opposite_key(self) -> tuple[str, str, str, str, str]:
        opposite = OPPOSITES.get(self.polarity, f"not_{self.polarity}")
        return (
            self.claim_type,
            self.subject.lower(),
            self.relation,
            self.target.lower(),
            opposite,
        )


class SemanticClaimCompactor:
    """Conservative sentence-level semantic redundancy compiler.

    It turns repeated *claims* into one canonical claim only when a structured
    signature says the sentences have the same subject, relation, target, and
    polarity. This is not free-form LLM rewriting; it is deterministic claim
    normalization plus optional local semantic validation traces.
    """

    def __init__(self) -> None:
        self.validator = SemanticValidator()

    def compact(self, text: str) -> tuple[str, list[dict]]:
        claims = self._claims(text)
        groups: dict[tuple[str, str, str, str, str], list[SemanticClaim]] = {}
        for claim in claims:
            groups.setdefault(claim.key, []).append(claim)
        self._attach_self_comparisons(claims, groups)
        self._attach_quality_to_comparatives(claims, groups)

        selected: list[tuple[int, int, str, list[SemanticClaim]]] = []
        traces: list[dict] = []
        occupied: list[tuple[int, int]] = []
        traced_keys: set[tuple[str, str, str, str, str]] = set()
        for key, group in groups.items():
            if self._has_contradiction(key, groups):
                traces.append(self._trace(group, group[0].canonical, 0.0, False, "opposite semantic claim present"))
                traced_keys.add(key)
                continue
            if len(group) < 2:
                continue
            start = min(item.start for item in group)
            end = max(item.end for item in group)
            if any(start < used_end and used_start < end for used_start, used_end in occupied):
                traces.append(self._trace(group, group[0].canonical, 0.0, False, "overlaps stronger semantic group"))
                continue
            canonical = self._best_canonical(group)
            original_span = " ".join(item.sentence for item in group)
            lexical = self.validator.similarity(original_span, canonical)["semantic_similarity"]
            confidence = max(lexical, self._structured_confidence(group))
            if confidence < 0.78:
                traces.append(self._trace(group, canonical, confidence, False, "semantic confidence below threshold"))
                continue
            selected.append((start, end, canonical, group))
            occupied.append((start, end))
            traces.append(self._trace(group, canonical, confidence, True, "repeated semantic claims canonicalized"))

        for key, group in groups.items():
            if key in traced_keys:
                continue
            if len(group) == 1 and self._has_contradiction(key, groups):
                traces.append(self._trace(group, group[0].canonical, 0.0, False, "opposite semantic claim present"))

        if not selected:
            return text, traces
        optimized = text
        for start, end, canonical, _ in sorted(selected, reverse=True):
            optimized = optimized[:start] + canonical + optimized[end:]
        optimized = re.sub(r"\s{2,}", " ", optimized).strip()
        optimized = re.sub(r"(?<=[.!?])(?=[A-Z])", " ", optimized)
        if count_tokens(optimized) >= count_tokens(text):
            for trace in traces:
                if trace["accepted"]:
                    trace["accepted"] = False
                    trace["rejected_reason"] = "final output was not shorter"
            return text, traces
        return optimized, traces

    def _claims(self, text: str) -> list[SemanticClaim]:
        claims: list[SemanticClaim] = []
        for match in re.finditer(r"[^\s\n].*?(?:[.!?](?=\s+|\s*$)|$)", text):
            sentence = match.group(0).strip()
            if not sentence:
                continue
            parsed = self._parse_sentence(sentence, match.start(), match.end())
            if parsed:
                claims.append(parsed)
        return claims

    def _parse_sentence(self, sentence: str, start: int, end: int) -> SemanticClaim | None:
        clean = _clean(sentence)
        return (
            self._parse_identity(sentence, clean, start, end)
            or self._parse_need(sentence, clean, start, end)
            or self._parse_event(sentence, clean, start, end)
            or self._parse_context_quality(sentence, clean, start, end)
            or self._parse_preference_comparative(sentence, clean, start, end)
            or self._parse_comparative(sentence, clean, start, end)
            or self._parse_requirement(sentence, clean, start, end)
            or self._parse_capability(sentence, clean, start, end)
            or self._parse_quality(sentence, clean, start, end)
        )

    def _parse_preference_comparative(self, sentence: str, clean: str, start: int, end: int) -> SemanticClaim | None:
        clean = clean.strip(" .!?")
        match = re.search(rf"^(?P<subject>{NAME})\s+is\s+(?P<neg>not\s+)?better\s+than\s+(?P<target>{NAME})$", clean, flags=re.IGNORECASE)
        if not match:
            return None
        subject = _comparison_name(match.group("subject"))
        target = _comparison_name(match.group("target"))
        if subject.lower() == target.lower():
            return None
        if match.group("neg"):
            subject, target = target, subject
        canonical = f"{subject} is better than {target}."
        return SemanticClaim(sentence, "preference", subject, "better_than", target, "better", canonical, start, end)

    def _parse_identity(self, sentence: str, clean: str, start: int, end: int) -> SemanticClaim | None:
        clean = clean.strip(" .!?")
        patterns = [
            rf"^my name is (?P<name>{NAME})$",
            rf"^(?P<name>{NAME}) is my name$",
            rf"^(?P<name>{NAME}) is (?:the )?name (?:which|that) i (?:got|have|was given)$",
            rf"^i(?: am|'m) (?P<name>{NAME})$",
            rf"^i go by (?P<name>{NAME})$",
            rf"^(?P<name>{NAME}) is what i am called$",
        ]
        for pattern in patterns:
            match = re.search(pattern, clean, flags=re.IGNORECASE)
            if not match:
                continue
            name = _person_name(match.group("name"))
            if not name:
                return None
            canonical = f"My name is {name}."
            return SemanticClaim(sentence, "identity", "speaker", "name_is", name, "identity", canonical, start, end)
        return None

    def _parse_need(self, sentence: str, clean: str, start: int, end: int) -> SemanticClaim | None:
        clean = clean.strip(" .!?")
        patterns = [
            r"^(?P<actor>we|i|you|they|he|she|it)\s+(?P<modal>might|may|could|would|should|must|will)?\s*need\s+(?P<object>.+?)\s+to\s+(?P<purpose>.+)$",
            r"^to\s+(?P<purpose>.+?)\s+(?P<actor>we|i|you|they|he|she|it)\s+(?P<modal>might|may|could|would|should|must|will)?\s*need\s+(?P<object>.+)$",
            r"^(?P<object>.+?)\s+is what\s+(?P<actor>we|i|you|they|he|she|it)\s+(?P<modal>might|may|could|would|should|must|will)?\s*need\s+to\s+(?P<purpose>.+)$",
        ]
        for pattern in patterns:
            match = re.search(pattern, clean, flags=re.IGNORECASE)
            if not match:
                continue
            actor = match.group("actor").lower()
            modal = (match.groupdict().get("modal") or "").lower().strip()
            obj = _semantic_phrase(match.group("object"))
            purpose = _semantic_phrase(match.group("purpose"))
            if not obj or not purpose:
                return None
            relation = "need"
            prefix = f"{actor} {modal} need" if modal else f"{actor} need"
            canonical = f"{prefix.capitalize()} {_article_object(obj)} to {purpose}."
            return SemanticClaim(sentence, "need", actor, relation, purpose, obj, canonical, start, end)
        return None

    def _parse_event(self, sentence: str, clean: str, start: int, end: int) -> SemanticClaim | None:
        clean = clean.strip(" .!?")
        clean = re.sub(r"^(?:you know|as noted|again|basically|actually)\s+", "", clean, flags=re.IGNORECASE)
        patterns = [
            r"^(?P<actor>we|i|you|they|he|she)\s+(?P<modal>might|may|could|would|should|must|will)?\s*have\s+(?P<event>[a-z][a-z'-]+(?:\s+[a-z][a-z'-]+){0,5})\s+(?P<context>before|earlier|previously|already)$",
            r"^(?P<context>before|earlier|previously|already)\s+(?P<actor>we|i|you|they|he|she)\s+(?P<modal>might|may|could|would|should|must|will)?\s*have\s+(?P<event>[a-z][a-z'-]+(?:\s+[a-z][a-z'-]+){0,5})$",
        ]
        for pattern in patterns:
            match = re.search(pattern, clean, flags=re.IGNORECASE)
            if not match:
                continue
            actor = match.group("actor").lower()
            modal = (match.groupdict().get("modal") or "").lower().strip()
            event = _semantic_phrase(match.group("event"))
            context = _semantic_phrase(match.group("context"))
            if not event or not context:
                return None
            relation = "have_event"
            prefix = f"{actor} {modal} have" if modal else f"{actor} have"
            canonical = f"{prefix.capitalize()} {event} {context}."
            return SemanticClaim(sentence, "event", actor, relation, context, event, canonical, start, end)
        return None

    def _parse_context_quality(self, sentence: str, clean: str, start: int, end: int) -> SemanticClaim | None:
        clean = clean.strip(" .!?")
        match = re.search(
            r"^(?P<context>(?:when|while|after|before|during)\s+.+?)\s+it\s+(?:is|was|seems|feels|looks)\s+(?P<quality>not expensive|simple|easy|less hard|not hard|hard|difficult|complex|fast|quick|slow|reliable|stable|unreliable|expensive|costly|cheap|affordable|secure|safe)$",
            clean,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        context = _semantic_phrase(match.group("context"))
        quality = _quality(match.group("quality"))
        if not context or not quality:
            return None
        canonical = f"{context.capitalize()} it was {quality}."
        return SemanticClaim(sentence, "context_quality", context, "was", "", quality, canonical, start, end)

    def _parse_comparative(self, sentence: str, clean: str, start: int, end: int) -> SemanticClaim | None:
        subject = _subject_before(clean, r"is|seems|feels|looks|remains")
        target_match = re.search(rf"\b(?:than|versus|vs\.?|over|compared to)\s+(?P<target>{NAME})\b", clean, flags=re.IGNORECASE)
        quality_match = re.search(r"\b(less hard|not hard|less easy|not easy|more expensive|easier|easy|harder|hard|faster|fast|quick|slower|slow|cheaper|cheap|affordable|expensive|costly)\b", clean, flags=re.IGNORECASE)
        if not subject or not target_match or not quality_match:
            fallback = re.search(rf"^(?P<subject>{NAME}).*?\b(?P<quality>easy|easier|less hard|not hard)\b.*?\b(?P<target>[A-Z][A-Z0-9-]{{1,}})\b", clean, flags=re.IGNORECASE)
            if not fallback:
                return None
            subject = fallback.group("subject")
            target = fallback.group("target")
            quality_value = fallback.group("quality")
        else:
            target = target_match.group("target")
            quality_value = quality_match.group(1)
        if not _name_like(subject) or not _name_like(target):
            return None
        subject = _comparison_name(subject)
        target = _comparison_name(target)
        quality = _quality(quality_value)
        if quality == "hard":
            subject, target = target, subject
            quality = "easy"
        relation = "comparative"
        canonical_quality = _comparative_word(quality)
        canonical = f"{subject} is {canonical_quality} than {target}."
        return SemanticClaim(sentence, "comparative", subject, relation, target, quality, canonical, start, end)

    def _parse_quality(self, sentence: str, clean: str, start: int, end: int) -> SemanticClaim | None:
        subject = _subject_before(clean, r"is|seems|feels|looks|remains")
        quality_match = re.search(r"\b(simple|easy|less hard|not hard|hard|difficult|complex|fast|quick|slow|reliable|stable|unreliable|expensive|costly|cheap|affordable|secure|safe)\b", clean, flags=re.IGNORECASE)
        if not subject or not quality_match or not _name_like(subject):
            return None
        subject = _title_subject(subject)
        quality = _quality(quality_match.group(1))
        canonical = f"{subject} is {quality}."
        return SemanticClaim(sentence, "quality", subject, "is", "", quality, canonical, start, end)

    def _parse_requirement(self, sentence: str, clean: str, start: int, end: int) -> SemanticClaim | None:
        low = clean.lower()
        if not any(word in low for word in REQUIREMENT_WORDS):
            return None
        subject = _subject_before(clean, r"must|requires?|required|needs?|should|shall")
        target_match = re.search(r"\b(?P<modal>must|requires?|required|needs?|should|shall)\b\s+(?P<target>[a-z][a-z0-9 ,/-]{2,140})", clean, flags=re.IGNORECASE)
        if not subject or not target_match or not _name_like(subject):
            return None
        subject = _title_subject(subject)
        modal = target_match.group("modal").lower()
        raw_target = _trim_requirement_target(target_match.group("target"))
        if modal in {"requires", "require", "required", "needs", "need"} or raw_target.startswith(("require ", "requires ", "required ")):
            target = _normalize_requirement_target(raw_target)
            canonical = f"{subject} must require {target}."
        else:
            target = raw_target
            canonical = f"{subject} must {target}."
        return SemanticClaim(sentence, "requirement", subject, "must", target, "required", canonical, start, end)

    def _parse_capability(self, sentence: str, clean: str, start: int, end: int) -> SemanticClaim | None:
        low = clean.lower()
        if not any(word in low for word in CAPABILITY_WORDS):
            return None
        negated = any(word in low for word in NEGATION_WORDS)
        subject = _subject_before(clean, r"can|supports?|allows?|able to")
        target_match = re.search(r"\b(?:can|supports?|allows?|able to)\b\s+(?P<target>[a-z][a-z0-9 -]{2,60})", clean, flags=re.IGNORECASE)
        if not subject or not target_match or not _name_like(subject):
            return None
        subject = _title_subject(subject)
        target = _trim_target(target_match.group("target"))
        relation = "cannot" if negated else "can"
        canonical = f"{subject} {'cannot' if negated else 'can'} {target}."
        return SemanticClaim(sentence, "capability", subject, relation, target, "negative" if negated else "positive", canonical, start, end)

    def _attach_self_comparisons(self, claims: list[SemanticClaim], groups: dict[tuple[str, str, str, str, str], list[SemanticClaim]]) -> None:
        for claim in claims:
            if claim.claim_type != "comparative" or claim.subject.lower() != claim.target.lower():
                continue
            for key in list(groups):
                claim_type, subject, relation, target, polarity = key
                if claim_type == "comparative" and subject == claim.subject.lower() and polarity == claim.polarity and target != subject:
                    groups[key].append(SemanticClaim(claim.sentence, claim.claim_type, claim.subject, claim.relation, target.upper(), claim.polarity, claim.canonical, claim.start, claim.end))
                    break

    def _attach_quality_to_comparatives(self, claims: list[SemanticClaim], groups: dict[tuple[str, str, str, str, str], list[SemanticClaim]]) -> None:
        for claim in claims:
            if claim.claim_type != "quality":
                continue
            for key in list(groups):
                claim_type, subject, relation, target, polarity = key
                if claim_type == "comparative" and subject == claim.subject.lower() and polarity == claim.polarity:
                    groups[key].append(SemanticClaim(claim.sentence, "comparative", claim.subject, "comparative", target.upper(), claim.polarity, groups[key][0].canonical, claim.start, claim.end))
                    break

    def _has_contradiction(self, key: tuple[str, str, str, str, str], groups: dict[tuple[str, str, str, str, str], list[SemanticClaim]]) -> bool:
        claim_type, subject, relation, target, polarity = key
        opposite = OPPOSITES.get(polarity, f"not_{polarity}")
        if (claim_type, subject, relation, target, opposite) in groups:
            return True
        if claim_type == "comparative" and (claim_type, target, relation, subject, polarity) in groups:
            return True
        if claim_type == "identity":
            return any(
                other_type == "identity"
                and other_subject == subject
                and other_relation == relation
                and other_target != target
                for other_type, other_subject, other_relation, other_target, _ in groups
            )
        return False

    def _best_canonical(self, group: list[SemanticClaim]) -> str:
        if group[0].claim_type in {"comparative", "preference"}:
            span = " ".join(item.sentence for item in group)
            subject = _display_comparison_name(group[0].subject)
            target = _display_comparison_name(group[0].target)
            comparative = _comparative_word(group[0].polarity)
            if re.search(r"\b(generally|often|considered|viewed|find|students)\b", span, re.IGNORECASE):
                return f"{subject} is generally considered {comparative} than {_object_display(target)}."
            return f"{subject} is {comparative} than {_object_display(target)}."
        if group[0].claim_type in {"need", "event"}:
            uncertain = [
                item.canonical for item in group
                if re.search(r"\b(might|may|could|would|should)\b", item.canonical, flags=re.IGNORECASE)
            ]
            if uncertain:
                return min(uncertain, key=lambda item: (count_tokens(item), len(item)))
        return min((item.canonical for item in group), key=lambda item: (count_tokens(item), len(item)))

    def _structured_confidence(self, group: list[SemanticClaim]) -> float:
        if len(group) >= 3:
            return 0.93
        return 0.86

    def _trace(self, group: list[SemanticClaim], canonical: str, confidence: float, accepted: bool, reason: str) -> dict:
        span = " ".join(item.sentence for item in group)
        return {
            "backend": "semantic_claim_compactor",
            "candidate_type": f"semantic_duplicate_{group[0].claim_type}_claim",
            "reason": reason,
            "span_text": span,
            "removed_span": " ".join(item.sentence for item in group[1:]) if accepted else None,
            "retained_span": canonical,
            "score": round(confidence, 3),
            "confidence": round(confidence, 3),
            "semantic_similarity": round(confidence, 3),
            "tokens_saved": max(0, count_tokens(span) - count_tokens(canonical)),
            "risk_flags": [],
            "accepted": accepted,
            "rejected_reason": None if accepted else reason,
        }


def _clean(sentence: str) -> str:
    clean = sentence.strip()
    clean = re.sub(r"\b(" + "|".join(re.escape(word) for word in sorted(FILLER, key=len, reverse=True)) + r")\b", " ", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s+", " ", clean)
    return clean


def _quality(value: str) -> str:
    value = value.lower().strip()
    return QUALITY_CANONICAL.get(value, value)


def _comparative_word(quality: str) -> str:
    if quality == "easy":
        return "easier"
    if quality == "hard":
        return "harder"
    if quality == "fast":
        return "faster"
    if quality == "slow":
        return "slower"
    if quality == "cheap":
        return "cheaper"
    if quality == "expensive":
        return "more expensive"
    return quality


def _title_subject(value: str) -> str:
    words = [word for word in value.split() if word.lower() not in {"the", "a", "an", "it"}]
    return " ".join(words).strip()


def _comparison_name(value: str) -> str:
    value = re.split(r"\b(?:for|because|when|if|and|but|or)\b", value, maxsplit=1, flags=re.IGNORECASE)[0]
    return " ".join(value.split()).upper()


def _trim_target(value: str) -> str:
    value = re.split(r"\b(?:because|when|if|and|but|or)\b", value, maxsplit=1, flags=re.IGNORECASE)[0]
    return value.strip(" .,!?:;").lower()


def _trim_requirement_target(value: str) -> str:
    value = re.split(r"\b(?:because|when|if|but|before|after)\b", value, maxsplit=1, flags=re.IGNORECASE)[0]
    return value.strip(" .,!?:;")


def _normalize_requirement_target(value: str) -> str:
    value = _trim_target(value)
    value = re.sub(r"^(?:require|required|requires|need|needs)\s+", "", value, flags=re.IGNORECASE)
    return value.strip()


def _person_name(value: str) -> str | None:
    value = " ".join(value.strip(" .,!?:;").split())
    if not value:
        return None
    if re.search(r"\b(and|but|or|because|when|if|with|for|from|to)\b", value, flags=re.IGNORECASE):
        return None
    words = value.split()
    if len(words) > 3:
        return None
    blocked = {"name", "person", "someone", "thing", "it", "me", "you", "him", "her", "them"}
    if any(word.lower() in blocked for word in words):
        return None
    return value


def _semantic_phrase(value: str) -> str:
    value = re.sub(r"\b(?:really|very|extremely|quite|basically|actually|just|for some reason)\b", " ", value, flags=re.IGNORECASE)
    value = value.strip(" .,!?:;")
    value = re.sub(r"\s+", " ", value)
    return value.lower()


def _article_object(value: str) -> str:
    if re.fullmatch(r"[a-z][a-z'-]+", value) and value not in {"approval", "access", "help", "support"}:
        if value[0] in "aeiou":
            return f"an {value}"
        return f"a {value}"
    return value


def _subject_before(text: str, relation_pattern: str) -> str | None:
    match = re.search(rf"^\s*(?P<subject>.+?)\s+(?:{relation_pattern})\b", text, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group("subject").strip(" ,;:")


def _name_like(value: str) -> bool:
    value = value.strip()
    if not value:
        return False
    words = value.split()
    if len(words) > 4:
        return False
    return any(word[:1].isupper() or word.isupper() for word in words)


def _display_comparison_name(value: str) -> str:
    words = value.split()
    if words and words[0].lower() == "the":
        return "The " + " ".join(word.upper() for word in words[1:])
    return " ".join(word.upper() for word in words)


def _object_display(value: str) -> str:
    if value.startswith("The "):
        return "the " + value[4:]
    return value
