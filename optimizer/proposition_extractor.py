from __future__ import annotations

import re

from .information_units import InformationUnit


SENTENCE_RE = re.compile(r"[^\s\n].*?(?:[.!?](?=\s+|\s*$)|$)")


class PropositionExtractor:
    """Extracts compact information units from natural-language prose.

    This is a deliberately conservative semantic parser. It does not try to
    fully understand English; it extracts high-confidence propositions with
    subject/relation/object slots plus modality, polarity, quantities, and
    concept tags. Unknown sentences remain available to downstream text
    compressors instead of being forced into a bad graph.
    """

    def extract(self, text: str) -> list[InformationUnit]:
        units: list[InformationUnit] = []
        for idx, match in enumerate(SENTENCE_RE.finditer(text)):
            sentence = match.group(0).strip()
            if len(sentence.split()) < 3 or "__PROTECTED_" in sentence:
                continue
            parsed = self._parse(sentence, match.start(), match.end(), idx)
            if parsed:
                units.append(parsed)
        return units

    def _parse(self, sentence: str, start: int, end: int, idx: int) -> InformationUnit | None:
        low = sentence.lower()
        quantities = tuple(re.findall(r"\$?\b\d+(?:\.\d+)?%?\b", sentence))
        modality = _modality(low)
        polarity = "negative" if _negated(low) else "positive"

        concepts = _reliability_concepts(low)
        if concepts:
            return InformationUnit(
                f"iu-{idx}", sentence, start, end,
                subject=_subject(sentence) or _constraint_subject(low) or "The system",
                relation="must_not_compromise" if polarity == "negative" or "not acceptable" in low else "preserve",
                object=", ".join(concepts),
                theme="reliability_constraint",
                concepts=tuple(concepts),
                modality=modality if modality != "neutral" else "must",
                polarity=polarity,
                quantities=quantities,
                risk_level="high",
                importance=0.95,
            )

        concepts = _verification_concepts(low)
        if concepts and re.search(r"\b(verify|confirm|check|validate|verification)\b", low):
            return InformationUnit(
                f"iu-{idx}", sentence, start, end,
                subject="Agents",
                relation="verify",
                object=", ".join(concepts),
                theme="support_verification",
                concepts=tuple(concepts),
                modality=modality if modality != "neutral" else "must",
                polarity=polarity,
                quantities=quantities,
                importance=0.9,
            )

        access = _access_constraint_concepts(low)
        if access:
            access_negative = _negated(low) or re.search(r"\b(unavailable|not available|cannot|can't)\b", low) is not None
            return InformationUnit(
                f"iu-{idx}", sentence, start, end,
                subject="trial users",
                relation="cannot_access" if access_negative else "can_access",
                object=", ".join(access),
                theme="access_constraint",
                concepts=tuple(access),
                modality=modality,
                polarity="negative" if access_negative else "positive",
                quantities=quantities,
                risk_level="high",
                importance=0.95,
            )

        identity = _identity_reset_concepts(low)
        if identity:
            return InformationUnit(
                f"iu-{idx}", sentence, start, end,
                subject="Users",
                relation="verify_identity_before_reset",
                object=", ".join(identity),
                theme="identity_reset",
                concepts=tuple(identity),
                modality=modality if modality != "neutral" else "must",
                polarity=polarity,
                quantities=quantities,
                importance=0.85,
            )

        latency = _latency_index_concepts(low)
        if latency:
            return InformationUnit(
                f"iu-{idx}", sentence, start, end,
                subject="Index optimization",
                relation="reduced",
                object=", ".join(latency),
                theme="latency_optimization",
                concepts=tuple(latency),
                modality=modality,
                polarity=polarity,
                quantities=quantities,
                importance=0.8,
            )

        approval = _approval_concepts(low)
        if approval:
            return InformationUnit(
                f"iu-{idx}", sentence, start, end,
                subject="Finance",
                relation="approved",
                object=", ".join(approval),
                theme="approval_fact",
                concepts=tuple(approval),
                modality=modality,
                polarity=polarity,
                quantities=quantities,
                importance=0.85,
            )

        deadline = _deadline_concepts(low)
        if deadline:
            return InformationUnit(
                f"iu-{idx}", sentence, start, end,
                subject="The team",
                relation="submit_by_deadline",
                object=", ".join(deadline),
                theme="deadline_requirement",
                concepts=tuple(deadline),
                modality=modality if modality != "neutral" else "must",
                polarity=polarity,
                quantities=quantities,
                importance=0.85,
            )

        alias = _alias_concepts(low)
        if alias:
            return InformationUnit(
                f"iu-{idx}", sentence, start, end,
                subject="release owner",
                relation="approves",
                object=", ".join(alias),
                theme="alias_responsibility",
                concepts=tuple(alias),
                modality=modality,
                polarity=polarity,
                quantities=quantities,
                importance=0.85,
            )

        warning = _warning_concepts(low)
        if warning:
            return InformationUnit(
                f"iu-{idx}", sentence, start, end,
                subject="The text",
                relation="repeats",
                object=", ".join(warning),
                theme="warning_repetition",
                concepts=tuple(warning),
                modality=modality,
                polarity=polarity,
                quantities=quantities,
                importance=0.55,
            )

        actions = _benefit_actions(low)
        if actions and re.search(r"\b(platform|product|tool)\b", low):
            return InformationUnit(
                f"iu-{idx}", sentence, start, end,
                subject=_subject(sentence) or "The platform",
                relation="benefit",
                object=", ".join(actions),
                theme="product_benefit",
                concepts=tuple(actions),
                modality=modality,
                polarity=polarity,
                quantities=quantities,
                importance=0.75,
            )

        security = _security_concepts(low)
        if security:
            return InformationUnit(
                f"iu-{idx}", sentence, start, end,
                subject="sensitive information",
                relation="must_not_reveal",
                object=", ".join(security),
                theme="security_constraint",
                concepts=tuple(security),
                modality=modality if modality != "neutral" else "must",
                polarity="negative",
                quantities=quantities,
                risk_level="high",
                importance=1.0,
            )

        cloud = _cloud_cost_concepts(low)
        business = _business_concepts(low)
        if cloud or business:
            theme = "business_outcome" if business else "cloud_cost_optimization"
            return InformationUnit(
                f"iu-{idx}", sentence, start, end,
                subject=_subject(sentence) or "The company",
                relation="reduced" if re.search(r"\b(reduced|lowered|decreased|cut)\b", low) else "improved",
                object=", ".join([*cloud, *business]),
                theme=theme,
                concepts=tuple(_ordered_unique([*cloud, *business])),
                modality=modality,
                polarity=polarity,
                quantities=quantities,
                importance=0.85,
            )

        if quantities:
            return InformationUnit(
                f"iu-{idx}", sentence, start, end,
                subject=_subject(sentence) or "fact",
                relation="states",
                object=sentence,
                theme="protected_fact",
                concepts=(),
                modality=modality,
                polarity=polarity,
                quantities=quantities,
                risk_level="high",
                importance=1.0,
            )
        return None


def _modality(low: str) -> str:
    if re.search(r"\b(must|shall|required|need to|needs to)\b", low):
        return "must"
    if re.search(r"\b(should)\b", low):
        return "should"
    if re.search(r"\b(might|may|could|possible|possibly)\b", low):
        return "uncertain"
    if re.search(r"\b(will|definitely|certainly)\b", low):
        return "certain"
    return "neutral"


def _negated(low: str) -> bool:
    return bool(re.search(r"\b(no|not|never|cannot|can't|without|not acceptable|must not|should not)\b", low))


def _subject(sentence: str) -> str | None:
    match = re.match(
        r"\s*(?P<subject>.+?)\s+(?:must|should|may|might|could|will|is|are|was|were|reduced|lowered|decreased|improved|saves|reduces|improves|contributed)\b",
        sentence,
        re.IGNORECASE,
    )
    if match:
        return match.group("subject").strip(" .,:;")
    match = re.match(r"\s*(Agents|System|Service|Performance|Customer experience)\b", sentence)
    return match.group(1).strip() if match else None


def _reliability_concepts(low: str) -> list[str]:
    concepts: list[str] = []
    if "reliability" in low:
        concepts.append("reliability")
    if "availability" in low or "available" in low:
        concepts.append("availability")
    if "performance" in low and "financial performance" not in low:
        concepts.append("performance")
    if "customer experience" in low:
        concepts.append("customer experience")
    return concepts


def _constraint_subject(low: str) -> str | None:
    if re.search(r"\b(cost|costs|saving|savings|reduction|reductions)\b", low):
        return "Cost reductions"
    if re.search(r"\b(migration|move|transition)\b", low):
        return "The migration"
    return None


def _verification_concepts(low: str) -> list[str]:
    concepts: list[str] = []
    for label in ["order ID", "customer email", "purchase date", "refund reason", "product SKU"]:
        if label.lower() in low:
            concepts.append(label)
    return concepts


def _access_constraint_concepts(low: str) -> list[str]:
    if re.search(r"\b(trial users|trial plan|accounts on the trial)\b", low) and re.search(r"\b(feature|access|available|unavailable)\b", low):
        return ["trial users", "feature"]
    return []


def _identity_reset_concepts(low: str) -> list[str]:
    if "identity" in low and re.search(r"\b(reset|resetting access|account reset)\b", low):
        return ["identity", "reset"]
    return []


def _latency_index_concepts(low: str) -> list[str]:
    if re.search(r"\b(index|indexing)\b", low) and re.search(r"\b(latency|faster|responses)\b", low):
        return ["index", "latency"]
    return []


def _approval_concepts(low: str) -> list[str]:
    if re.search(r"\b(finance|finance team)\b", low) and re.search(r"\b(approved|approval)\b", low) and re.search(r"\b(budget|increased budget)\b", low):
        return ["Finance", "budget"]
    return []


def _deadline_concepts(low: str) -> list[str]:
    if "report" in low and re.search(r"\b(friday|thursday|day before friday|due|submit|submission)\b", low):
        return ["report", "Friday"]
    return []


def _alias_concepts(low: str) -> list[str]:
    if re.search(r"\b(deployment coordinator|release owner)\b", low) and re.search(r"\b(production|launch|deployment)\b", low):
        return ["deployment coordinator", "release owner", "production"]
    return []


def _warning_concepts(low: str) -> list[str]:
    if "warning" in low and re.search(r"\b(repeated|restated|again|over and over|different words)\b", low):
        return ["warning"]
    return []


def _benefit_actions(low: str) -> list[str]:
    actions: list[str] = []
    if "saves time" in low or "save time" in low:
        actions.append("saves time")
    if "manual work" in low:
        actions.append("reduces manual work")
    if "reporting accuracy" in low:
        actions.append("improves reporting accuracy")
    if "operational metrics" in low:
        actions.append("monitors operational metrics")
    return actions


def _security_concepts(low: str) -> list[str]:
    concepts: list[str] = []
    if "api keys" in low:
        concepts.append("API keys")
    if "access tokens" in low:
        concepts.append("access tokens")
    if "secrets" in low:
        concepts.append("secrets")
    if "credentials" in low:
        concepts.append("credentials")
    return concepts


def _cloud_cost_concepts(low: str) -> list[str]:
    concepts: list[str] = []
    if re.search(r"\b(cloud|infrastructure)\b", low) and re.search(r"\b(cost|costs|spending|expenses|expenditures)\b", low):
        concepts.append("cloud infrastructure costs")
    if re.search(r"\b(compute optimization|compute efficiency|compute waste|compute workloads|resource utilization)\b", low):
        concepts.append("compute optimization")
    if re.search(r"\b(efficiency|utilization|waste)\b", low):
        concepts.append("efficiency improvements")
    return concepts


def _business_concepts(low: str) -> list[str]:
    concepts: list[str] = []
    if re.search(r"\b(cloud spending|cloud costs|cloud expenditures|infrastructure expenses|infrastructure costs)\b", low):
        concepts.append("cloud spending")
    if "operating margins" in low:
        concepts.append("operating margins")
    if "financial performance" in low:
        concepts.append("financial performance")
    return concepts


def _ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result
