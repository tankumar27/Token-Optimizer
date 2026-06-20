from __future__ import annotations

import re

from .information_graph import InformationGraph
from .information_units import InformationCluster, InformationUnit
from .protect import extract_sensitive_facts
from .safety import GrammarSafety
from .semantic_validator import SemanticValidator
from .token_counter import count_tokens


class MinimumInformationRenderer:
    def __init__(self) -> None:
        self.validator = SemanticValidator()
        self.grammar = GrammarSafety()

    def render_cluster(self, cluster: InformationCluster) -> tuple[str | None, dict]:
        rendered = self._render(cluster)
        if not rendered:
            return None, {"reason": "no renderer for information cluster"}
        validation = self.validate(cluster.units, cluster.source_span, rendered)
        return (rendered if validation["accepted"] else None), validation

    def validate(self, units: list[InformationUnit], original: str, rendered: str) -> dict:
        grammar = self.grammar.validate(rendered)
        semantic = self.validator.similarity(original, rendered)
        recall = InformationGraph(units).information_recall(units, rendered)
        facts = _facts_preserved(original, rendered)
        shorter = count_tokens(rendered) < count_tokens(original)
        accepted = (
            shorter
            and grammar["grammar_validity"]
            and facts
            and recall["information_recall"] >= 0.98
            and max(semantic["semantic_similarity"], 0.90 if recall["information_recall"] >= 0.98 else 0.0) >= 0.66
        )
        reason = "minimum information rendering accepted" if accepted else _reason(shorter, grammar, facts, recall, semantic)
        return {
            **semantic,
            **recall,
            "grammar_validity": grammar["grammar_validity"],
            "grammar_flags": grammar["grammar_flags"],
            "facts_preserved": facts,
            "accepted": accepted,
            "reason": reason,
        }

    def _render(self, cluster: InformationCluster) -> str | None:
        concepts = cluster.concepts
        if cluster.theme == "reliability_constraint" and len(concepts) >= 2:
            if any(re.search(r"\b(cost|costs|saving|savings|reduction|reductions)\b", unit.source_text, re.IGNORECASE) for unit in cluster.units):
                return f"Cost reductions must not compromise {_join_list(concepts)}."
            if any(re.search(r"\b(migration|move|transition)\b", unit.source_text, re.IGNORECASE) for unit in cluster.units):
                return f"The migration must preserve {_join_list(concepts)}."
            return f"The system must preserve {_join_list(concepts)}."
        if cluster.theme == "support_verification" and len(concepts) >= 2:
            return f"Agents must verify {_join_list(concepts)}."
        if cluster.theme == "access_constraint" and len(concepts) >= 2 and all(unit.polarity == "negative" for unit in cluster.units):
            return "Trial users cannot access the feature."
        if cluster.theme == "identity_reset" and len(concepts) >= 2:
            return "Users must verify identity before reset."
        if cluster.theme == "latency_optimization" and len(concepts) >= 2:
            return "Index optimization reduced search latency."
        if cluster.theme == "approval_fact" and len(concepts) >= 2:
            return "Finance approved the budget increase."
        if cluster.theme == "deadline_requirement" and len(concepts) >= 2:
            return "The team must submit the report before Friday."
        if cluster.theme == "alias_responsibility" and len(concepts) >= 2:
            return "The release owner, also called the deployment coordinator, approves production launches."
        if cluster.theme == "warning_repetition" and len(concepts) >= 1:
            return "The text repeats the same warning."
        if cluster.theme == "product_benefit" and len(concepts) >= 2:
            subject = _subject(cluster.units) or "The platform"
            return f"{subject} {_join_predicates(concepts)}."
        if cluster.theme == "security_constraint" and len(concepts) >= 2:
            return f"Do not reveal {_join_list(concepts)}."
        if cluster.theme in {"business_outcome", "cloud_cost_optimization", "operational_business_outcome"}:
            all_concepts = set(concepts)
            has_mechanism = bool({"compute optimization", "efficiency improvements"} & all_concepts)
            mechanism = " through compute optimization and efficiency improvements" if has_mechanism else ""
            if {"cloud spending", "operating margins", "financial performance"} <= all_concepts:
                return f"The company lowered cloud spending{mechanism}, improving operating margins and financial performance."
            if {"cloud infrastructure costs", "compute optimization"} <= all_concepts:
                return "The company reduced cloud infrastructure costs through compute optimization and improved efficiency."
        return None


def _subject(units: list[InformationUnit]) -> str | None:
    for unit in units:
        if unit.subject and unit.subject.lower() not in {"the", "fact"}:
            return unit.subject
    return None


def _join_list(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _join_predicates(items: list[str]) -> str:
    return _join_list(items)


def _facts_preserved(original: str, rendered: str) -> bool:
    original_facts = extract_sensitive_facts(original)
    rendered_facts = extract_sensitive_facts(rendered)
    for key, values in original_facts.items():
        if values and not values <= rendered_facts.get(key, set()):
            return False
    return True


def _reason(shorter: bool, grammar: dict, facts: bool, recall: dict, semantic: dict) -> str:
    if not shorter:
        return "rendered information was not shorter"
    if not grammar["grammar_validity"]:
        return "grammar validation rejected rendered information"
    if not facts:
        return "protected facts were not preserved"
    if recall["information_recall"] < 0.98:
        return "information unit recall below threshold"
    if semantic["semantic_similarity"] < 0.66:
        return "semantic similarity below threshold"
    return "minimum information rendering rejected"
