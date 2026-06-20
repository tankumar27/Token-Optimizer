from __future__ import annotations

from collections import defaultdict
import re

from .information_units import InformationCluster, InformationUnit


class InformationGraph:
    def __init__(self, units: list[InformationUnit]) -> None:
        self.units = units

    def compressible_clusters(self) -> list[InformationCluster]:
        by_theme: dict[str, list[InformationUnit]] = defaultdict(list)
        for unit in self.units:
            if unit.theme == "protected_fact":
                continue
            theme = "operational_business_outcome" if unit.theme in {"business_outcome", "cloud_cost_optimization"} else unit.theme
            by_theme[theme].append(unit)
        clusters = [
            InformationCluster(theme, units)
            for theme, units in by_theme.items()
            if len(units) >= 2 and not self._contradictory(units)
        ]
        return sorted(clusters, key=lambda cluster: (len(cluster.units), cluster.end - cluster.start), reverse=True)

    def information_recall(self, original: list[InformationUnit], rendered: str) -> dict:
        rendered_low = rendered.lower()
        total = 0
        preserved = 0
        missing: list[str] = []
        for unit in original:
            for concept in unit.concepts:
                total += 1
                if concept.lower() in rendered_low or _concept_alias_present(concept, rendered_low):
                    preserved += 1
                else:
                    missing.append(concept)
            for quantity in unit.quantities:
                total += 1
                if quantity in rendered:
                    preserved += 1
                else:
                    missing.append(quantity)
        score = 1.0 if total == 0 else preserved / total
        return {"information_recall": round(score, 3), "missing_information_units": missing}

    def _contradictory(self, units: list[InformationUnit]) -> bool:
        joined = " ".join(unit.source_text.lower() for unit in units)
        if re.search(r"\b(costs?|spending|expenses|expenditures)\s+(?:decreased|reduced|lowered)\b", joined) and re.search(r"\b(costs?|spending|expenses|expenditures)\s+(?:increased|raised|rose)\b", joined):
            return True
        if any(unit.theme == "access_constraint" for unit in units) and {unit.polarity for unit in units} == {"positive", "negative"}:
            return True
        polarities = {unit.polarity for unit in units}
        return len(polarities) > 1 and any(unit.theme != "reliability_constraint" for unit in units)


def _concept_alias_present(concept: str, rendered_low: str) -> bool:
    aliases = {
        "cloud infrastructure costs": [r"cloud .*cost", r"cloud spending", r"infrastructure expenses"],
        "efficiency improvements": [r"efficiency", r"optimization", r"utilization"],
        "compute optimization": [r"compute optimization", r"compute efficiency"],
    }
    return any(re.search(pattern, rendered_low) for pattern in aliases.get(concept, []))
