from __future__ import annotations

from .information_graph import InformationGraph
from .minimum_renderer import MinimumInformationRenderer
from .proposition_extractor import PropositionExtractor
from .token_counter import count_tokens


BACKEND_NAME = "information_representation_backend"


def information_representation_backend(text: str, level: str = "balanced") -> tuple[str, list[dict]]:
    units = PropositionExtractor().extract(text)
    if len(units) < 2:
        return text, []
    graph = InformationGraph(units)
    renderer = MinimumInformationRenderer()
    traces: list[dict] = []
    replacements: list[tuple[int, int, str]] = []
    occupied: list[tuple[int, int]] = []

    for cluster in graph.compressible_clusters():
        if cluster.theme in {"business_outcome", "cloud_cost_optimization"} and _overlaps_existing_theme(cluster, occupied):
            continue
        if any(cluster.start < end and start < cluster.end for start, end in occupied):
            continue
        rendered, validation = renderer.render_cluster(cluster)
        accepted = rendered is not None
        trace = {
            "backend": BACKEND_NAME,
            "candidate_type": "minimum_information_representation",
            "cluster_theme": cluster.theme,
            "information_units": [unit.to_trace() for unit in cluster.units],
            "extracted_concepts": cluster.concepts,
            "span_text": cluster.source_span,
            "generated_minimum_representation": rendered,
            "retained_span": rendered,
            "removed_span": cluster.source_span if accepted else None,
            "semantic_similarity": validation.get("semantic_similarity"),
            "information_recall": validation.get("information_recall"),
            "tokens_saved": max(0, count_tokens(cluster.source_span) - count_tokens(rendered or cluster.source_span)),
            "grammar_validity": validation.get("grammar_validity"),
            "facts_preserved": validation.get("facts_preserved"),
            "accepted": accepted,
            "reason": validation.get("reason"),
            "rejected_reason": None if accepted else validation.get("reason"),
            "risk_flags": [] if accepted else [validation.get("reason", "rejected")],
            "validation": validation,
        }
        traces.append(trace)
        if accepted:
            replacements.append((cluster.start, cluster.end, rendered))
            occupied.append((cluster.start, cluster.end))

    if not replacements:
        return text, traces
    optimized = text
    for start, end, rendered in sorted(replacements, reverse=True):
        optimized = optimized[:start] + rendered + optimized[end:]
    optimized = _clean_text(optimized)
    if count_tokens(optimized) >= count_tokens(text):
        for trace in traces:
            if trace["accepted"]:
                trace["accepted"] = False
                trace["rejected_reason"] = "final output was not shorter"
        return text, traces
    return optimized, traces


def _clean_text(text: str) -> str:
    import re

    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([.!?,;:])", r"\1", text)
    text = re.sub(r"(?<=[.!?])(?=[A-Z])", " ", text)
    return text.strip()


def _overlaps_existing_theme(cluster, occupied: list[tuple[int, int]]) -> bool:
    return any(cluster.start <= end and start <= cluster.end for start, end in occupied)
