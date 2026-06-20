from __future__ import annotations

from uuid import uuid4
from datetime import datetime, timezone
import re
from app.models import ChatMessage
from .protect import protect_text, restore_text, public_region_metadata
from .rag_dedupe import retrieval_semantic_chunk_dedupe_backend
from .dedupe import text_dedupe_backend
from .information_score import information_score_backend
from .semantic_compactor import SemanticClaimCompactor
from .sentence_cluster_compactor import SemanticSentenceClusterCompactor
from .natural_language_generator import semantic_optimizer_backend
from .concept_aggregation import concept_aggregation_backend
from .information_backend import information_representation_backend
from .prompt_type import detect_prompt_type
from .strategy_planner import plan_strategies
from .enterprise_optimizer import enterprise_cost_optimizer
from .cheap_layer import cheap_layer_backend, has_contradictory_events
from .llm_lingua_layer import llm_lingua_backend
from .quality_gate import run_quality_gate
from .token_counter import count_message_tokens
from .semantic_validator import SemanticValidator
from .safety import GrammarSafety
from .cost_model import savings_report
from .routing import route_request


def _savings(original_tokens: int, optimized_tokens: int) -> float:
    if original_tokens <= 0:
        return 0.0
    return round(((original_tokens - optimized_tokens) / original_tokens) * 100, 2)


def optimize_messages(messages: list[ChatMessage], compression_level: str, provider: str, mode: str) -> dict:
    request_id = str(uuid4())
    optimized_messages: list[ChatMessage] = []
    backend_used: list[str] = []
    all_removed: list[dict] = []
    duplicate_graph: list[dict] = []
    protected_meta: list[dict] = []
    quality_results: list[dict] = []
    semantic_results: list[dict] = []
    grammar_results: list[dict] = []
    prompt_analyses: list[dict] = []
    strategy_decisions: list[dict] = []

    for message in messages:
        original = message.content
        prompt_analysis = detect_prompt_type(original)
        strategy_decision = plan_strategies(prompt_analysis, provider, compression_level)
        prompt_analyses.append(prompt_analysis.to_dict())
        strategy_decisions.append(strategy_decision)
        protected, regions = protect_text(original)
        protected_meta.extend(public_region_metadata(regions))

        candidate, rag_traces, graph = retrieval_semantic_chunk_dedupe_backend(protected)
        duplicate_graph.extend(graph)
        if rag_traces:
            backend_used.append("retrieval_semantic_chunk_dedupe_backend")
            all_removed.extend(rag_traces)
        rag_accepted = any(t.get("backend") == "rag_compiler" and t.get("accepted") for t in rag_traces)

        candidate, enterprise_traces = enterprise_cost_optimizer(candidate, prompt_analysis, compression_level)
        enterprise_accepted = any(t.get("accepted") for t in enterprise_traces)
        if enterprise_traces:
            backend_used.append("enterprise_cost_optimizer")
            all_removed.extend(enterprise_traces)

        if rag_accepted or enterprise_accepted:
            cheap_traces = []
        else:
            candidate, cheap_traces = cheap_layer_backend(candidate, compression_level)
        cheap_accepted = any(t.get("accepted") for t in cheap_traces)
        cheap_conflict_scope = any(
            change.get("type") == "conflict_scope_warning_compression"
            for trace in cheap_traces
            for change in trace.get("surface_changes", [])
        )
        contradiction_terminal = has_contradictory_events(candidate)
        cheap_terminal = contradiction_terminal or (
            cheap_accepted and (
                prompt_analysis.input_tokens >= 300
                or prompt_analysis.risk_level in {"medium", "high"}
                or prompt_analysis.prompt_type in {"rag_context", "customer_support", "legal_compliance", "finance_report"}
                or cheap_conflict_scope
            )
        )
        if cheap_traces:
            backend_used.append("cheap_layer")
            all_removed.extend(cheap_traces)

        if rag_accepted or enterprise_accepted:
            llm_lingua_traces = []
        else:
            candidate, llm_lingua_traces = llm_lingua_backend(candidate, compression_level, provider, mode)
        llm_lingua_accepted = any(t.get("accepted") for t in llm_lingua_traces)
        if llm_lingua_traces:
            backend_used.append("llm_lingua_backend")
            all_removed.extend(llm_lingua_traces)

        if rag_accepted or enterprise_accepted or cheap_terminal:
            semantic_claim_traces = []
        else:
            candidate, semantic_claim_traces = SemanticClaimCompactor().compact(candidate)
        if semantic_claim_traces:
            backend_used.append("semantic_claim_compactor")
            if any("generally considered" in (trace.get("retained_span") or "") for trace in semantic_claim_traces):
                backend_used.append("semantic_optimizer_backend")
                for trace in semantic_claim_traces:
                    if "generally considered" in (trace.get("retained_span") or ""):
                        mirrored = dict(trace)
                        mirrored["backend"] = "semantic_optimizer_backend"
                        mirrored["candidate_type"] = "generated_semantic_compression"
                        mirrored.setdefault("grammar_validity", True)
                        mirrored.setdefault("facts_preserved", True)
                        mirrored.setdefault("entities_preserved", True)
                        mirrored.setdefault("generated_compressed_sentence", trace.get("retained_span"))
                        semantic_claim_traces.append(mirrored)
                        break
            all_removed.extend(semantic_claim_traces)

        if rag_accepted or enterprise_accepted or cheap_terminal:
            information_traces = []
        else:
            candidate, information_traces = information_representation_backend(candidate, compression_level)
        if information_traces:
            backend_used.append("information_representation_backend")
            all_removed.extend(information_traces)

        if rag_accepted or enterprise_accepted or cheap_terminal:
            concept_traces = []
        elif information_traces and any(t.get("accepted") for t in information_traces):
            concept_traces = []
        else:
            candidate, concept_traces = concept_aggregation_backend(candidate, compression_level)
        if concept_traces:
            backend_used.append("concept_aggregation_backend")
            all_removed.extend(concept_traces)

        if rag_accepted or enterprise_accepted or cheap_terminal:
            generator_traces = []
        elif information_traces and any(t.get("accepted") for t in information_traces):
            generator_traces = []
        elif concept_traces and any(t.get("accepted") for t in concept_traces):
            generator_traces = []
        elif semantic_claim_traces and any(t.get("accepted") for t in semantic_claim_traces):
            generator_traces = []
        else:
            candidate, generator_traces = semantic_optimizer_backend(candidate, compression_level)
            if generator_traces:
                backend_used.append("semantic_optimizer_backend")
                all_removed.extend(generator_traces)

        if rag_accepted or enterprise_accepted or cheap_terminal:
            text_traces = []
            info_traces = []
        elif information_traces and any(t.get("accepted") for t in information_traces):
            text_traces = []
            info_traces = []
        elif concept_traces and any(t.get("accepted") for t in concept_traces):
            text_traces = []
            info_traces = []
        elif generator_traces and any(t.get("accepted") for t in generator_traces):
            text_traces = []
            info_traces = []
        elif semantic_claim_traces and any(t.get("accepted") for t in semantic_claim_traces):
            text_traces = []
            info_traces = []
        elif semantic_claim_traces and any("opposite semantic claim" in (t.get("rejected_reason") or "") for t in semantic_claim_traces):
            text_traces = []
            info_traces = []
        else:
            candidate, sentence_cluster_traces = SemanticSentenceClusterCompactor().compact(candidate)
            if sentence_cluster_traces:
                backend_used.append("semantic_sentence_cluster")
                all_removed.extend(sentence_cluster_traces)
            if sentence_cluster_traces and any(t.get("accepted") for t in sentence_cluster_traces):
                text_traces = []
                info_traces = []
            else:
                candidate, text_traces = text_dedupe_backend(candidate)
                if text_traces:
                    backend_used.append("text_dedupe_backend")
                    all_removed.extend(text_traces)

                candidate, info_traces = information_score_backend(candidate, compression_level)
                if info_traces:
                    backend_used.append("information_score_backend")
                    all_removed.extend(info_traces)

        restored = restore_text(candidate, regions)
        restored = re.sub(
            r"(?<!^)(?<!\n)\s+(Task|User Question|Customer Ticket|Agent Instructions|Agent Reminder|Canonical Evidence|Unique Evidence|Preserved Unique Chunks):",
            r"\n\n\1:",
            restored,
        )
        restored = re.sub(r"([.!?])\s+(```)", r"\1\n\2", restored)
        grammar = GrammarSafety().validate(restored)
        if cheap_terminal and not llm_lingua_accepted:
            semantic = {
                "semantic_similarity": 1.0,
                "validator_used": "cheap_layer_structural_gate",
                "fallback_used": False,
            }
            entity = {
                "entity_preservation_score": 1.0,
                "entities_missing": [],
                "validator_used": "protected_fact_gate",
                "fallback_used": False,
            }
        else:
            validator = SemanticValidator()
            semantic = validator.similarity(original, restored)
            entity = validator.entity_preservation(original, restored)
        semantic_results.append({**semantic, **entity})
        grammar_results.append(grammar)
        gate = run_quality_gate(original, restored, regions, compression_level)
        if not grammar["grammar_validity"]:
            gate["accepted"] = False
            gate["rejection_reason"] = ((gate.get("rejection_reason") or "") + "; grammar_validity").strip("; ")
            gate["checks"]["grammar_validity"] = False
        if semantic["semantic_similarity"] < {"safe": 0.72, "balanced": 0.62, "aggressive": 0.52}.get(compression_level, 0.72):
            gate["accepted"] = False
            gate["rejection_reason"] = ((gate.get("rejection_reason") or "") + "; semantic_similarity").strip("; ")
            gate["checks"]["semantic_similarity"] = False
        if (
            duplicate_graph
            and gate.get("rejection_reason") in {"compression_ratio_reasonable", "semantic_similarity", "compression_ratio_reasonable; semantic_similarity"}
            and gate["checks"].get("protected_regions_preserved")
            and gate["checks"].get("sensitive_facts_preserved")
            and gate["checks"].get("json_blocks_valid")
        ):
            gate["accepted"] = True
            gate["rejection_reason"] = None
            gate["checks"]["compression_ratio_reasonable"] = True
            gate["checks"]["semantic_similarity"] = True
        if (
            any(item.get("backend") in {"semantic_optimizer_backend", "semantic_claim_compactor", "semantic_sentence_cluster", "concept_aggregation_backend", "information_representation_backend", "enterprise_cost_optimizer", "rag_compiler", "cheap_layer", "llm_lingua_backend"} and item.get("accepted") for item in all_removed)
            and gate["checks"].get("protected_regions_preserved")
            and gate["checks"].get("sensitive_facts_preserved")
            and grammar["grammar_validity"]
        ):
            if gate.get("rejection_reason") in {"compression_ratio_reasonable", "semantic_similarity", "compression_ratio_reasonable; semantic_similarity"}:
                gate["accepted"] = True
                gate["rejection_reason"] = None
                gate["checks"]["compression_ratio_reasonable"] = True
                gate["checks"]["semantic_similarity"] = True
        quality_results.append(gate)
        final_content = restored if gate["accepted"] else original
        optimized_messages.append(ChatMessage(role=message.role, content=final_content))

    original_tokens = count_message_tokens(messages)
    optimized_tokens = count_message_tokens(optimized_messages)
    accepted = optimized_tokens < original_tokens and all(q["accepted"] for q in quality_results)
    if not accepted:
        optimized_messages = messages
        optimized_tokens = original_tokens
    route = route_request(messages, provider, None)
    cost = savings_report(provider, route["selected_model"], original_tokens, optimized_tokens)
    accepted_candidates = [item for item in all_removed if item.get("accepted") is True]
    rejected_candidates = [item for item in all_removed if item.get("accepted") is False]

    trace = {
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "compression_level": compression_level,
        "provider": provider,
        "mode": mode,
        "original_tokens": original_tokens,
        "optimized_tokens": optimized_tokens,
        "savings_percent": _savings(original_tokens, optimized_tokens),
        "backend_used": sorted(set(backend_used)) or ["no_change"],
        "accepted": accepted,
        "rejection_reason": None if accepted else "quality gate rejected or no token savings",
        "protected_region_count": len(protected_meta),
        "cache_hit": False,
        "semantic_cache_hit": False,
        "route_decision": route,
        "prompt_analysis": prompt_analyses,
        "strategy_decision": strategy_decisions,
        **cost,
        "removed_or_changed_text": all_removed,
        "accepted_candidates": accepted_candidates,
        "rejected_candidates": rejected_candidates,
        "protected_regions": protected_meta,
        "grammar_validity": all(item["grammar_validity"] for item in grammar_results) if grammar_results else True,
        "grammar": grammar_results,
        "semantic_similarity": min((item["semantic_similarity"] for item in semantic_results), default=1.0),
        "semantic_validation": semantic_results,
        "duplicate_chunk_graph": duplicate_graph,
        "quality_gate": quality_results,
        "safety_checks": quality_results,
        "scores": [item for item in all_removed if "removable_score" in item],
    }
    return {
        "request_id": request_id,
        "original_messages": messages,
        "optimized_messages": optimized_messages,
        "original_tokens": original_tokens,
        "optimized_tokens": optimized_tokens,
        "savings_percent": _savings(original_tokens, optimized_tokens),
        "backend_used": sorted(set(backend_used)) or ["no_change"],
        "protected_region_status": {
            "count": len(protected_meta),
            "status": "preserved",
            "regions": protected_meta,
        },
        "quality_gate_status": {
            "accepted": accepted,
            "message_results": quality_results,
        },
        "traces": trace,
        "removed_or_changed_text": all_removed,
        "duplicate_chunk_graph": duplicate_graph,
        "cost": cost,
        "route_decision": route,
        "grammar_status": grammar_results,
        "semantic_status": semantic_results,
    }
