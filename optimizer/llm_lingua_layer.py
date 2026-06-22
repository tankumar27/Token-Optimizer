from __future__ import annotations

import re
import os
from dataclasses import dataclass
from functools import lru_cache

import httpx

from app.config import get_settings
from .cheap_layer import (
    canonical_fact_keys,
    event_signatures,
    risk_keys,
    split_sentences,
    state_signatures,
    validation_gate,
)
from .semantic_validator import SemanticValidator
from .token_counter import count_tokens


BACKEND_NAME = "llm_lingua_backend"
PLACEHOLDER_RE = re.compile(r"__PROTECTED_\d+__")
LOW_INFORMATION_REPEAT_WORDS = {
    "please",
    "kindly",
    "really",
    "very",
    "basically",
    "actually",
    "just",
    "simply",
}
LOW_INFORMATION_ENTITY_FALSE_POSITIVES = LOW_INFORMATION_REPEAT_WORDS | {
    "the",
    "a",
    "an",
    "this",
    "that",
    "these",
    "those",
    "it",
    "if",
    "do",
    "use",
    "return",
    "system",
    "task",
    "context",
    "background",
    "long",
    "final",
}


@dataclass
class LinguaCandidate:
    source: str
    text: str
    generator_trace: dict


def llm_lingua_backend(text: str, level: str, provider: str, mode: str) -> tuple[str, list[dict]]:
    """Second-stage prompt compression with a strict safety fallback.

    This layer intentionally runs after the deterministic cheap layer. It can use
    a model as a LLMLingua-style generator, but model output is never trusted
    directly: the original cheap-layer output remains the fallback unless every
    invariant check passes.
    """

    candidates: list[LinguaCandidate] = []
    pre_rejected_traces: list[dict] = []
    local_text, local_trace = _deterministic_token_cleanup(text)
    if local_text != text:
        candidates.append(LinguaCandidate("deterministic_token_cleanup", local_text, local_trace))

    settings = get_settings()
    input_tokens = count_tokens(text)
    render_model_disabled = (
        os.getenv("ENABLE_RENDER_LLM_LINGUA_MODEL") == "0"
        or (bool(os.getenv("RENDER")) and not settings.enable_render_llm_lingua_model)
    )
    model_allowed = (
        settings.enable_llm_lingua
        and not render_model_disabled
        and input_tokens <= settings.llm_lingua_max_input_tokens
    )
    if settings.enable_llm_lingua and render_model_disabled:
        pre_rejected_traces.append(_trace(
            original=text,
            candidate=text,
            source=settings.llm_lingua_backend,
            accepted=False,
            reason="LLMLingua model skipped on Render; set ENABLE_RENDER_LLM_LINGUA_MODEL=1 to enable heavy local model loading",
            validation={},
            generator_trace={
                "generator": settings.llm_lingua_backend,
                "accepted": False,
                "reason": "render_model_skip",
                "input_tokens": input_tokens,
            },
        ))
    elif settings.enable_llm_lingua and not model_allowed:
        pre_rejected_traces.append(_trace(
            original=text,
            candidate=text,
            source=settings.llm_lingua_backend,
            accepted=False,
            reason=f"LLMLingua skipped for large prompt ({input_tokens} tokens > {settings.llm_lingua_max_input_tokens} token limit)",
            validation={},
            generator_trace={
                "generator": settings.llm_lingua_backend,
                "accepted": False,
                "reason": "large_prompt_skip",
                "input_tokens": input_tokens,
                "max_input_tokens": settings.llm_lingua_max_input_tokens,
            },
        ))
    if model_allowed:
        model_text, model_trace = _model_candidate(text, level, provider, mode)
        if model_text and model_text != text:
            candidates.append(LinguaCandidate(model_trace["generator"], model_text, model_trace))
        elif model_trace:
            pre_rejected_traces.append(_trace(
                original=text,
                candidate=text,
                source=model_trace.get("generator", settings.llm_lingua_backend),
                accepted=False,
                reason=model_trace.get("reason", "model did not return a shorter candidate"),
                validation={},
                generator_trace=model_trace,
            ))
        if settings.llm_lingua_backend.lower().strip() in {"llmlingua2", "local", "llmlingua"}:
            candidates.extend(_llmlingua2_segment_candidates(text, level))

    if not candidates:
        return text, pre_rejected_traces

    evaluated: list[tuple[int, LinguaCandidate, dict]] = []
    rejected_traces: list[dict] = [*pre_rejected_traces]
    for candidate in candidates:
        accepted, reason, validation = validate_llm_lingua_candidate(text, candidate.text, level)
        trace = _trace(text, candidate.text, candidate.source, accepted, reason, validation, candidate.generator_trace)
        if accepted:
            evaluated.append((count_tokens(candidate.text), candidate, validation))
        else:
            rejected_traces.append(trace)

    if not evaluated:
        return text, rejected_traces

    _, winner, validation = min(evaluated, key=lambda item: item[0])
    accepted_trace = _trace(
        text,
        winner.text,
        winner.source,
        True,
        "LLMLingua layer accepted shorter validated prompt",
        validation,
        winner.generator_trace,
    )
    return winner.text, [*rejected_traces, accepted_trace]


def validate_llm_lingua_candidate(original: str, candidate: str, level: str) -> tuple[bool, str, dict]:
    original_tokens = count_tokens(original)
    candidate_tokens = count_tokens(candidate)
    validator = SemanticValidator()
    semantic = validator.similarity(original, candidate)
    entity = validator.entity_preservation(original, candidate)
    gate = validation_gate(original, candidate)
    original_placeholders = set(PLACEHOLDER_RE.findall(original))
    candidate_placeholders = set(PLACEHOLDER_RE.findall(candidate))

    reasons: list[str] = []
    if not candidate.strip():
        reasons.append("empty candidate")
    if candidate_tokens >= original_tokens:
        reasons.append("candidate is not shorter")
    missing_placeholders = sorted(original_placeholders - candidate_placeholders)
    added_placeholders = sorted(candidate_placeholders - original_placeholders)
    if missing_placeholders:
        reasons.append("protected placeholders missing")
    if added_placeholders:
        reasons.append("unexpected protected placeholders added")
    if not gate["passed"]:
        reasons.append("cheap-layer invariant gate failed")
    semantic_threshold = {"safe": 0.92, "balanced": 0.88, "aggressive": 0.82}.get(level, 0.88)
    if (
        semantic["fallback_used"]
        and gate["passed"]
        and not missing_placeholders
        and not added_placeholders
    ):
        semantic_threshold = min(semantic_threshold, 0.78)
    if (
        _same_content_without_low_information_tokens(original, candidate)
        and gate["passed"]
        and not missing_placeholders
        and not added_placeholders
    ):
        semantic_threshold = min(semantic_threshold, 0.78)
    if semantic["semantic_similarity"] < semantic_threshold:
        reasons.append("semantic similarity below LLMLingua threshold")
    missing_entities = [item for item in entity["entities_missing"] if item.lower() not in LOW_INFORMATION_ENTITY_FALSE_POSITIVES]
    if missing_entities:
        reasons.append("entities missing")
    if _has_unresolved_placeholder_spacing(candidate):
        reasons.append("protected placeholder spacing broken")

    validation = {
        "original_tokens": original_tokens,
        "candidate_tokens": candidate_tokens,
        "tokens_saved": max(0, original_tokens - candidate_tokens),
        "semantic_similarity": semantic["semantic_similarity"],
        "semantic_validator_used": semantic["validator_used"],
        "semantic_fallback_used": semantic["fallback_used"],
        "entity_preservation_score": entity["entity_preservation_score"],
        "entities_missing": entity["entities_missing"],
        "entities_missing_after_low_information_filter": missing_entities,
        "entity_validator_used": entity["validator_used"],
        "entity_fallback_used": entity["fallback_used"],
        "missing_placeholders": missing_placeholders,
        "added_placeholders": added_placeholders,
        "missing_fact_keys": sorted(canonical_fact_keys(original) - canonical_fact_keys(candidate)),
        "missing_state_signatures": sorted(state_signatures(original) - state_signatures(candidate)),
        "missing_event_signatures": sorted(event_signatures(original) - event_signatures(candidate)),
        "missing_risk_keys": sorted(risk_keys(original) - risk_keys(candidate)),
        "quality_gate": gate,
    }
    return not reasons, "; ".join(reasons) if reasons else "validated", validation


def _deterministic_token_cleanup(text: str) -> tuple[str, dict]:
    cleaned = text
    changes: list[dict] = []

    def collapse_repeat(match: re.Match) -> str:
        word = match.group("word")
        lowered = word.lower()
        if lowered not in LOW_INFORMATION_REPEAT_WORDS:
            return match.group(0)
        changes.append({
            "type": "adjacent_low_information_repeat",
            "removed_span": match.group(0),
            "retained_span": word,
        })
        return word

    cleaned = re.sub(
        r"\b(?P<word>[A-Za-z][A-Za-z'-]{1,})\b(?:\s+(?P=word)\b)+",
        collapse_repeat,
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\bPlease\s+kindly\s+(?=\w)",
        lambda match: _record_phrase(changes, match.group(0), "Please "),
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
    return cleaned, {
        "generator": "deterministic_token_cleanup",
        "reason": "removed adjacent low-information filler tokens",
        "changes": changes,
    }


def _record_phrase(changes: list[dict], removed: str, retained: str) -> str:
    changes.append({
        "type": "low_information_phrase_cleanup",
        "removed_span": removed,
        "retained_span": retained,
    })
    return retained


def _model_candidate(text: str, level: str, provider: str, mode: str) -> tuple[str | None, dict]:
    settings = get_settings()
    backend = settings.llm_lingua_backend.lower().strip()
    if backend in {"llmlingua2", "local", "llmlingua"}:
        return _llmlingua2_candidate(text, level)
    if backend == "gemini":
        return _gemini_candidate(text, level)
    return None, {
        "generator": backend or "unknown",
        "accepted": False,
        "reason": "unsupported LLMLingua backend",
        "provider": provider,
        "mode": mode,
    }


def _gemini_candidate(text: str, level: str) -> tuple[str | None, dict]:
    settings = get_settings()
    if not settings.gemini_api_key:
        return None, {
            "generator": "gemini_llm_lingua",
            "accepted": False,
            "reason": "GEMINI_API_KEY is not configured",
        }
    prompt = _compression_prompt(text, level)
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0,
            "topP": 0.1,
            "candidateCount": 1,
        },
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{settings.llm_lingua_model}:generateContent"
    try:
        with httpx.Client(timeout=settings.llm_lingua_timeout_seconds) as client:
            response = client.post(url, params={"key": settings.gemini_api_key}, json=payload)
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        return None, {
            "generator": "gemini_llm_lingua",
            "accepted": False,
            "reason": f"Gemini LLMLingua request failed: {type(exc).__name__}",
            "model": settings.llm_lingua_model,
        }
    text_out = _extract_gemini_text(data)
    return _strip_model_wrapping(text_out), {
        "generator": "gemini_llm_lingua",
        "accepted": bool(text_out),
        "reason": "Gemini generated candidate" if text_out else "Gemini returned empty candidate",
        "model": settings.llm_lingua_model,
    }


def _llmlingua2_candidate(text: str, level: str) -> tuple[str | None, dict]:
    settings = get_settings()
    rate = {"safe": 0.9, "balanced": 0.78, "aggressive": 0.65}.get(level, 0.78)
    force_tokens = _force_tokens_for_llmlingua2(text)
    try:
        compressor = _get_llmlingua2_compressor(settings.llm_lingua2_model, settings.llm_lingua2_device)
        result = compressor.compress_prompt(
            [text],
            rate=rate,
            force_tokens=force_tokens,
            force_reserve_digit=True,
            drop_consecutive=True,
            use_sentence_level_filter=False,
            use_context_level_filter=False,
            use_token_level_filter=True,
        )
        compressed = result.get("compressed_prompt", result) if isinstance(result, dict) else result
        if isinstance(compressed, list):
            compressed = "\n".join(str(item) for item in compressed)
        return str(compressed).strip(), {
            "generator": "llmlingua2",
            "accepted": True,
            "reason": "true LLMLingua2 package generated candidate",
            "model": settings.llm_lingua2_model,
            "device": settings.llm_lingua2_device,
            "rate": rate,
            "force_token_count": len(force_tokens),
            "package_result": _public_llmlingua2_result(result),
        }
    except Exception as exc:
        return None, {
            "generator": "llmlingua2",
            "accepted": False,
            "reason": f"LLMLingua2 package failed: {type(exc).__name__}",
            "model": settings.llm_lingua2_model,
            "device": settings.llm_lingua2_device,
        }


def _llmlingua2_segment_candidates(text: str, level: str) -> list[LinguaCandidate]:
    sentences = split_sentences(text)
    candidates: list[LinguaCandidate] = []
    block: list[str] = []
    for sentence in sentences + [""]:
        if sentence and _is_llmlingua2_eligible_prose(sentence):
            block.append(sentence)
            continue
        if len(block) >= 2 and count_tokens(" ".join(block)) >= 35:
            candidate = _compress_sentence_block(text, block, level)
            if candidate:
                candidates.append(candidate)
                if len(candidates) >= 2:
                    return candidates
        block = []
    return candidates


def _compress_sentence_block(text: str, block: list[str], level: str) -> LinguaCandidate | None:
    original_block = " ".join(block)
    if original_block not in text:
        return None
    compressed_block, trace = _llmlingua2_candidate(original_block, level)
    if not compressed_block or compressed_block == original_block:
        return None
    candidate_text = text.replace(original_block, compressed_block, 1)
    trace = {
        **trace,
        "generator": "llmlingua2_segment",
        "reason": "true LLMLingua2 compressed an eligible low-risk prose segment",
        "segment_original_tokens": count_tokens(original_block),
        "segment_compressed_tokens": count_tokens(compressed_block),
        "segment_original": original_block,
        "segment_compressed": compressed_block,
    }
    return LinguaCandidate("llmlingua2_segment", candidate_text, trace)


def _is_llmlingua2_eligible_prose(sentence: str) -> bool:
    low = sentence.lower()
    if len(sentence.split()) < 8:
        return False
    if PLACEHOLDER_RE.search(sentence):
        return False
    if canonical_fact_keys(sentence) or state_signatures(sentence) or event_signatures(sentence):
        return False
    if re.search(
        r"\b(do not|never|conflict|contradiction|claim|formula|must be preserved|"
        r"unavailable|excluded|blocked|failed|approved|denied|pending|active|inactive|"
        r"migrated|deleted|exposed|leaked|rotated|retained|tested|drained|paused|"
        r"reached|increased|decreased|refund|reserve|latency|rate|deadline|legal hold)\b",
        low,
    ):
        return False
    return True


@lru_cache(maxsize=2)
def _get_llmlingua2_compressor(model_name: str, device: str):
    try:
        from llmlingua import PromptCompressor  # type: ignore
    except Exception:
        raise RuntimeError("llmlingua package is not installed")
    return PromptCompressor(
        model_name=model_name,
        device_map=device,
        use_llmlingua2=True,
    )


def _force_tokens_for_llmlingua2(text: str) -> list[str]:
    tokens = ["\n", "?", ".", ":", ";"]
    tokens.extend(sorted(set(PLACEHOLDER_RE.findall(text))))
    return tokens[:100]


def _public_llmlingua2_result(result) -> dict:
    if not isinstance(result, dict):
        return {}
    public = {}
    for key in ["origin_tokens", "compressed_tokens", "ratio", "rate", "saving"]:
        if key in result:
            public[key] = result[key]
    return public


def _compression_prompt(text: str, level: str) -> str:
    ratio = {"safe": "about 90%", "balanced": "about 75-80%", "aggressive": "about 60-70%"}.get(level, "about 75-80%")
    return (
        "You are a LLMLingua-style prompt compressor. Compress by removing low-information tokens, repeated wording, "
        "and verbose filler. Do not summarize freely.\n\n"
        "Hard rules:\n"
        "- Return only the compressed prompt text.\n"
        "- Preserve every __PROTECTED_N__ placeholder exactly.\n"
        "- Preserve all IDs, numbers, dates, times, money, percentages, URLs, emails, formulas, code symbols, negation, uncertainty, warnings, and contradictions.\n"
        "- Preserve factual states and event predicates such as reached, paused, increased, decreased, migrated, excluded, blocked, failed, unavailable, quarantined, and watchlist.\n"
        "- Do not resolve conflicts or convert factual states into instructions.\n"
        f"- Target length: {ratio} of the input if safe; otherwise leave wording mostly unchanged.\n\n"
        "Prompt:\n"
        f"{text}"
    )


def _extract_gemini_text(data: dict) -> str:
    candidates = data.get("candidates") or []
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(part.get("text", "") for part in parts)


def _strip_model_wrapping(text: str | None) -> str | None:
    if text is None:
        return None
    stripped = text.strip()
    fence = re.fullmatch(r"```(?:text)?\s*([\s\S]*?)\s*```", stripped, re.I)
    if fence:
        stripped = fence.group(1).strip()
    return stripped


def _has_unresolved_placeholder_spacing(text: str) -> bool:
    return bool(re.search(r"__\s+PROTECTED|PROTECTED\s+_|__PROTECTED_\s+\d+|__PROTECTED_\d+\s+__", text))


def _same_content_without_low_information_tokens(original: str, candidate: str) -> bool:
    low_info = "|".join(sorted(re.escape(word) for word in LOW_INFORMATION_REPEAT_WORDS))
    def strip_low_info(text: str) -> str:
        stripped = re.sub(rf"\b(?:{low_info})\b", " ", text, flags=re.I)
        stripped = re.sub(r"\s+", " ", stripped)
        stripped = re.sub(r"\s+([.,;:!?])", r"\1", stripped)
        return stripped.strip().lower()

    return strip_low_info(original) == strip_low_info(candidate)


def _trace(
    original: str,
    candidate: str,
    source: str,
    accepted: bool,
    reason: str,
    validation: dict,
    generator_trace: dict,
) -> dict:
    before = count_tokens(original)
    after = count_tokens(candidate)
    return {
        "backend": BACKEND_NAME,
        "candidate_type": "second_stage_llm_lingua_compression",
        "accepted": accepted,
        "reason": reason if accepted else "LLMLingua candidate rejected by safety validator",
        "rejected_reason": None if accepted else reason,
        "span_text": original,
        "removed_span": original if accepted else None,
        "retained_span": candidate if accepted else None,
        "generated_compressed_prompt": candidate,
        "generator": source,
        "generator_trace": generator_trace,
        "tokens_saved": max(0, before - after),
        "score": round(max(0, before - after) / max(1, before), 3),
        "semantic_similarity": validation.get("semantic_similarity"),
        "entity_preservation_score": validation.get("entity_preservation_score"),
        "grammar_validity": not validation.get("quality_gate", {}).get("surface_quality_issues", []),
        "facts_preserved": not validation.get("missing_fact_keys"),
        "state_signatures_preserved": not validation.get("missing_state_signatures"),
        "event_signatures_preserved": not validation.get("missing_event_signatures"),
        "risk_keys_preserved": not validation.get("missing_risk_keys"),
        "validation": validation,
        "risk_flags": [] if accepted else [reason],
    }
