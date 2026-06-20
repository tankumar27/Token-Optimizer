from __future__ import annotations

from .candidates import CandidateGenerator
from .planner import CompressionPlanner, TraceRecorder
from .safety import SafetyGate
from .scorer import CandidateScorer
from .segmenter import Segmenter
from .semantic_validator import SemanticValidator
from .token_counter import count_tokens


def information_score_backend(text: str, level: str = "safe") -> tuple[str, list[dict]]:
    validator = SemanticValidator()
    segmenter = Segmenter()
    generator = CandidateGenerator()
    scorer = CandidateScorer()
    safety = SafetyGate(validator)
    planner = CompressionPlanner(level, validator)
    recorder = TraceRecorder()

    all_candidates = []
    for segment in segmenter.split(text):
        if segment.kind != "natural_language":
            continue
        for candidate in generator.generate(segment):
            all_candidates.append(scorer.score(safety.check(candidate, text), text))

    selected = planner.select(all_candidates, text)
    optimized = planner.apply(text, selected)
    if count_tokens(optimized) >= count_tokens(text):
        for candidate in selected:
            candidate.accepted = False
            candidate.rejected_reason = "final output was not shorter"
        optimized = text
    return optimized, recorder.record(all_candidates)
