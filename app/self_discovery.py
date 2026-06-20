from __future__ import annotations

import json
import random

from app.models import ChatMessage
from optimizer.pipeline import optimize_messages
from optimizer.proposition_extractor import PropositionExtractor


TEMPLATES = [
    [
        "System reliability must be preserved.",
        "Service availability must remain high.",
        "Performance degradation is not acceptable.",
        "Customer experience should not suffer.",
    ],
    [
        "Agents must verify order ID.",
        "Agents must verify customer email.",
        "Agents must verify purchase date.",
        "Agents must verify refund reason.",
    ],
    [
        "The platform saves time.",
        "The platform reduces manual work.",
        "The platform improves reporting accuracy.",
    ],
    [
        "The company reduced cloud spending.",
        "Lower infrastructure expenses improved operating margins.",
        "Reduced spending contributed positively to financial performance.",
    ],
]


def run_self_discovery(seed: int = 7, cases: int = 100) -> dict:
    random.seed(seed)
    extractor = PropositionExtractor()
    failures: list[dict] = []
    total_savings = 0.0
    for idx in range(cases):
        chosen = random.choice(TEMPLATES)
        sentences = chosen[:]
        random.shuffle(sentences)
        if random.random() < 0.25:
            sentences.append("However, customer acquisition costs increased by 11%.")
        text = " ".join(sentences)
        result = optimize_messages([ChatMessage(role="user", content=text)], "balanced", "dry-run", "dry-run")
        output = result["optimized_messages"][0].content
        before = extractor.extract(text)
        after = extractor.extract(output)
        before_concepts = {concept for unit in before for concept in unit.concepts} | {quantity for unit in before for quantity in unit.quantities}
        after_text = output.lower()
        missing = [concept for concept in sorted(before_concepts) if concept.lower() not in after_text and not _alias(concept, after_text)]
        if missing:
            failures.append({"case": idx, "missing": missing, "input": text, "output": output})
        total_savings += result["savings_percent"]
    return {
        "cases": cases,
        "failures": len(failures),
        "failure_rate": round(len(failures) / max(1, cases), 3),
        "average_savings_percent": round(total_savings / max(1, cases), 2),
        "failure_examples": failures[:10],
    }


def _alias(concept: str, output: str) -> bool:
    if concept == "cloud infrastructure costs":
        return "cloud spending" in output or "infrastructure expenses" in output
    if concept == "efficiency improvements":
        return "efficiency" in output or "optimization" in output
    return False


def main() -> None:
    print(json.dumps(run_self_discovery(), indent=2))


if __name__ == "__main__":
    main()
