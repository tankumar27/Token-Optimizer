from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field

from app.models import ChatMessage
from optimizer.pipeline import optimize_messages


@dataclass
class AbstractCase:
    name: str
    text: str
    expected: list[str] = field(default_factory=list)
    should_reduce: bool = True
    category: str = "abstract"


ABSTRACT_FAMILIES = {
    "identity_alias": [
        "The deployment coordinator, also called the release owner, approves production launches.",
        "Production launches are approved by the release owner, who is the deployment coordinator.",
        "In other words, the release owner is the person responsible for approving production deployments.",
    ],
    "nominalized_cost": [
        "Reducing unused compute lowered infrastructure expense.",
        "The reduction of idle compute capacity produced lower infrastructure spending.",
        "Infrastructure costs fell because idle compute was reduced.",
    ],
    "constraint_anaphora": [
        "Reliability is required during migration.",
        "It must remain intact while systems move.",
        "Availability also has to stay high during that transition.",
        "Performance cannot degrade during the move.",
    ],
    "mixed_list_redundancy": [
        "Before issuing a refund, agents must verify the order ID.",
        "They must also verify the customer email.",
        "The purchase date needs verification too.",
        "Refund reason is another required verification item.",
    ],
    "passive_active": [
        "The finance team approved the budget increase.",
        "The budget increase was approved by Finance.",
        "Approval for the increased budget came from the finance team.",
    ],
    "cause_effect_reworded": [
        "Better indexing reduced search latency.",
        "Search responses became faster after indexing improved.",
        "Latency dropped because the index was optimized.",
    ],
    "temporal_equivalence": [
        "The report is due before Friday.",
        "The team must submit the report by Thursday.",
        "Submission needs to happen no later than the day before Friday.",
    ],
    "negative_equivalence": [
        "The feature is not available to trial users.",
        "Trial users cannot access the feature.",
        "The feature is unavailable for accounts on the trial plan.",
    ],
    "rhetorical_repetition": [
        "The same warning is repeated again and again.",
        "It is restated repeatedly in different words.",
        "The text keeps saying the identical warning over and over.",
    ],
    "bullet_like": [
        "- Verify identity before reset.",
        "- Confirm identity before resetting access.",
        "- Identity confirmation is required prior to account reset.",
    ],
}

PRESERVATION = {
    "identity_alias": ["deployment coordinator", "release owner", "production"],
    "nominalized_cost": ["compute", "infrastructure"],
    "constraint_anaphora": ["reliability", "availability", "performance"],
    "mixed_list_redundancy": ["order ID", "customer email", "purchase date", "refund reason"],
    "passive_active": ["Finance", "budget"],
    "cause_effect_reworded": ["index", "latency"],
    "temporal_equivalence": ["report", "Friday"],
    "negative_equivalence": ["trial users", "feature"],
    "rhetorical_repetition": ["warning"],
    "bullet_like": ["identity", "reset"],
}

CONTRAST_CASES = [
    AbstractCase(
        "abstract_contrast_latency",
        "Better indexing reduced search latency. However, cache misses increased during peak traffic.",
        expected=["latency", "However", "cache misses"],
        should_reduce=False,
        category="contrast",
    ),
    AbstractCase(
        "abstract_contradiction_access",
        "Trial users cannot access the feature. Trial users can access the feature.",
        expected=["cannot access", "can access"],
        should_reduce=False,
        category="contradiction",
    ),
]


def generate_case(rng: random.Random, idx: int) -> AbstractCase:
    if idx % 19 == 0:
        return rng.choice(CONTRAST_CASES)
    family = rng.choice(list(ABSTRACT_FAMILIES))
    sentences = rng.sample(ABSTRACT_FAMILIES[family], k=rng.randint(2, len(ABSTRACT_FAMILIES[family])))
    if rng.random() < 0.45:
        sentences = _reshape(sentences, rng)
    text = _wrap(sentences, rng)
    return AbstractCase(f"{family}_{idx}", text, expected=_expected_for_text(family, text), category=family)


def _reshape(sentences: list[str], rng: random.Random) -> list[str]:
    style = rng.choice(["parenthetical", "newline", "semicolon", "prefix_noise"])
    if style == "parenthetical" and len(sentences) >= 2:
        return [f"{sentences[0]} ({sentences[1].rstrip('.')}).", *sentences[2:]]
    if style == "newline":
        return [sentence.replace(". ", ".\n") for sentence in sentences]
    if style == "semicolon":
        return ["; ".join(sentence.rstrip(".") for sentence in sentences) + "."]
    return [f"For clarity, {sentence[0].lower() + sentence[1:]}" for sentence in sentences]


def _wrap(sentences: list[str], rng: random.Random) -> str:
    random_order = sentences[:]
    rng.shuffle(random_order)
    wrapper = rng.choice([
        "",
        "Context:\n",
        "Notes from review:\n",
        "The following points were collected:\n",
    ])
    return wrapper + " ".join(random_order)


def evaluate(case: AbstractCase) -> dict:
    result = optimize_messages([ChatMessage(role="user", content=case.text)], "balanced", "dry-run", "dry-run")
    output = result["optimized_messages"][0].content
    missing = [item for item in case.expected if item.lower() not in output.lower()]
    wrong_reduction = (not case.should_reduce) and output != case.text
    no_reduction = case.should_reduce and result["optimized_tokens"] >= result["original_tokens"]
    ok = not missing and not wrong_reduction
    return {
        "name": case.name,
        "category": case.category,
        "ok": ok,
        "missing": missing,
        "wrong_reduction": wrong_reduction,
        "no_reduction": no_reduction,
        "original_tokens": result["original_tokens"],
        "optimized_tokens": result["optimized_tokens"],
        "savings_percent": result["savings_percent"],
        "backend_used": result["backend_used"],
        "input": case.text,
        "output": output,
    }


def _expected_for_text(family: str, text: str) -> list[str]:
    low = text.lower()
    expected: list[str] = []
    for item in PRESERVATION[family]:
        item_low = item.lower()
        if item_low in low:
            expected.append(item)
            continue
        if item == "Finance" and "finance team" in low:
            expected.append(item)
        elif item == "budget" and "increased budget" in low:
            expected.append(item)
        elif item == "index" and "indexing" in low:
            expected.append(item)
        elif item == "latency" and "faster" in low:
            expected.append(item)
        elif item == "reset" and "resetting access" in low:
            expected.append(item)
        elif item == "Friday" and ("thursday" in low or "day before friday" in low):
            expected.append(item)
        elif item == "trial users" and "trial plan" in low:
            expected.append(item)
    return expected


def run_abstract_stress(seed: int = 91, cases: int = 1000) -> dict:
    rng = random.Random(seed)
    rows = [evaluate(generate_case(rng, idx)) for idx in range(cases)]
    failures = [row for row in rows if not row["ok"]]
    by_category: dict[str, dict] = {}
    for row in rows:
        bucket = by_category.setdefault(row["category"], {"total": 0, "failures": 0, "no_reduction": 0, "average_savings": 0.0})
        bucket["total"] += 1
        bucket["failures"] += 0 if row["ok"] else 1
        bucket["no_reduction"] += 1 if row["no_reduction"] else 0
        bucket["average_savings"] += row["savings_percent"]
    for bucket in by_category.values():
        bucket["average_savings"] = round(bucket["average_savings"] / max(1, bucket["total"]), 2)
        bucket["failure_rate"] = round(bucket["failures"] / max(1, bucket["total"]), 3)
    return {
        "seed": seed,
        "cases": cases,
        "passed": cases - len(failures),
        "failed": len(failures),
        "failure_rate": round(len(failures) / max(1, cases), 3),
        "average_savings_percent": round(sum(row["savings_percent"] for row in rows) / max(1, cases), 2),
        "by_category": by_category,
        "failure_examples": failures[:25],
        "no_reduction_examples": [row for row in rows if row["no_reduction"]][:15],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=91)
    args = parser.parse_args()
    print(json.dumps(run_abstract_stress(args.seed, args.cases), indent=2))


if __name__ == "__main__":
    main()
