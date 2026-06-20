from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field

from app.models import ChatMessage
from optimizer.pipeline import optimize_messages


@dataclass
class StressCase:
    name: str
    text: str
    expected: list[str] = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)
    should_reduce: bool = True
    category: str = "general"


SCENARIOS = {
    "reliability": [
        "System reliability must be preserved.",
        "Service availability must remain high.",
        "Performance degradation is not acceptable.",
        "Customer experience should not suffer.",
        "Critical services must maintain performance and availability.",
    ],
    "support_verification": [
        "Agents must verify order ID.",
        "Agents must verify customer email.",
        "Agents must verify purchase date.",
        "Agents must verify refund reason.",
        "Agents must verify product SKU.",
    ],
    "refund_policy": [
        "Customers may request refunds within 30 days of purchase.",
        "Refund requests must include order ID, customer email, purchase date, refund reason, and product SKU.",
        "Refunds exceeding $500 require Finance approval.",
        "Agents must not promise approval before Finance reviews the request.",
    ],
    "product_benefits": [
        "The platform saves time.",
        "The platform reduces manual work.",
        "The platform improves reporting accuracy.",
        "The platform helps teams monitor operational metrics.",
    ],
    "cloud_cost": [
        "The company reduced cloud infrastructure costs through compute optimization initiatives.",
        "By improving compute efficiency, the organization lowered cloud spending.",
        "Reduced compute waste contributed significantly to lower infrastructure expenses.",
        "Lower infrastructure expenses improved operating margins.",
        "Reduced spending contributed positively to financial performance.",
    ],
    "incident_management": [
        "Incident response should minimize downtime.",
        "Teams must restore normal operations quickly.",
        "Responders should communicate status updates during active incidents.",
        "Post-incident reviews should identify corrective actions.",
    ],
    "privacy_security": [
        "Do not expose API keys.",
        "Never reveal access tokens.",
        "Secrets must not be included in customer-facing responses.",
        "Credentials should remain confidential.",
    ],
}

CONTRADICTIONS = [
    StressCase(
        "cost_contradiction",
        "Cloud costs decreased. Cloud costs increased.",
        expected=["Cloud costs decreased", "Cloud costs increased"],
        should_reduce=False,
        category="contradiction",
    ),
    StressCase(
        "modality_shift",
        "We might need additional funding next quarter. We will need additional funding next quarter.",
        expected=["might", "will"],
        should_reduce=False,
        category="modality",
    ),
    StressCase(
        "contrast_quantity",
        "Cloud costs decreased. However, customer acquisition costs increased by 11%.",
        expected=["However", "11%", "customer acquisition costs"],
        should_reduce=False,
        category="contrast",
    ),
]


def generate_case(rng: random.Random, idx: int) -> StressCase:
    if idx % 17 == 0:
        return rng.choice(CONTRADICTIONS)
    family = rng.choice(list(SCENARIOS))
    sentences = rng.sample(SCENARIOS[family], k=rng.randint(2, min(5, len(SCENARIOS[family]))))
    if family in {"cloud_cost", "reliability", "refund_policy"} and rng.random() < 0.35:
        sentences.append("However, customer acquisition costs increased by 11%.")
    rng.shuffle(sentences)
    wrapper = rng.choice([
        "You are an enterprise assistant.\n\nContext:\n",
        "Retrieved Context Chunk 1:\n",
        "Executive Summary:\n",
        "",
    ])
    text = wrapper + " ".join(sentences)
    expected = expected_for_family(family, sentences)
    return StressCase(f"{family}_{idx}", text, expected=expected, category=family)


def expected_for_family(family: str, sentences: list[str]) -> list[str]:
    joined = " ".join(sentences).lower()
    expected: list[str] = []
    checks = {
        "reliability": ["reliability", "availability", "performance", "customer experience"],
        "support_verification": ["order ID", "customer email", "purchase date", "refund reason", "product SKU"],
        "refund_policy": ["30", "order ID", "customer email", "purchase date", "refund reason", "product SKU", "$500", "Finance"],
        "product_benefits": ["saves time", "manual work", "reporting accuracy", "operational metrics"],
        "cloud_cost": ["cloud", "compute", "efficiency", "operating margins", "financial performance"],
        "incident_management": ["downtime", "normal operations", "status updates", "corrective actions"],
        "privacy_security": ["API keys", "access tokens", "Secrets", "Credentials"],
    }
    for value in checks.get(family, []):
        if value.lower() in joined:
            expected.append(value)
    if "11%" in joined:
        expected.append("11%")
    return expected


def evaluate_case(case: StressCase) -> dict:
    result = optimize_messages([ChatMessage(role="user", content=case.text)], "balanced", "dry-run", "dry-run")
    output = result["optimized_messages"][0].content
    reduced = result["optimized_tokens"] < result["original_tokens"]
    missing = [item for item in case.expected if not present(item, output)]
    forbidden_present = [item for item in case.forbidden if present(item, output)]
    wrong_reduction = (not case.should_reduce) and output != case.text
    no_reduction = case.should_reduce and not reduced
    ok = not missing and not forbidden_present and not wrong_reduction
    return {
        "name": case.name,
        "category": case.category,
        "ok": ok,
        "missing": missing,
        "forbidden_present": forbidden_present,
        "wrong_reduction": wrong_reduction,
        "no_reduction": no_reduction,
        "original_tokens": result["original_tokens"],
        "optimized_tokens": result["optimized_tokens"],
        "savings_percent": result["savings_percent"],
        "backend_used": result["backend_used"],
        "input": case.text,
        "output": output,
    }


def present(needle: str, output: str) -> bool:
    low = output.lower()
    needle_low = needle.lower()
    if needle_low in low:
        return True
    aliases = {
        "manual work": ["reduces manual work"],
        "reporting accuracy": ["improves reporting accuracy"],
        "operational metrics": ["monitors operational metrics", "operational metrics"],
        "cloud": ["cloud spending", "cloud infrastructure costs", "cloud costs"],
        "compute": ["compute optimization", "compute efficiency"],
        "efficiency": ["efficiency improvements", "improved efficiency"],
        "api keys": ["api keys"],
        "secrets": ["secrets"],
    }
    return any(alias in low for alias in aliases.get(needle_low, []))


def run_stress(seed: int = 11, cases: int = 1000) -> dict:
    rng = random.Random(seed)
    rows = [evaluate_case(generate_case(rng, idx)) for idx in range(cases)]
    failures = [row for row in rows if not row["ok"]]
    no_reductions = [row for row in rows if row["no_reduction"]]
    by_category: dict[str, dict] = {}
    for row in rows:
        bucket = by_category.setdefault(row["category"], {"total": 0, "failures": 0, "average_savings": 0.0})
        bucket["total"] += 1
        bucket["failures"] += 0 if row["ok"] else 1
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
        "no_reduction_count": len(no_reductions),
        "average_savings_percent": round(sum(row["savings_percent"] for row in rows) / max(1, cases), 2),
        "by_category": by_category,
        "failure_examples": failures[:20],
        "no_reduction_examples": no_reductions[:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    print(json.dumps(run_stress(args.seed, args.cases), indent=2))


if __name__ == "__main__":
    main()
