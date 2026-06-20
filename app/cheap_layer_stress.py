from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field

from optimizer.cheap_layer import cheap_layer_backend, canonical_fact_keys, state_signatures, risk_keys, VERSION


@dataclass
class CheapCase:
    name: str
    text: str
    must_include: list[str] = field(default_factory=list)
    should_compress: bool = True


def generate_cases(seed: int, count: int) -> list[CheapCase]:
    rng = random.Random(seed)
    factories = [
        exact_instruction_case,
        metric_number_word_case,
        percent_number_word_case,
        different_entity_case,
        same_entity_different_state_case,
        protected_domain_terms_case,
        sectioned_instruction_case,
    ]
    return [rng.choice(factories)(rng, idx) for idx in range(count)]


def exact_instruction_case(rng: random.Random, idx: int) -> CheapCase:
    sentence = rng.choice([
        "Do not promise refund approval.",
        "Inform the customer that the request is under review.",
        "Agents must preserve ticket identifiers.",
    ])
    return CheapCase(f"exact_instruction_{idx}", f"{sentence} {sentence}", [sentence], True)


def metric_number_word_case(rng: random.Random, idx: int) -> CheapCase:
    pairs = [
        ("12 ms", "twelve ms"),
        ("28 ms", "twenty-eight ms"),
        ("510 ms", "five hundred ten ms"),
        ("220 ms", "two hundred twenty ms"),
    ]
    a, aw = rng.choice(pairs)
    b, bw = rng.choice(pairs)
    return CheapCase(
        f"metric_number_word_{idx}",
        f"Database latency changed from {a} to {b}. Database latency changed from {aw} to {bw}.",
        [a.split()[0], b.split()[0]],
        True,
    )


def percent_number_word_case(rng: random.Random, idx: int) -> CheapCase:
    options = [("72%", "seventy-two percent"), ("83%", "eighty-three percent")]
    numeric, words = rng.choice(options)
    return CheapCase(f"percent_number_word_{idx}", f"The rollout paused at {numeric}. The rollout paused at {words}.", [numeric], True)


def different_entity_case(rng: random.Random, idx: int) -> CheapCase:
    left, right = rng.choice([("us-east-1", "us-west-2"), ("zn-11", "zn-12"), ("ledger_v2_read", "ledger_v2_write")])
    return CheapCase(f"different_entity_{idx}", f"Region {left} was affected. Region {right} was affected.", [left, right], False)


def same_entity_different_state_case(rng: random.Random, idx: int) -> CheapCase:
    entity = rng.choice(["CACHE-WARMER", "DBN-08", "emergency_ledger_rollback"])
    return CheapCase(f"different_state_{idx}", f"Service {entity} is degraded. Service {entity} is not offline.", [entity, "degraded", "not offline"], False)


def protected_domain_terms_case(rng: random.Random, idx: int) -> CheapCase:
    term = rng.choice(["X-chromosome", "10-K", "T+2", "CUSIP 9128285M8", "A/R"])
    return CheapCase(f"protected_domain_{idx}", f"{term} must be preserved. {term} must be preserved.", [term], True)


def sectioned_instruction_case(rng: random.Random, idx: int) -> CheapCase:
    return CheapCase(
        f"sectioned_instruction_{idx}",
        "Agent Instructions:\nDo not promise refund approval.\nDo not promise refund approval.\nTask:\nWrite the response.",
        ["Task:\nWrite the response.", "Do not promise refund approval."],
        True,
    )


def evaluate(case: CheapCase) -> dict:
    output, traces = cheap_layer_backend(case.text, "balanced")
    trace = traces[0]
    original_facts = canonical_fact_keys(case.text)
    output_facts = canonical_fact_keys(output)
    missing = [item for item in case.must_include if item not in output]
    facts_ok = original_facts <= output_facts
    states_ok = state_signatures(case.text) <= state_signatures(output)
    risks_ok = risk_keys(case.text) <= risk_keys(output)
    compressed = trace["accepted"] and trace["after"]["tokens"] < trace["before"]["tokens"]
    wrong_compression = compressed and not case.should_compress
    ok = not missing and facts_ok and states_ok and risks_ok and not wrong_compression
    return {
        "name": case.name,
        "ok": ok,
        "accepted": trace["accepted"],
        "wrong_compression": wrong_compression,
        "missing": missing,
        "facts_ok": facts_ok,
        "states_ok": states_ok,
        "risks_ok": risks_ok,
        "original_tokens": trace["before"]["tokens"],
        "compressed_tokens": trace["after"]["tokens"],
        "savings_percent": round(100 * max(0, trace["before"]["tokens"] - trace["after"]["tokens"]) / max(1, trace["before"]["tokens"]), 2),
        "input": case.text,
        "output": output,
    }


def run(seed: int = 19, count: int = 250) -> dict:
    rows = [evaluate(case) for case in generate_cases(seed, count)]
    failures = [row for row in rows if not row["ok"]]
    return {
        "version": VERSION,
        "cases": count,
        "passed": count - len(failures),
        "failed": len(failures),
        "failure_rate": round(len(failures) / max(1, count), 3),
        "average_savings_percent": round(sum(row["savings_percent"] for row in rows) / max(1, count), 2),
        "failures": failures[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=250)
    parser.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()
    print(json.dumps(run(args.seed, args.cases), indent=2))


if __name__ == "__main__":
    main()
