from __future__ import annotations

import json
import re
from .protect import ProtectedRegion, protected_regions_preserved, extract_sensitive_facts
from .token_counter import count_tokens


def _missing_facts(original: str, optimized: str) -> dict[str, list[str]]:
    missing: dict[str, list[str]] = {}
    for kind, values in extract_sensitive_facts(original).items():
        if kind in {"percentage", "measurement", "number", "date", "time", "money"}:
            optimized_keys = {_canonical_numeric_fact(kind, value) for value in extract_sensitive_facts(optimized).get(kind, set())}
            absent = sorted(value for value in values if value and _canonical_numeric_fact(kind, value) not in optimized_keys)
        else:
            absent = sorted(value for value in values if value and value not in optimized)
        if absent:
            missing[kind] = absent
    return missing


NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
    "million": 1000000,
}

MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

NUMBER_WORD_PATTERN = (
    r"zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million"
)
NUMBER_WORD_SEQUENCE = rf"(?:{NUMBER_WORD_PATTERN})(?:[-\s]+(?:{NUMBER_WORD_PATTERN}))*"


def _canonical_numeric_fact(kind: str, value: str) -> str:
    low = value.lower().replace(",", "").strip()
    if kind == "percentage":
        raw = low[:-1] if low.endswith("%") else re.sub(r"\s+percent$", "", low)
        return "percentage:" + _normalize_number(raw)
    if kind == "measurement":
        match = re.fullmatch(r"(.+?)\s*(ms|milliseconds|seconds|minutes|hours|days)", low)
        if match:
            unit = match.group(2)
            unit = "ms" if unit in {"ms", "milliseconds"} else unit.rstrip("s")
            return f"measurement:{_normalize_number(match.group(1))}:{unit}"
    if kind == "number":
        return "number:" + _normalize_number(low)
    if kind == "date":
        return "date:" + _normalize_date(low)
    if kind == "time":
        return "time:" + _normalize_time(low)
    if kind == "money":
        return "money:" + _normalize_money(low)
    return kind + ":" + low


def _normalize_date(value: str) -> str:
    iso = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", value)
    if iso:
        return value
    written = re.fullmatch(r"([a-z]+)\s+(\d{1,2})\s+(\d{4})", value, re.I)
    if written:
        month = MONTHS.get(written.group(1).lower())
        if month:
            return f"{int(written.group(3)):04d}-{month:02d}-{int(written.group(2)):02d}"
    return value


def _normalize_time(value: str) -> str:
    numeric = re.fullmatch(r"(\d{1,2}):(\d{2})\s*(am|pm)", value, re.I)
    if numeric:
        return f"{int(numeric.group(1)):02d}:{int(numeric.group(2)):02d}:{numeric.group(3).lower()}"
    words = re.fullmatch(rf"({NUMBER_WORD_SEQUENCE})\s+(?:oh\s+)?({NUMBER_WORD_SEQUENCE})\s*(am|pm)", value, re.I)
    if words:
        hour = _parse_number_words(words.group(1))
        minute = _parse_number_words(words.group(2))
        if hour is not None and minute is not None and 0 <= hour <= 23 and 0 <= minute <= 59:
            return f"{hour:02d}:{minute:02d}:{words.group(3).lower()}"
    return value


def _normalize_money(value: str) -> str:
    numeric = re.fullmatch(r"\$([0-9]+(?:\.[0-9]+)?)", value)
    if numeric:
        return numeric.group(1)
    words = re.fullmatch(rf"({NUMBER_WORD_SEQUENCE})\s+dollars", value, re.I)
    if words:
        parsed = _parse_number_words(words.group(1))
        if parsed is not None:
            return str(parsed)
    return value


def _normalize_number(value: str) -> str:
    parsed = _parse_number_words(value)
    if parsed is not None:
        return str(parsed)
    return value.strip()


def _parse_number_words(text: str) -> int | None:
    parts = [part for part in re.split(r"[-\s]+", text.lower().strip()) if part]
    if not parts or any(part not in NUMBER_WORDS for part in parts):
        return None
    total = 0
    current = 0
    for part in parts:
        value = NUMBER_WORDS[part]
        if part == "hundred":
            current = max(1, current) * 100
        elif part in {"thousand", "million"}:
            total += max(1, current) * value
            current = 0
        else:
            current += value
    return total + current


def _json_blocks_valid(text: str) -> bool:
    for match in re.findall(r"```json\s*([\s\S]*?)```", text, flags=re.IGNORECASE):
        try:
            json.loads(match)
        except Exception:
            return False
    return True


def run_quality_gate(
    original: str,
    optimized: str,
    regions: list[ProtectedRegion],
    compression_level: str,
    backend_uncertain: bool = False,
) -> dict:
    original_tokens = count_tokens(original)
    optimized_tokens = count_tokens(optimized)
    missing = _missing_facts(original, optimized)
    ratio = optimized_tokens / max(1, original_tokens)
    checks = {
        "tokens_reduced": optimized_tokens < original_tokens,
        "protected_regions_preserved": protected_regions_preserved(optimized, regions),
        "sensitive_facts_preserved": not missing,
        "json_blocks_valid": _json_blocks_valid(optimized),
        "compression_ratio_reasonable": ratio >= {"safe": 0.45, "balanced": 0.30, "aggressive": 0.20}.get(compression_level, 0.45),
        "backend_confident": not backend_uncertain,
    }
    accepted = all(checks.values())
    reasons = [name for name, ok in checks.items() if not ok]
    return {
        "accepted": accepted,
        "rejection_reason": "; ".join(reasons) if reasons else None,
        "checks": checks,
        "missing_facts": missing,
        "original_tokens": original_tokens,
        "optimized_tokens": optimized_tokens,
    }
