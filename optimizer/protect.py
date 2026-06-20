from __future__ import annotations

from dataclasses import dataclass, asdict
import re


@dataclass
class ProtectedRegion:
    placeholder: str
    value: str
    kind: str

    def to_dict(self) -> dict:
        data = asdict(self)
        data["length"] = len(self.value)
        data.pop("value", None)
        return data


NUMBER_WORD_PATTERN = (
    r"zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million"
)
NUMBER_WORD_SEQUENCE = rf"(?:{NUMBER_WORD_PATTERN})(?:[-\s]+(?:{NUMBER_WORD_PATTERN}))*"


PATTERNS: list[tuple[str, str]] = [
    ("fenced_code", r"```[\s\S]*?```"),
    ("inline_code", r"`[^`\n]+`"),
    ("latex", r"\$(?!\d)[^$\n]+(?<!\d)\$"),
    ("json_block", r"(?s)(\{(?:[^{}]|(?:\{[^{}]*\}))*\}|\[(?:[^\[\]]|(?:\[[^\[\]]*\]))*\])"),
    ("xml_html", r"<[A-Za-z][^>]*>[\s\S]*?</[A-Za-z][^>]*>"),
    ("url", r"https?://[^\s)>\"]+"),
    ("email", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    ("api_key_like", r"\b(?:sk-|AIza|AQ\.)[A-Za-z0-9_\-.]{16,}\b"),
    ("date", r"\b\d{4}-\d{2}-\d{2}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]* \d{1,2}, \d{4}\b"),
    ("time", r"\b\d{1,2}:\d{2}\s*(?:AM|PM)\b"),
    ("time", rf"\b(?:{NUMBER_WORD_SEQUENCE})\s+(?:oh\s+)?(?:{NUMBER_WORD_SEQUENCE})\s*(?:AM|PM)\b"),
    ("money", r"\$\d[\d,]*(?:\.\d+)?\b"),
    ("money", rf"\b{NUMBER_WORD_SEQUENCE}\s+dollars\b"),
    ("percentage", r"\b\d+(?:\.\d+)?%"),
    ("percentage", rf"\b(?:\d+(?:\.\d+)?|{NUMBER_WORD_SEQUENCE})\s+percent\b"),
    ("legal_clause", r"\bClause \d+(?:\.\d+)*\b"),
    ("id", r"(?-i:\b[A-Z]{2,}(?:-[A-Z0-9]+)+\b)"),
    ("quoted_text", r"\"[^\"\n]{2,}\"|'[^'\n]{2,}'"),
    ("equation", r"\b[a-zA-Z]\s*(?:\([a-zA-Z]\))?\s*=\s*[-+*/^(). 0-9a-zA-Z]*?(?=\s+(?:appears|must)\b|[.,;:]|$)"),
]


def _protect_matches(text: str, kind: str, pattern: str, regions: list[ProtectedRegion], seen: dict[tuple[str, str], str]) -> str:
    def replace(match: re.Match) -> str:
        value = match.group(0)
        if value.startswith("__PROTECTED_"):
            return value
        key = _canonical_protected_key(kind, value)
        if key in seen:
            return seen[key]
        placeholder = f"__PROTECTED_{len(regions)}__"
        seen[key] = placeholder
        regions.append(ProtectedRegion(placeholder, value, kind))
        return placeholder

    return re.sub(pattern, replace, text, flags=re.IGNORECASE)


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


def _canonical_protected_key(kind: str, value: str) -> tuple[str, str]:
    low = value.lower().replace(",", "").strip()
    if kind == "date":
        return kind, _normalize_date(low)
    if kind == "time":
        return kind, _normalize_time(low)
    if kind == "money":
        return kind, _normalize_money(low)
    if kind == "percentage":
        raw = low[:-1] if low.endswith("%") else re.sub(r"\s+percent$", "", low)
        return kind, _normalize_number(raw)
    if kind == "measurement":
        match = re.fullmatch(r"(.+?)\s*(ms|milliseconds|seconds|minutes|hours|days)", low)
        if match:
            unit = match.group(2)
            unit = "ms" if unit in {"ms", "milliseconds"} else unit.rstrip("s")
            return kind, f"{_normalize_number(match.group(1))}:{unit}"
    return kind, value


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
        hour = int(numeric.group(1))
        minute = int(numeric.group(2))
        return f"{hour:02d}:{minute:02d}:{numeric.group(3).lower()}"
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


def protect_text(text: str) -> tuple[str, list[ProtectedRegion]]:
    regions: list[ProtectedRegion] = []
    seen: dict[tuple[str, str], str] = {}
    protected = text
    for kind, pattern in PATTERNS:
        protected = _protect_matches(protected, kind, pattern, regions, seen)
    return protected, regions


def restore_text(text: str, regions: list[ProtectedRegion]) -> str:
    restored = text
    for region in regions:
        restored = restored.replace(region.placeholder, region.value)
    return restored


def protected_regions_preserved(restored_text: str, regions: list[ProtectedRegion]) -> bool:
    return all(region.value in restored_text for region in regions)


def public_region_metadata(regions: list[ProtectedRegion]) -> list[dict]:
    return [region.to_dict() for region in regions]


def extract_sensitive_facts(text: str) -> dict[str, set[str]]:
    facts: dict[str, set[str]] = {}
    for kind, pattern in PATTERNS:
        if kind in {"fenced_code", "inline_code", "json_block", "xml_html", "quoted_text"}:
            continue
        facts.setdefault(kind, set()).update(re.findall(pattern, text, flags=re.IGNORECASE))
    number_text = re.sub(
        r"(?im)^(?:Retrieved (?:context|contract) chunk|Source|Chunk|Document section)\s+[A-Z0-9]+(?:\s*[-\u2013\u2014?]\s*[^:]+)?:.*$",
        "",
        text,
    )
    number_text = re.sub(
        r"(?im)^(?:User Question|Question|Task|System Prompt|Instructions|Metadata|Customer Ticket):.*$",
        "",
        number_text,
    )
    for pattern in [pattern for kind, pattern in PATTERNS if kind == "date"]:
        number_text = re.sub(pattern, "", number_text, flags=re.IGNORECASE)
    facts["number"] = set(re.findall(r"\b\d+(?:\.\d+)?\b", number_text))
    return facts
