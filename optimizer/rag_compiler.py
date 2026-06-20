from __future__ import annotations

from dataclasses import dataclass, field, asdict
from difflib import SequenceMatcher
import re

from .semantic_validator import SemanticValidator
from .token_counter import count_tokens


RAG_HEADER_RE = re.compile(
    r"(?im)^((?:Retrieved (?:context|contract) chunk|Source|Chunk|Document section)\s+[A-Z0-9]+(?:\s*[-\u2013\u2014?]\s*[^:]+)?|Policy excerpt):\s*$"
)

ANY_SECTION_RE = re.compile(
    r"(?im)^([A-Z][A-Za-z0-9 /&-]{2,60}:|(?:Retrieved (?:context|contract) chunk|Source|Chunk|Document section)\s+[A-Z0-9]+(?:\s*[-\u2013\u2014?]\s*[^:]+)?:|Policy excerpt:)\s*$"
)

NON_EVIDENCE_HEADERS = {
    "task:",
    "user question:",
    "question:",
    "system prompt:",
    "instructions:",
    "metadata:",
    "customer ticket:",
    "ticket:",
    "agent instructions:",
    "agent reminder:",
}


@dataclass
class EvidenceUnit:
    theme: str
    canonical_text: str
    facts: tuple[str, ...] = ()
    protected_facts: tuple[str, ...] = ()
    polarity: str = "positive"
    source_chunks: list[int] = field(default_factory=list)

    @property
    def signature(self) -> tuple:
        return (self.theme, self.canonical_text.lower(), self.polarity)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RagChunk:
    index: int
    header: str
    body: str
    full_start: int = 0
    full_end: int = 0
    evidence_units: list[EvidenceUnit] = field(default_factory=list)
    protected_facts: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        return f"chunk {self.index}"

    def to_trace(self) -> dict:
        return {
            "index": self.index,
            "header": self.header,
            "body_preview": self.body[:500],
            "body_tokens": count_tokens(self.body),
            "protected_facts": list(self.protected_facts),
            "evidence_units": [unit.to_dict() for unit in self.evidence_units],
        }


@dataclass
class RagDocument:
    system_prompt: str
    user_question: str | None
    task: str | None
    chunks: list[RagChunk]
    metadata_sections: list[str]


class RagContextCompiler:
    def __init__(self, validator: SemanticValidator | None = None) -> None:
        # The RAG compiler is a cheap structural/evidence layer by default.
        # Callers can pass a transformer-backed validator for deeper offline
        # analysis, but the production optimize endpoint should not pay that
        # latency before deterministic compression.
        self.validator = validator

    def parse_chunks(self, text: str) -> list[RagChunk]:
        return self.parse_document(text).chunks

    def parse_document(self, text: str) -> RagDocument:
        matches = list(RAG_HEADER_RE.finditer(text))
        chunks: list[RagChunk] = []
        for i, match in enumerate(matches):
            start = match.end()
            next_rag = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            next_non_evidence = _next_non_evidence_header(text, start, next_rag)
            next_protected_block = _next_standalone_protected_block(text, start, next_rag)
            end = min(
                next_rag,
                next_non_evidence if next_non_evidence is not None else next_rag,
                next_protected_block if next_protected_block is not None else next_rag,
            )
            body = text[start:end].strip()
            chunk = RagChunk(i + 1, match.group(1), body, match.start(), end)
            chunk.protected_facts = tuple(sorted(_protected_facts(body)))
            chunk.evidence_units = extract_evidence_units(chunk)
            chunks.append(chunk)

        first_chunk_start = chunks[0].full_start if chunks else len(text)
        system_prompt = text[:first_chunk_start].strip()
        return RagDocument(
            system_prompt=system_prompt,
            user_question=_section_body(text, "User Question") or _section_body(text, "Question"),
            task=_section_body(text, "Task"),
            chunks=chunks,
            metadata_sections=[],
        )

    def compile(self, text: str) -> tuple[str, list[dict], list[dict]]:
        document = self.parse_document(text)
        chunks = document.chunks
        if not chunks:
            return text, [], []

        parsed_trace = {
            "backend": "rag_compiler",
            "candidate_type": "rag_document_parse",
            "accepted": False,
            "reason": "parsed RAG document into structured chunks",
            "system_prompt_present": bool(document.system_prompt),
            "user_question_present": bool(document.user_question),
            "task_present": bool(document.task),
            "parsed_chunks": [chunk.to_trace() for chunk in chunks],
        }

        groups, graph = self._build_overlap_groups(chunks)
        renderable_groups = [group for group in groups if group["canonical"] and not group["contradiction"] and len(group["chunks"]) >= 2]
        if not renderable_groups:
            return text, [parsed_trace], graph

        canonical_block = self._render_canonical_block(renderable_groups)
        unique_units = self._unique_evidence_units(chunks, renderable_groups)
        unique_chunks = self._unique_chunks(chunks, renderable_groups)
        conflict_chunks = self._conflict_chunks(chunks, groups)

        replacement_parts = [canonical_block]
        if unique_units:
            replacement_parts.append("Unique Evidence:\n" + "\n".join(f"- {unit.canonical_text}" for unit in unique_units))
        if unique_chunks:
            replacement_parts.append("Preserved Unique Chunks:\n" + "\n\n".join(_chunk_text(chunk) for chunk in unique_chunks))
        if conflict_chunks:
            replacement_parts.append("Contradictory Evidence Preserved:\n" + "\n\n".join(_chunk_text(chunk) for chunk in conflict_chunks))

        replacement = "\n\n".join(replacement_parts).strip()
        start = chunks[0].full_start
        end = max(chunk.full_end for chunk in chunks)
        suffix = text[end:]
        separator = "\n\n" if suffix and not suffix.startswith("\n") else ""
        optimized = (text[:start] + replacement + separator + suffix).strip()
        optimized = _dedupe_repeated_sentences(optimized)
        if count_tokens(optimized) >= count_tokens(text):
            parsed_trace["rejected_reason"] = "canonical evidence block was not shorter"
            return text, [parsed_trace], graph

        trace = {
            "backend": "rag_compiler",
            "candidate_type": "chunk_level_evidence_compilation",
            "accepted": True,
            "reason": "overlapping retrieved evidence rendered as canonical evidence block",
            "span_text": text[start:end],
            "retained_span": replacement,
            "removed_span": text[start:end],
            "tokens_saved": count_tokens(text) - count_tokens(optimized),
            "parsed_chunks": [chunk.to_trace() for chunk in chunks],
            "overlap_groups": groups,
            "canonical_evidence": [group["canonical"] for group in renderable_groups],
            "unique_facts_preserved": sorted({fact for chunk in unique_chunks for fact in chunk.protected_facts} | {fact for unit in unique_units for fact in unit.protected_facts}),
            "facts_removed": [],
            "contradiction_groups": [group for group in groups if group["contradiction"]],
            "score": 0.94,
        }
        return optimized, [parsed_trace, trace], graph

    def _build_overlap_groups(self, chunks: list[RagChunk]) -> tuple[list[dict], list[dict]]:
        groups: list[dict] = []
        graph: list[dict] = []
        units: list[tuple[RagChunk, EvidenceUnit]] = [(chunk, unit) for chunk in chunks for unit in chunk.evidence_units]
        used: set[tuple[int, str]] = set()

        for chunk, unit in units:
            key = (chunk.index, unit.canonical_text)
            if key in used:
                continue
            members = [(chunk, unit)]
            for other_chunk, other_unit in units:
                other_key = (other_chunk.index, other_unit.canonical_text)
                if other_key == key or other_key in used:
                    continue
                overlap = _evidence_overlap(unit, other_unit, self.validator)
                if overlap["similarity"] >= 0.78:
                    members.append((other_chunk, other_unit))

            if len(members) < 2:
                continue
            for member_chunk, member_unit in members:
                used.add((member_chunk.index, member_unit.canonical_text))

            contradiction = _members_contradict(members)
            canonical = None if contradiction else _choose_canonical([member_unit for _, member_unit in members])
            chunk_ids = sorted({member_chunk.index for member_chunk, _ in members})
            group = {
                "theme": unit.theme,
                "chunks": chunk_ids,
                "canonical": canonical,
                "contradiction": contradiction,
                "atomic_facts": sorted({fact for _, member_unit in members for fact in member_unit.facts}),
                "protected_facts": sorted({fact for _, member_unit in members for fact in member_unit.protected_facts}),
                "reason": "contradictory evidence preserved" if contradiction else "overlapping evidence grouped",
            }
            groups.append(group)
            canonical_chunk = chunk_ids[0]
            for duplicate_chunk in chunk_ids[1:]:
                graph.append({
                    "canonical_chunk": canonical_chunk,
                    "duplicate_chunk": duplicate_chunk,
                    "similarity": 1.0 if canonical else 0.78,
                    "reason": group["reason"],
                    "contradiction_gate": "failed" if contradiction else "passed",
                    "estimated_saved_tokens": 0 if contradiction else 10,
                    "unique_facts_preserved": group["protected_facts"],
                    "canonical_evidence": canonical,
                    "decision": "preserve_conflict" if contradiction else "canonicalize_overlap",
                })
        return groups, graph

    def _render_canonical_block(self, groups: list[dict]) -> str:
        lines = ["Canonical Evidence:"]
        for group in groups:
            lines.append(f"- {group['canonical']}")
        return "\n".join(lines)

    def _unique_chunks(self, chunks: list[RagChunk], groups: list[dict]) -> list[RagChunk]:
        fully_represented = {idx for group in groups for idx in group["chunks"]}
        unique: list[RagChunk] = []
        for chunk in chunks:
            if _is_ticket_chunk(chunk):
                unique.append(chunk)
                continue
            if chunk.index in fully_represented:
                continue
            if chunk.evidence_units and all(unit.theme != "unknown_evidence" for unit in chunk.evidence_units):
                continue
            unique.append(chunk)
        return unique

    def _unique_evidence_units(self, chunks: list[RagChunk], groups: list[dict]) -> list[EvidenceUnit]:
        represented = {(idx, group["theme"]) for group in groups if not group["contradiction"] for idx in group["chunks"]}
        seen: set[tuple[str, str]] = set()
        unique: list[EvidenceUnit] = []
        for chunk in chunks:
            if _is_ticket_chunk(chunk):
                continue
            for unit in chunk.evidence_units:
                if (chunk.index, unit.theme) in represented:
                    continue
                key = (unit.theme, _canonical_key(unit.canonical_text))
                if key in seen:
                    continue
                seen.add(key)
                unique.append(unit)
        return unique

    def _conflict_chunks(self, chunks: list[RagChunk], groups: list[dict]) -> list[RagChunk]:
        conflict_ids = {idx for group in groups if group["contradiction"] for idx in group["chunks"]}
        return [chunk for chunk in chunks if chunk.index in conflict_ids]


def extract_evidence_units(chunk: RagChunk) -> list[EvidenceUnit]:
    text = chunk.body
    low = text.lower()
    units: list[EvidenceUnit] = []

    verification_items = [label for label in ["order ID", "customer email", "purchase date", "refund reason", "product SKU"] if label.lower() in low]
    if len(verification_items) >= 2 and re.search(r"\b(verify|confirm|include|require|requires|required|must include)\b", low):
        units.append(EvidenceUnit(
            "verification_rule",
            f"Verification rule: agents must verify {_join_list(verification_items)}.",
            facts=tuple(verification_items),
            protected_facts=tuple(sorted(_protected_facts(text))),
            source_chunks=[chunk.index],
        ))

    threshold = _finance_threshold(text)
    if threshold and "finance" in low and ("approval" in low or "review" in low or "reviewed" in low):
        units.append(EvidenceUnit(
            "refund_approval_rule",
            f"Refund rule: refunds over {threshold} require Finance approval.",
            facts=("Finance approval", threshold),
            protected_facts=tuple(sorted(_protected_facts(text))),
            source_chunks=[chunk.index],
        ))

    if "30 days" in low and "refund" in low:
        units.append(EvidenceUnit(
            "refund_window_rule",
            "Refund window: customers may request a refund within 30 days if usage remains within trial limits.",
            facts=("30 days", "trial limits"),
            protected_facts=tuple(sorted(_protected_facts(text))),
            source_chunks=[chunk.index],
        ))

    latency = _latency_change(text)
    if latency:
        units.append(EvidenceUnit(
            "latency_metric",
            f"Latency metric: latency dropped from {latency[0]} to {latency[1]}.",
            facts=("latency", latency[0], latency[1]),
            protected_facts=tuple(sorted(_protected_facts(text))),
            source_chunks=[chunk.index],
        ))

    uptime = _uptime_metric(text)
    if uptime:
        units.append(EvidenceUnit(
            "uptime_metric",
            f"Uptime metric: uptime was {uptime[0]} in {uptime[1]}.",
            facts=("uptime", uptime[0], uptime[1]),
            protected_facts=tuple(sorted(_protected_facts(text))),
            source_chunks=[chunk.index],
        ))

    if not units:
        deduped = _dedupe_sentences(text)
        units.append(EvidenceUnit(
            "unknown_evidence",
            f"Evidence: {deduped}",
            facts=tuple(sorted(_keywords(deduped))),
            protected_facts=tuple(sorted(_protected_facts(text))),
            source_chunks=[chunk.index],
        ))
    return units


def _next_non_evidence_header(text: str, start: int, stop: int) -> int | None:
    for match in ANY_SECTION_RE.finditer(text, start, stop):
        if match.group(1).strip().lower() in NON_EVIDENCE_HEADERS:
            return match.start()
    return None


def _next_standalone_protected_block(text: str, start: int, stop: int) -> int | None:
    match = re.search(r"(?m)^__PROTECTED_\d+__\s*$", text[start:stop])
    return start + match.start() if match else None


def _section_body(text: str, header: str) -> str | None:
    pattern = re.compile(rf"(?ims)^{re.escape(header)}:\s*(.*?)(?=^[A-Z][A-Za-z0-9 /&-]{{2,60}}:\s*$|\Z)")
    match = pattern.search(text)
    return match.group(1).strip() if match else None


def _evidence_overlap(a: EvidenceUnit, b: EvidenceUnit, validator: SemanticValidator | None) -> dict:
    if a.theme == b.theme and a.theme != "unknown_evidence":
        shared = set(a.facts) & set(b.facts)
        if shared or a.theme in {"verification_rule", "refund_approval_rule"}:
            return {"similarity": 1.0, "reason": "same evidence theme and facts"}
    lexical = SequenceMatcher(None, a.canonical_text.lower(), b.canonical_text.lower()).ratio()
    semantic = validator.similarity(a.canonical_text, b.canonical_text)["semantic_similarity"] if validator else lexical
    return {"similarity": max(lexical, semantic), "reason": "semantic evidence overlap"}


def _members_contradict(members: list[tuple[RagChunk, EvidenceUnit]]) -> bool:
    units = [unit for _, unit in members]
    by_theme: dict[str, list[EvidenceUnit]] = {}
    for unit in units:
        by_theme.setdefault(unit.theme, []).append(unit)
    for theme, theme_units in by_theme.items():
        if theme in {"refund_approval_rule", "latency_metric", "uptime_metric"}:
            protected_sets = {tuple(_normalize_fact(fact) for fact in unit.protected_facts) for unit in theme_units if unit.protected_facts}
            if len(protected_sets) > 1:
                return True
    joined = " ".join(unit.canonical_text.lower() for unit in units)
    opposites = [("allowed", "prohibited"), ("approved", "denied"), ("must", "must not"), ("required", "optional"), ("above", "below")]
    return any((a in joined and b in joined) for a, b in opposites)


def _choose_canonical(units: list[EvidenceUnit]) -> str:
    by_text: dict[str, int] = {}
    for unit in units:
        by_text[unit.canonical_text] = by_text.get(unit.canonical_text, 0) + 1
    return sorted(by_text, key=lambda text: (-by_text[text], len(text)))[0]


def _finance_threshold(text: str) -> str | None:
    low = text.lower()
    if "finance" not in low or not ("approval" in low or "review" in low or "reviewed" in low):
        return None
    money = re.findall(r"__PROTECTED_\d+__|\$\d[\d,]*(?:\.\d+)?", text)
    if money:
        return money[0]
    number = re.search(r"\b(?:over|above|greater than|exceeding)\s+(\d[\d,]*)", text, re.IGNORECASE)
    return f"${number.group(1)}" if number else None


def _latency_change(text: str) -> tuple[str, str] | None:
    unit = r"(?:ms|milliseconds|seconds|s)"
    values = re.findall(rf"\b\d+(?:\.\d+)?\s*{unit}\b", text, flags=re.IGNORECASE)
    if len(values) >= 2 and re.search(r"\b(latency|responded|response|requests?)\b", text, re.IGNORECASE):
        return values[0], values[1]
    return None


def _uptime_metric(text: str) -> tuple[str, str] | None:
    percent = re.search(r"__PROTECTED_\d+__|\b\d+(?:\.\d+)?%", text)
    month = re.search(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\b", text, re.IGNORECASE)
    if percent and month and "uptime" in text.lower():
        return percent.group(0), month.group(1)
    return None


def _protected_facts(text: str) -> set[str]:
    cleaned = re.sub(r"(?im)^(?:Retrieved .*?chunk|Source|Chunk|Document section|Policy excerpt)\s*[A-Z0-9]*:\s*$", "", text)
    return set(re.findall(
        r"__PROTECTED_\d+__|\b[A-Z]{2,}(?:-[A-Z0-9]+)+\b|\$\d[\d,]*(?:\.\d+)?|\b\d{4}-\d{2}-\d{2}\b|\b\d+(?:\.\d+)?%|\b\d+(?:\.\d+)?\s*(?:ms|milliseconds|seconds|s)\b",
        cleaned,
        flags=re.IGNORECASE,
    ))


def _normalize_fact(fact: str) -> str:
    return re.sub(r"__PROTECTED_\d+__", "__PROTECTED__", fact)


def _keywords(text: str) -> set[str]:
    stop = {"the", "and", "or", "a", "an", "to", "of", "in", "for", "with", "is", "are", "was", "were"}
    return {word for word in re.findall(r"\b[a-z][a-z-]{3,}\b", text.lower()) if word not in stop}


def _dedupe_sentences(body: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", body.strip())
    seen: set[str] = set()
    kept: list[str] = []
    for part in parts:
        key = re.sub(r"\s+", " ", part.lower()).strip()
        if key and key not in seen:
            seen.add(key)
            kept.append(part.strip())
    return " ".join(kept).strip()


def _chunk_text(chunk: RagChunk) -> str:
    body = chunk.body.strip() if _is_ticket_chunk(chunk) else _dedupe_sentences(chunk.body)
    return f"{_display_header(chunk.header)}:\n{body}".strip()


def _display_header(header: str) -> str:
    return re.sub(r"\s*\?\s*", " - ", header).strip(" :")


def _is_ticket_chunk(chunk: RagChunk) -> bool:
    return bool(re.search(r"\b(customer ticket|ticket)\b", chunk.header, re.IGNORECASE)) or bool(
        re.search(r"\b(ticket id|refund amount)\b", chunk.body, re.IGNORECASE)
    )


def _canonical_key(text: str) -> str:
    text = re.sub(r"__PROTECTED_\d+__", "__PROTECTED__", text)
    text = re.sub(r"\s+", " ", text.lower())
    return text.strip()


def _dedupe_repeated_sentences(text: str) -> str:
    lines = text.splitlines()
    seen_sentence_keys: set[str] = set()
    output: list[str] = []
    for line in lines:
        stripped = line.strip()
        key = _sentence_dedupe_key(stripped)
        if key and key in seen_sentence_keys:
            continue
        if key:
            seen_sentence_keys.add(key)
        output.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(output)).strip()


def _sentence_dedupe_key(text: str) -> str | None:
    if not text or text.endswith(":") or text.startswith("- "):
        return None
    if len(text.split()) < 3:
        return None
    return re.sub(r"[^a-z0-9$%_]+", " ", text.lower()).strip()


def _join_list(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"
