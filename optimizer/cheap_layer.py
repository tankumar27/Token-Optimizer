from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
import re

from .semantic_validator import SemanticValidator
from .token_counter import count_tokens


VERSION = "cheap_layer_v1_2026_06_19"
BACKEND_NAME = "cheap_layer"


NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000, "million": 1000000,
}
NUMBER_WORD_RE = r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million)(?:[-\s]+(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million)){0,9}"

RISK_MARKERS = {
    "not", "no", "never", "cannot", "can't", "must not", "should not", "do not",
    "however", "but", "except", "unless", "denied", "restricted", "prohibited",
    "failed", "failure", "pending", "blocked", "degraded", "offline", "inactive",
    "read-only", "unavailable", "secret", "credential", "confidential",
}

STATE_WORDS = {
    "degraded", "offline", "inactive", "read-only", "unavailable", "pending",
    "approved", "denied", "retained", "deleted", "disabled", "enabled",
    "paused", "quarantined", "watchlist", "flagged", "blocked", "locked",
}

HEADER_RE = re.compile(
    r"(?im)^((?:Retrieved\s+Context\s+Chunk|Chunk|Source|Document)\s+[A-Z0-9]+(?:\s*[-\u2013\u2014?]\s*[^:]{1,100})?|"
    r"System|Context|Background|Background Note|Long Background Note|"
    r"Task|User Question|Customer Question|Customer Ticket|Ticket|Agent Instructions|Agent Reminder|"
    r"Clinical Operations Instructions|User Request|Business Question|Internal Instructions|System Instructions|"
    r"Developer Instructions|Support Instructions|Policy excerpt|Final instruction|Final instructions):\s*$"
)


@dataclass
class Section:
    label: str
    body: str
    kind: str


def cheap_layer_backend(text: str, level: str = "balanced") -> tuple[str, list[dict]]:
    compressor = CheapLayerCompressor()
    return compressor.compress(text, level)


class CheapLayerCompressor:
    def __init__(self, validator: SemanticValidator | None = None) -> None:
        # Keep the cheap layer actually cheap. Transformer-backed validation is
        # reserved for downstream semantic compression; this layer only removes
        # redundancy when lexical/fact anchors make equivalence obvious.
        self.validator = validator

    def compress(self, text: str, level: str = "balanced") -> tuple[str, list[dict]]:
        before = analyze_text(text)
        sections = parse_sections(text)
        rebuilt: list[str] = []
        surface_changes: list[dict] = []
        semantic_removed: list[dict] = []
        rejected: list[dict] = []

        for section in sections:
            sentences = split_sentences(section.body)
            if section.kind == "variable":
                body = section.body.strip()
            else:
                deduped, conflict_warning_changes = conflict_scope_warning_compress(sentences, section.label)
                surface_changes.extend(conflict_warning_changes)
                sentences = deduped
                deduped, formula_changes = formula_preservation_compress(sentences, section.label)
                surface_changes.extend(formula_changes)
                sentences = deduped
                deduped, changes = exact_dedupe_sentences(sentences, section.label)
                surface_changes.extend(changes)
                deduped, subsumption_changes = subsumed_sentence_compress(deduped, section.label)
                surface_changes.extend(subsumption_changes)
                deduped, frame_changes = frame_compress_sentences(deduped, section.label)
                surface_changes.extend(frame_changes)
                deduped, claim_changes = do_not_claim_frame_compress(deduped, section.label)
                surface_changes.extend(claim_changes)
                deduped, status_changes = same_subject_status_compress(deduped, section.label)
                surface_changes.extend(status_changes)
                deduped, negative_changes = negative_frame_list_compress(deduped, section.label)
                surface_changes.extend(negative_changes)
                deduped, fact_warning_changes = fact_warning_fusion_compress(deduped, section.label)
                surface_changes.extend(fact_warning_changes)
                deduped, list_changes = repeated_frame_list_compress(deduped, section.label)
                surface_changes.extend(list_changes)
                deduped, general_list_changes = general_list_frame_compress(deduped, section.label)
                surface_changes.extend(general_list_changes)
                deduped, list_cleanup_changes = existing_list_source_cleanup(deduped, section.label)
                surface_changes.extend(list_cleanup_changes)
                deduped, removed, rejected_pairs = self.semantic_dedupe_sentences(deduped, section.label, level)
                semantic_removed.extend(removed)
                rejected.extend(rejected_pairs)
                body = " ".join(deduped).strip()
            if body:
                rebuilt.append(render_section(section, body, len(sections)))

        compressed = "\n\n".join(part for part in rebuilt if part).strip()
        compressed = re.sub(r"[ \t]+\n", "\n", compressed)
        compressed = re.sub(r"\n{3,}", "\n\n", compressed)
        gate = validation_gate(text, compressed)
        after = analyze_text(compressed) if gate["passed"] else before
        structural_changes = bool(surface_changes or semantic_removed)
        if not gate["passed"] or not structural_changes or (after["tokens"] >= before["tokens"] and len(compressed) >= len(text)):
            trace = self._trace(text, text, before, before, [], [], rejected, gate, "cheap layer rejected or no token savings", False)
            return text, [trace]
        trace = self._trace(text, compressed, before, after, surface_changes, semantic_removed, rejected, gate, "cheap layer accepted", True)
        return compressed, [trace]

    def semantic_dedupe_sentences(self, sentences: list[str], label: str, level: str) -> tuple[list[str], list[dict], list[dict]]:
        if len(sentences) <= 1:
            return sentences, [], []
        thresholds = {"safe": 0.92, "balanced": 0.88, "aggressive": 0.84}
        threshold = thresholds.get(level, 0.88)
        removed: set[int] = set()
        accepted: list[dict] = []
        rejected: list[dict] = []
        pairs: list[tuple[float, int, int, float, float]] = []
        fact_cache = [canonical_fact_keys(sentence) for sentence in sentences]
        key_cache = [sentence_key(sentence) for sentence in sentences]
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                left_facts = fact_cache[i]
                right_facts = fact_cache[j]
                facts_match = bool(left_facts and left_facts == right_facts)
                keys_match = key_cache[i] == key_cache[j]
                if not facts_match and not keys_match:
                    # Cheap layer semantic removal is only allowed when facts anchor equivalence.
                    # Unanchored paraphrases go to the smarter downstream semantic layer.
                    continue
                lex = lexical_similarity(sentences[i], sentences[j])
                sem = self.validator.similarity(sentences[i], sentences[j])["semantic_similarity"] if self.validator else lex
                score = max(lex, 0.85 * sem + 0.15 * lex)
                if facts_match:
                    score = max(score, 0.95)
                pairs.append((score, i, j, sem, lex))
        for score, i, j, sem, lex in sorted(pairs, reverse=True):
            if i in removed or j in removed or score < threshold:
                continue
            keep_idx, remove_idx = choose_keeper_index(sentences, i, j)
            ok, reasons = can_remove_sentence(sentences[remove_idx], sentences[keep_idx])
            record = {
                "section_label": label,
                "removed_sentence": sentences[remove_idx],
                "kept_sentence": sentences[keep_idx],
                "semantic_similarity": round(sem, 3),
                "lexical_similarity": round(lex, 3),
                "combined_score": round(score, 3),
                "reason": "conservative semantic duplicate" if ok else "semantic duplicate rejected by safety firewall",
            }
            if ok:
                removed.add(remove_idx)
                accepted.append(record)
            else:
                record["rejected_reason"] = "; ".join(reasons)
                rejected.append(record)
        return [sentence for idx, sentence in enumerate(sentences) if idx not in removed], accepted, rejected

    def _trace(
        self,
        original: str,
        compressed: str,
        before: dict,
        after: dict,
        surface_changes: list[dict],
        semantic_removed: list[dict],
        rejected: list[dict],
        gate: dict,
        reason: str,
        accepted: bool,
    ) -> dict:
        return {
            "backend": BACKEND_NAME,
            "candidate_type": "cheap_layer_compression",
            "version": VERSION,
            "accepted": accepted,
            "reason": reason,
            "rejected_reason": None if accepted else reason,
            "span_text": original,
            "retained_span": compressed if accepted else None,
            "removed_span": original if accepted else None,
            "tokens_saved": max(0, before["tokens"] - after["tokens"]),
            "score": round((before["tokens"] - after["tokens"]) / max(1, before["tokens"]), 3),
            "before": before,
            "after": after,
            "surface_changes": surface_changes,
            "semantic_removed": semantic_removed,
            "rejected_removals": rejected,
            "validation_gate": gate,
            "missing_protected_facts": gate["missing_protected_fact_keys"],
            "missing_state_signatures": gate["missing_state_signatures"],
            "final_action": choose_final_action(before, after, accepted),
            "risk_flags": [] if accepted else [reason],
        }


def parse_sections(text: str) -> list[Section]:
    matches = list(HEADER_RE.finditer(text))
    if not matches:
        return [Section("FULL_PROMPT", text.strip(), "context")]
    sections: list[Section] = []
    if matches[0].start() > 0:
        preface = text[:matches[0].start()].strip()
        if preface:
            sections.append(Section("PREFACE", preface, "instruction"))
    for idx, match in enumerate(matches):
        label = match.group(1).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if not body:
            continue
        low = label.lower()
        kind = "variable" if re.search(r"question|task|ticket|request", low) else "instruction" if re.search(r"instructions?|reminder|system", low) else "context"
        sections.append(Section(label, body, kind))
    return sections


def render_section(section: Section, body: str, count: int) -> str:
    if section.label in {"FULL_PROMPT", "PREFACE"}:
        return body
    return f"{section.label}:\n{body.strip()}"


def split_sentences(text: str) -> list[str]:
    pieces = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [piece.strip() for piece in pieces if piece.strip()]


def exact_dedupe_sentences(sentences: list[str], label: str) -> tuple[list[str], list[dict]]:
    seen: set[str] = set()
    kept: list[str] = []
    changes: list[dict] = []
    for sentence in sentences:
        key = sentence_key(sentence)
        if key in seen:
            changes.append({
                "type": "exact_sentence_duplicate",
                "section_label": label,
                "removed_sentence": sentence,
                "reason": "exact normalized duplicate sentence",
            })
            continue
        seen.add(key)
        kept.append(sentence)
    return kept, changes


CONFLICT_WARNING_RE = re.compile(
    r"^(?P<prefix>(?:Do not|Never)\s+(?:automatically\s+)?(?:resolve|merge|reconcile|collapse|dedupe|deduplicate|combine|erase|remove|ignore|hide)\s+)"
    r"(?P<det>this|that|the|a|an)\s+"
    r"(?P<noun>conflict|contradiction|discrepancy|mismatch|inconsistency)"
    r"(?P<suffix>[^.!?]*)(?P<punct>[.!?])$",
    re.I,
)

CONFLICT_SCOPE_RE = re.compile(
    r"\b(conflict|contradiction|contradicts?|discrepanc(?:y|ies)|mismatch(?:es)?|inconsisten(?:t|cy|cies)|"
    r"disagree(?:s|ment)?|differs?|versus|vs\.?|but|however|on the other hand)\b",
    re.I,
)


def conflict_scope_warning_compress(sentences: list[str], label: str) -> tuple[list[str], list[dict]]:
    groups: dict[tuple[str, str, str], list[tuple[int, str, re.Match]]] = {}
    for idx, sentence in enumerate(sentences):
        match = CONFLICT_WARNING_RE.match(sentence.strip())
        if not match:
            continue
        key = (
            normalize_text(match.group("prefix")),
            match.group("noun").lower(),
            normalize_text(match.group("suffix")),
        )
        groups.setdefault(key, []).append((idx, sentence, match))

    replacements: dict[int, str] = {}
    consumed: set[int] = set()
    changes: list[dict] = []
    conflict_context = sum(1 for sentence in sentences if CONFLICT_SCOPE_RE.search(sentence) and not CONFLICT_WARNING_RE.match(sentence.strip()))
    for entries in groups.values():
        if len(entries) < 3:
            continue
        if conflict_context < 2:
            continue
        first_idx, _, first_match = entries[0]
        plural = _plural_conflict_noun(first_match.group("noun"))
        replacement = (
            f"{first_match.group('prefix')}these {plural}"
            f"{first_match.group('suffix')}{first_match.group('punct')}"
        )
        replacements[first_idx] = replacement
        consumed.update(idx for idx, _, _ in entries)
        consumed.remove(first_idx)
        changes.append({
            "type": "conflict_scope_warning_compression",
            "section_label": label,
            "removed_sentence": " ".join(sentence for _, sentence, _ in entries),
            "kept_sentence": replacement,
            "reason": "repeated singular contradiction warning scoped to multiple conflict records",
            "conflict_records_seen": conflict_context,
        })

    if not replacements:
        return sentences, []
    output: list[str] = []
    for idx, sentence in enumerate(sentences):
        if idx in replacements:
            output.append(replacements[idx])
        elif idx not in consumed:
            output.append(sentence)
    return output, changes


def _plural_conflict_noun(noun: str) -> str:
    low = noun.lower()
    if low == "discrepancy":
        return "discrepancies"
    if low == "inconsistency":
        return "inconsistencies"
    return low + "s"


FORMULA_APPEARS_RE = re.compile(r"^Formula\s+(?P<formula>.+?)\s+appears in\s+.+?[.]?$", re.I)
FORMULA_PRESERVE_RE = re.compile(r"^Formula\s+(?P<formula>.+?)\s+must be preserved exactly[.]?$", re.I)


def formula_preservation_compress(sentences: list[str], label: str) -> tuple[list[str], list[dict]]:
    appears: dict[str, list[tuple[int, str]]] = {}
    preserves: dict[str, list[tuple[int, str]]] = {}
    for idx, sentence in enumerate(sentences):
        match = FORMULA_APPEARS_RE.match(sentence.strip())
        if match:
            appears.setdefault(normalize_text(match.group("formula")), []).append((idx, sentence))
            continue
        match = FORMULA_PRESERVE_RE.match(sentence.strip())
        if match:
            preserves.setdefault(normalize_text(match.group("formula")), []).append((idx, sentence))

    remove: set[int] = set()
    changes: list[dict] = []
    for formula_key, appear_entries in appears.items():
        preserve_entries = preserves.get(formula_key)
        if not preserve_entries:
            continue
        kept = preserve_entries[0][1]
        for idx, sentence in appear_entries:
            ok, reasons = can_remove_sentence(sentence, kept)
            if not ok:
                continue
            remove.add(idx)
            changes.append({
                "type": "formula_preservation_compression",
                "section_label": label,
                "removed_sentence": sentence,
                "kept_sentence": kept,
                "reason": "formula appearance note subsumed by exact-preservation instruction",
            })
    if not remove:
        return sentences, []
    return [sentence for idx, sentence in enumerate(sentences) if idx not in remove], changes


def subsumed_sentence_compress(sentences: list[str], label: str) -> tuple[list[str], list[dict]]:
    removed: set[int] = set()
    changes: list[dict] = []
    normalized = [normalize_text(sentence).rstrip(".") for sentence in sentences]
    for i, short in enumerate(normalized):
        if i in removed or len(short.split()) < 3:
            continue
        for j, long in enumerate(normalized):
            if i == j or j in removed:
                continue
            if long.startswith(short + " because") and can_remove_sentence(sentences[i], sentences[j])[0]:
                removed.add(i)
                changes.append({
                    "type": "subsumed_sentence_duplicate",
                    "section_label": label,
                    "removed_sentence": sentences[i],
                    "kept_sentence": sentences[j],
                    "reason": "short restatement subsumed by detailed because-clause",
                })
                break
    if not removed:
        return sentences, []
    return [sentence for idx, sentence in enumerate(sentences) if idx not in removed], changes


def frame_compress_sentences(sentences: list[str], label: str) -> tuple[list[str], list[dict]]:
    # Conservative frame compression: "Agents must verify X. Agents must verify Y." -> one list.
    groups: dict[str, list[str]] = {}
    originals: dict[str, list[str]] = {}
    passthrough: list[str] = []
    for sentence in sentences:
        match = re.match(r"^(?P<prefix>Agents? must verify|Agents? should verify|Support agents must confirm)\s+(?P<item>.+?)[.]?$", sentence, re.I)
        if not match:
            passthrough.append(sentence)
            continue
        prefix = "Agents must verify"
        item = re.sub(r"^(the)\s+", "", match.group("item").strip(), flags=re.I)
        groups.setdefault(prefix, []).append(item)
        originals.setdefault(prefix, []).append(sentence)
    changes: list[dict] = []
    compressed: list[str] = []
    for prefix, items in groups.items():
        normalized_items = []
        for item in items:
            for part in re.split(r",\s*| and ", item):
                cleaned = part.strip(" .")
                if cleaned and cleaned not in normalized_items:
                    normalized_items.append(cleaned)
        if len(originals[prefix]) >= 2 or len(normalized_items) >= 2:
            sentence = f"{prefix} {join_list(normalized_items)}."
            compressed.append(sentence)
            changes.append({
                "type": "frame_list_compression",
                "section_label": label,
                "removed_sentence": " ".join(originals[prefix]),
                "kept_sentence": sentence,
                "reason": "shared sentence frame with listable items",
            })
        else:
            compressed.extend(originals[prefix])
    return passthrough + compressed, changes


DO_NOT_CLAIM_RE = re.compile(
    r"^Do not claim\s+(?P<subject>.+?)\s+(?P<predicate>is healthy|is approved|is inactive|is active|completed|was deleted|was migrated|was skipped|was patched|is writable)[.]?$",
    re.I,
)


def do_not_claim_frame_compress(sentences: list[str], label: str) -> tuple[list[str], list[dict]]:
    groups: dict[str, list[tuple[int, str, str]]] = {}
    for idx, sentence in enumerate(sentences):
        match = DO_NOT_CLAIM_RE.match(sentence.strip())
        if not match:
            continue
        predicate = match.group("predicate").lower()
        subject = match.group("subject").strip()
        groups.setdefault(predicate, []).append((idx, subject, sentence))

    replacements: dict[int, str] = {}
    consumed: set[int] = set()
    changes: list[dict] = []
    for predicate, entries in groups.items():
        subjects: list[str] = []
        for _, subject, _ in entries:
            if subject not in subjects:
                subjects.append(subject)
        if len(subjects) < 2:
            continue
        first_idx = entries[0][0]
        aux_predicate = predicate
        if predicate.startswith("is "):
            aux_predicate = "are " + predicate[3:]
        elif predicate.startswith("was "):
            aux_predicate = "were " + predicate[4:]
        replacement = f"Do not claim {join_list(subjects)} {aux_predicate}."
        if count_tokens(replacement) >= sum(count_tokens(sentence) for _, _, sentence in entries):
            continue
        replacements[first_idx] = replacement
        consumed.update(idx for idx, _, _ in entries)
        consumed.remove(first_idx)
        changes.append({
            "type": "do_not_claim_frame_compression",
            "section_label": label,
            "removed_sentence": " ".join(sentence for _, _, sentence in entries),
            "kept_sentence": replacement,
            "reason": "same do-not-claim predicate repeated across listable subjects",
        })

    if not replacements:
        return sentences, []
    output: list[str] = []
    for idx, sentence in enumerate(sentences):
        if idx in replacements:
            output.append(replacements[idx])
        elif idx not in consumed:
            output.append(sentence)
    return output, changes


FRAME_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"^Cohort\s+(?P<item>.+?)\s+was affected[.]?$", re.I), "Affected cohorts", "list_colon"),
    (re.compile(r"^Cell\s+(?P<item>.+?)\s+was affected[.]?$", re.I), "Affected cells", "list_colon"),
    (re.compile(r"^Airport\s+(?P<item>.+?)\s+was affected[.]?$", re.I), "Affected airports", "list_colon"),
    (re.compile(r"^Gate\s+(?P<item>.+?)\s+was affected[.]?$", re.I), "Affected gates", "list_colon"),
    (re.compile(r"^(?P<item>Zone-[A-Z0-9-]+)\s+routing tables were migrated[.]?$", re.I), "routing tables were migrated", "suffix_predicate"),
    (re.compile(r"^(?P<item>Class-[A-Z0-9-]+)\s+seating records were migrated[.]?$", re.I), "Migrated seating classes", "list_colon"),
    (re.compile(r"^(?P<item>Class-[A-Z0-9-]+)\s+seating records were not migrated[.]?$", re.I), "Not-migrated seating classes", "list_colon"),
    (re.compile(r"^Flight\s+(?P<item>.+?)\s+is operational[.]?$", re.I), "Operational flights", "list_colon"),
    (re.compile(r"^Passenger\s+(?P<item>.+?)\s+baggage was recovered[.]?$", re.I), "Recovered passenger baggage", "list_colon"),
    (re.compile(r"^Baggage\s+(?P<item>.+?)\s+was recovered[.]?$", re.I), "Recovered baggage IDs", "list_colon"),
    (re.compile(r"^The\s+(?P<item>.+?)\s+note pool is excluded from migration[.]?$", re.I), "Excluded note pools", "list_colon"),
    (re.compile(r"^Biomarker\s+(?P<item>.+?)\s+status is healthy[.]?$", re.I), "Healthy biomarkers", "list_colon"),
    (re.compile(r"^Biomarker\s+(?P<item>.+?)\s+is healthy[.]?$", re.I), "Healthy biomarkers", "list_colon"),
    (re.compile(r"^Service\s+(?P<item>.+?)\s+is online[.]?$", re.I), "Online services", "list_colon"),
    (re.compile(r"^Drug\s+(?P<item>.+?)\s+is active[.]?$", re.I), "Active drugs", "list_colon"),
    (re.compile(r"^Approved\s+(?P<item>.+?)[.]?$", re.I), "Approved findings", "list_colon"),
]


@dataclass
class GeneralListSlot:
    index: int
    sentence: str
    entity_type: str
    entity_id: str
    owned_object: str
    predicate: str
    state: str
    polarity: str
    role: str
    count: str = ""
    complaint_kind: str = ""

    @property
    def frame_key(self) -> tuple[str, str, str, str, str, str]:
        return (
            _canonical_entity_type(self.entity_type),
            normalize_text(self.owned_object),
            self.predicate,
            self.state,
            self.polarity,
            self.role,
        )

    @property
    def signature(self) -> str:
        if self.predicate == "logged":
            return "::".join([
                "logged",
                normalize_text(self.entity_type),
                normalize_text(self.count),
                normalize_text(self.complaint_kind),
                self.role,
            ])
        return "::".join([
            "slot",
            _canonical_entity_type(self.entity_type),
            normalize_text(self.entity_id),
            normalize_text(self.owned_object),
            self.predicate,
            self.state,
            self.polarity,
            self.role,
        ])


GENERAL_LIST_PREDICATES = {
    "was affected": ("were affected", "affected", "positive"),
    "were affected": ("were affected", "affected", "positive"),
    "was reviewed": ("were reviewed", "reviewed", "positive"),
    "were reviewed": ("were reviewed", "reviewed", "positive"),
    "was migrated": ("were migrated", "migrated", "positive"),
    "were migrated": ("were migrated", "migrated", "positive"),
    "was tested": ("were tested", "tested", "positive"),
    "were tested": ("were tested", "tested", "positive"),
    "was retained": ("were retained", "retained", "positive"),
    "were retained": ("were retained", "retained", "positive"),
    "was restarted": ("were restarted", "restarted", "positive"),
    "were restarted": ("were restarted", "restarted", "positive"),
    "was drained": ("were drained", "drained", "positive"),
    "were drained": ("were drained", "drained", "positive"),
    "was scanned": ("were scanned", "scanned", "positive"),
    "were scanned": ("were scanned", "scanned", "positive"),
}


def general_list_frame_compress(sentences: list[str], label: str) -> tuple[list[str], list[dict]]:
    groups: dict[tuple[str, str, str, str, str, str], list[GeneralListSlot]] = {}
    for idx, sentence in enumerate(sentences):
        slot = parse_general_list_slot(sentence, idx)
        if slot:
            groups.setdefault(slot.frame_key, []).append(slot)

    replacements: dict[int, str] = {}
    consumed: set[int] = set()
    changes: list[dict] = []
    for group in groups.values():
        group_indices = {slot.index for slot in group}
        eligible_group = [
            slot for slot in group
            if not _list_item_has_blocking_external_mention(slot.entity_id, sentences, group_indices)
        ]
        unique_group: list[GeneralListSlot] = []
        seen: set[str] = set()
        for slot in eligible_group:
            if slot.signature not in seen:
                seen.add(slot.signature)
                unique_group.append(slot)
        if len(unique_group) < 2:
            continue

        entry_indices = {slot.index for slot in eligible_group}
        item_ids = [slot.entity_id for slot in unique_group if slot.entity_id]
        if item_ids and _list_items_have_external_mentions(item_ids, sentences, entry_indices):
            continue
        replacement = render_general_list_frame(unique_group)
        if not replacement:
            continue
        original_text = " ".join(slot.sentence for slot in unique_group)
        if not general_list_slots_preserved(original_text, replacement):
            continue
        if not _list_frame_preserves_signatures(original_text, replacement):
            continue
        original_token_sum = sum(count_tokens(slot.sentence) for slot in unique_group)
        original_char_sum = sum(len(slot.sentence) for slot in unique_group)
        if count_tokens(replacement) >= original_token_sum and len(replacement) >= original_char_sum:
            continue

        first_idx = unique_group[0].index
        replacements[first_idx] = replacement
        consumed.update(slot.index for slot in eligible_group if slot.index != first_idx)
        changes.append({
            "type": "general_list_frame_compression",
            "section_label": label,
            "removed_sentence": original_text,
            "kept_sentence": replacement,
            "reason": "same safe slot frame repeated across listable entities",
        })

    if not replacements:
        return sentences, []
    output: list[str] = []
    for idx, sentence in enumerate(sentences):
        if idx in replacements:
            output.append(replacements[idx])
        elif idx not in consumed:
            output.append(sentence)
    return output, changes


def existing_list_source_cleanup(sentences: list[str], label: str) -> tuple[list[str], list[dict]]:
    covered: dict[str, tuple[int, str]] = {}
    for idx, sentence in enumerate(sentences):
        rendered_slots = parse_rendered_general_list_sentence(sentence, idx)
        if len(rendered_slots) < 2:
            continue
        for slot in rendered_slots:
            covered[slot.signature] = (idx, sentence)

    if not covered:
        return sentences, []

    consumed: set[int] = set()
    grouped_removed: dict[int, list[str]] = {}
    for idx, sentence in enumerate(sentences):
        slot = parse_general_list_slot(sentence, idx)
        if not slot or slot.signature not in covered:
            continue
        keeper_idx, keeper_sentence = covered[slot.signature]
        if keeper_idx == idx:
            continue
        consumed.add(idx)
        grouped_removed.setdefault(keeper_idx, []).append(sentence)

    if not consumed:
        return sentences, []

    output = [sentence for idx, sentence in enumerate(sentences) if idx not in consumed]
    changes = [
        {
            "type": "existing_list_source_cleanup",
            "section_label": label,
            "removed_sentence": " ".join(removed),
            "kept_sentence": sentences[keeper_idx],
            "reason": "individual source sentences already represented by existing same-frame list",
        }
        for keeper_idx, removed in grouped_removed.items()
    ]
    return output, changes


def parse_general_list_slot(sentence: str, idx: int = -1) -> GeneralListSlot | None:
    raw = sentence.strip()
    if not raw or _is_conflict_record(raw):
        return None
    if re.search(r"\bdo not\b|\bnever\b|\bexcept\b|\bexcluded\b|\bunavailable\b|\bnot\b", raw, re.I):
        return None

    complaint = re.match(
        rf"^(?P<actor>[A-Z][A-Za-z ]{{1,40}})\s+logged\s+(?P<count>__PROTECTED_\d+__|\d+|{NUMBER_WORD_RE})\s+(?P<kind>[A-Za-z0-9][A-Za-z0-9 -]{{1,80}}?)\s+complaints[.]?$",
        raw,
        re.I,
    )
    if complaint:
        actor = complaint.group("actor").strip()
        kind = complaint.group("kind").strip()
        return GeneralListSlot(
            index=idx,
            sentence=raw,
            entity_type=actor,
            entity_id=kind,
            owned_object="complaints",
            predicate="logged",
            state="logged",
            polarity="positive",
            role=sentence_role(raw),
            count=normalize_number_text(complaint.group("count").strip()),
            complaint_kind=kind,
        )

    predicate_options = "|".join(re.escape(key) for key in sorted(GENERAL_LIST_PREDICATES, key=len, reverse=True))
    match = re.match(rf"^(?P<head>.+?)\s+(?P<predicate>{predicate_options})[.]?$", raw, re.I)
    if not match:
        return None
    render_predicate, state, polarity = GENERAL_LIST_PREDICATES[match.group("predicate").lower()]
    subject = split_general_list_subject(match.group("head").strip())
    if not subject:
        return None
    entity_type, entity_id, owned_object = subject
    if not entity_id or entity_id.lower() in {"it", "this", "that", "they", "these", "those"}:
        return None
    return GeneralListSlot(
        index=idx,
        sentence=raw,
        entity_type=entity_type,
        entity_id=entity_id,
        owned_object=owned_object,
        predicate=render_predicate,
        state=state,
        polarity=polarity,
        role=sentence_role(raw),
    )


def split_general_list_subject(head: str) -> tuple[str, str, str] | None:
    tokens = head.split()
    if len(tokens) < 2:
        return None
    owned_object = ""
    entity_id = tokens[-1]
    entity_type_tokens = tokens[:-1]
    if len(tokens) >= 3 and _looks_like_entity_id(tokens[-2]) and _looks_like_owned_object(tokens[-1]):
        entity_id = tokens[-2]
        owned_object = tokens[-1]
        entity_type_tokens = tokens[:-2]
    entity_type = " ".join(entity_type_tokens).strip()
    if not entity_type:
        return None
    return entity_type, entity_id.strip(","), owned_object


def render_general_list_frame(group: list[GeneralListSlot]) -> str | None:
    first = group[0]
    if first.predicate == "logged" and first.owned_object == "complaints":
        items = [f"{slot.count} {slot.complaint_kind} complaints" for slot in group]
        return f"{first.entity_type} logged {join_list(items)}."
    items = [slot.entity_id for slot in group]
    owned = f" {first.owned_object}" if first.owned_object else ""
    return f"{pluralize_entity_type(first.entity_type)} {join_list(items)}{owned} {first.predicate}."


def general_list_slots_preserved(original_text: str, replacement: str) -> bool:
    original = {slot.signature for slot in parse_general_list_slots_from_text(original_text)}
    rendered = {slot.signature for slot in parse_general_list_slots_from_text(replacement)}
    return bool(original) and original <= rendered


def parse_general_list_slots_from_text(text: str) -> list[GeneralListSlot]:
    slots: list[GeneralListSlot] = []
    for idx, sentence in enumerate(split_sentences(text)):
        rendered = parse_rendered_general_list_sentence(sentence, idx)
        if rendered:
            slots.extend(rendered)
            continue
        slot = parse_general_list_slot(sentence, idx)
        if slot:
            slots.append(slot)
    return slots


def parse_rendered_general_list_sentence(sentence: str, idx: int = -1) -> list[GeneralListSlot]:
    raw = sentence.strip()
    complaint = re.match(r"^(?P<actor>[A-Z][A-Za-z ]{1,40})\s+logged\s+(?P<items>.+?)\s+complaints[.]?$", raw, re.I)
    if complaint:
        actor = complaint.group("actor").strip()
        output: list[GeneralListSlot] = []
        for item in split_rendered_items(complaint.group("items")):
            match = re.match(rf"^(?P<count>__PROTECTED_\d+__|\d+|{NUMBER_WORD_RE})\s+(?P<kind>.+?)(?:\s+complaints)?$", item, re.I)
            if not match:
                return []
            output.append(GeneralListSlot(
                index=idx,
                sentence=raw,
                entity_type=actor,
                entity_id=match.group("kind").strip(),
                owned_object="complaints",
                predicate="logged",
                state="logged",
                polarity="positive",
                role=sentence_role(raw),
                count=normalize_number_text(match.group("count").strip()),
                complaint_kind=match.group("kind").strip(),
            ))
        return output

    predicate_options = "|".join(re.escape(value[0]) for value in sorted(set(GENERAL_LIST_PREDICATES.values()), key=lambda x: len(x[0]), reverse=True))
    match = re.match(rf"^(?P<entity_type>[A-Z][A-Za-z0-9&/ -]{{1,60}}?)\s+(?P<body>.+?)\s+(?P<predicate>{predicate_options})[.]?$", raw, re.I)
    if not match:
        return []
    entity_type = singularize_entity_type(match.group("entity_type").strip())
    body = match.group("body").strip()
    body_tokens = body.split()
    if len(body_tokens) >= 2 and body_tokens[0].islower() and body_tokens[0].endswith("s") and _looks_like_entity_id(body_tokens[1]):
        entity_type = singularize_entity_type(f"{entity_type} {body_tokens[0]}")
        body = " ".join(body_tokens[1:])
    predicate = match.group("predicate").lower()
    owned_object = ""
    if body.endswith(" data"):
        body = body[:-5].strip()
        owned_object = "data"
    state = next((state for render, state, _ in GENERAL_LIST_PREDICATES.values() if render == predicate), predicate.split()[-1])
    polarity = next((polarity for render, _, polarity in GENERAL_LIST_PREDICATES.values() if render == predicate), "positive")
    return [
        GeneralListSlot(
            index=idx,
            sentence=raw,
            entity_type=entity_type,
            entity_id=item,
            owned_object=owned_object,
            predicate=predicate,
            state=state,
            polarity=polarity,
            role=sentence_role(raw),
        )
        for item in split_rendered_items(body)
        if item
    ]


def split_rendered_items(text: str) -> list[str]:
    cleaned = re.sub(r"\s+and\s+", ", ", text.strip())
    return [part.strip(" ,") for part in cleaned.split(",") if part.strip(" ,")]


def _looks_like_entity_id(value: str) -> bool:
    return bool(
        re.search(r"[A-Z0-9/]", value)
        and (re.search(r"[-_/0-9]", value) or value.isupper() or "/" in value)
    )


def _looks_like_owned_object(value: str) -> bool:
    return bool(re.match(r"^[a-z][a-z0-9_-]{1,30}s?$", value))


def _canonical_entity_type(value: str) -> str:
    return singularize_entity_type(normalize_text(value))


def singularize_entity_type(value: str) -> str:
    cleaned = value.strip()
    low = cleaned.lower()
    irregular = {"policies": "policy", "companies": "company", "hashes": "hash", "classes": "class"}
    if low in irregular:
        return irregular[low]
    if low.endswith("ies") and len(low) > 4:
        return cleaned[:-3] + "y"
    if low.endswith("es") and len(low) > 3 and low[-3] in {"s", "x", "z"}:
        return cleaned[:-2]
    if low.endswith("s") and not low.endswith("ss") and len(low) > 1:
        return cleaned[:-1]
    return cleaned


def pluralize_entity_type(value: str) -> str:
    cleaned = value.strip()
    low = cleaned.lower()
    irregular = {"policy": "Policies", "company": "Companies", "hash": "Hashes", "class": "Classes"}
    if low in irregular:
        return irregular[low]
    if cleaned.endswith("y") and len(cleaned) > 1 and cleaned[-2].lower() not in "aeiou":
        return cleaned[:-1] + "ies"
    if cleaned.endswith(("s", "x", "z")):
        return cleaned + "es"
    return cleaned + "s"


def repeated_frame_list_compress(sentences: list[str], label: str) -> tuple[list[str], list[dict]]:
    groups: dict[str, dict[str, object]] = {}
    consumed: set[int] = set()
    for idx, sentence in enumerate(sentences):
        for pattern, frame, render_mode in FRAME_PATTERNS:
            match = pattern.match(sentence.strip())
            if not match:
                continue
            item = match.group("item").strip(" .")
            if item:
                group = groups.setdefault(frame, {"render_mode": render_mode, "entries": []})
                group["entries"].append((idx, item, sentence))  # type: ignore[index, union-attr]
            break

    replacements: dict[int, str] = {}
    changes: list[dict] = []
    for frame, group in groups.items():
        entries = group["entries"]  # type: ignore[index]
        render_mode = group["render_mode"]  # type: ignore[index]
        items: list[str] = []
        originals: list[str] = []
        for idx, item, sentence in entries:
            originals.append(sentence)
            if item not in items:
                items.append(item)
        if len(items) < 2 or len(entries) < 2:
            continue
        entry_indices = {idx for idx, _, _ in entries}
        if _list_items_have_external_mentions(items, sentences, entry_indices):
            continue
        first_idx = entries[0][0]
        replacement = f"{join_list(items)} {frame}." if render_mode == "suffix_predicate" else f"{frame}: {join_list(items)}."
        if count_tokens(replacement) >= sum(count_tokens(sentence) for _, _, sentence in entries):
            continue
        original_text = " ".join(originals)
        if not _list_frame_preserves_signatures(original_text, replacement):
            continue
        replacements[first_idx] = replacement
        consumed.update(idx for idx, _, _ in entries)
        consumed.remove(first_idx)
        changes.append({
            "type": "repeated_frame_list_compression",
            "section_label": label,
            "removed_sentence": " ".join(originals),
            "kept_sentence": replacement,
            "reason": "same predicate repeated across listable entities",
        })

    if not replacements:
        return sentences, []
    output: list[str] = []
    for idx, sentence in enumerate(sentences):
        if idx in replacements:
            output.append(replacements[idx])
        elif idx not in consumed:
            output.append(sentence)
    return output, changes


def _list_frame_preserves_signatures(original_text: str, replacement: str) -> bool:
    return (
        canonical_fact_keys(original_text) <= canonical_fact_keys(replacement)
        and state_signatures(original_text) <= state_signatures(replacement)
        and event_signatures(original_text) <= event_signatures(replacement)
        and risk_keys(original_text) <= risk_keys(replacement)
    )


def _list_items_have_external_mentions(items: list[str], sentences: list[str], entry_indices: set[int]) -> bool:
    item_keys = [normalize_text(item) for item in items if normalize_text(item)]
    if not item_keys:
        return False
    for idx, sentence in enumerate(sentences):
        if idx in entry_indices:
            continue
        sentence_norm = normalize_text(sentence)
        if not any(_normalized_text_contains_item(sentence_norm, item) for item in item_keys):
            continue
        if _external_sentence_blocks_list_item(sentence):
            return True
    return False


def _list_item_has_blocking_external_mention(item: str, sentences: list[str], entry_indices: set[int]) -> bool:
    item_key = normalize_text(item)
    if not item_key:
        return False
    for idx, sentence in enumerate(sentences):
        if idx in entry_indices:
            continue
        if not _normalized_text_contains_item(normalize_text(sentence), item_key):
            continue
        if _external_sentence_blocks_list_item(sentence):
            return True
    return False


def _normalized_text_contains_item(text: str, item: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(item)}(?!\w)", text))


def _external_sentence_blocks_list_item(sentence: str) -> bool:
    low = sentence.lower()
    if re.search(r"\b(also called|alias|same as|two separate|separate routes|separate items|do not count|count .* as two)\b", low):
        return True
    if re.search(r"\bdo not\s+compress\b|\bdo not\s+merge\b|\bdo not\s+combine\b", low):
        return True
    if "after draining" in low:
        return False
    if re.match(r"^\s*(do not|never)\b", sentence, re.I):
        return bool(re.search(r"\b(affected|reviewed|migrated|tested|retained|drained|restarted|excluded|unavailable|inactive|active|pending|approved|denied|deleted|modified|exposed|degraded|paused|quarantined|watchlist|healthy)\b", low))
    return bool(re.search(
        r"\b(is|was|were|still|later)\b[^.!?;:]{0,80}\b(degraded|paused|quarantined|watchlist|excluded|unavailable|inactive|pending|denied|blocked|failed|locked|not\s+\w+|alias|also called)\b",
        low,
    ))


STATUS_PATTERN = re.compile(
    r"^(?P<subject>(?!Do not\b).+?)\s+"
    r"(?P<predicate>"
    r"is inactive|is active|is degraded|is not offline|is not disabled|is not deleted|"
    r"is read-only|status is unavailable|still exists|"
    r"were migrated|was migrated|were retained|was retained|were not deleted|was not deleted|were not migrated|was not migrated|"
    r"is pending|is not approved|is stale|is locked|is not ready|is blocked"
    r")[.]?$",
    re.I,
)


def same_subject_status_compress(sentences: list[str], label: str) -> tuple[list[str], list[dict]]:
    groups: dict[str, list[tuple[int, str, str]]] = {}
    for idx, sentence in enumerate(sentences):
        match = STATUS_PATTERN.match(sentence.strip())
        if not match:
            continue
        subject = match.group("subject").strip()
        predicate = match.group("predicate").lower()
        groups.setdefault(normalize_text(subject), []).append((idx, subject, predicate))

    replacements: dict[int, str] = {}
    consumed: set[int] = set()
    changes: list[dict] = []
    for entries in groups.values():
        predicates: list[str] = []
        for _, _, predicate in entries:
            if predicate not in predicates:
                predicates.append(predicate)
        if _status_predicates_conflict(predicates):
            continue
        if len(predicates) < 2:
            continue
        first_idx, subject, _ = entries[0]
        replacement = f"{subject} {join_predicates(predicates)}."
        if count_tokens(replacement) >= sum(count_tokens(sentences[idx]) for idx, *_ in entries):
            continue
        replacements[first_idx] = replacement
        consumed.update(idx for idx, *_ in entries)
        consumed.remove(first_idx)
        changes.append({
            "type": "same_subject_status_compression",
            "section_label": label,
            "removed_sentence": " ".join(sentences[idx] for idx, *_ in entries),
            "kept_sentence": replacement,
            "reason": "same subject repeated with listable status predicates",
        })

    if not replacements:
        return sentences, []
    output: list[str] = []
    for idx, sentence in enumerate(sentences):
        if idx in replacements:
            output.append(replacements[idx])
        elif idx not in consumed:
            output.append(sentence)
    return output, changes


def _status_predicates_conflict(predicates: list[str]) -> bool:
    statuses = set(predicates)
    conflicting_pairs = [
        ("is active", "is inactive"),
        ("is approved", "is denied"),
        ("was migrated", "was not migrated"),
        ("were migrated", "were not migrated"),
        ("was deleted", "was not deleted"),
        ("were deleted", "were not deleted"),
        ("is offline", "is not offline"),
        ("is disabled", "is not disabled"),
        ("is ready", "is not ready"),
    ]
    return any(left in statuses and right in statuses for left, right in conflicting_pairs)


def negative_frame_list_compress(sentences: list[str], label: str) -> tuple[list[str], list[dict]]:
    groups: dict[str, list[tuple[int, str, str, str, str]]] = {}
    for idx, sentence in enumerate(sentences):
        stripped = sentence.strip()
        match = re.match(r"^No\s+(?P<subject>.+?)\s+(?P<aux>were|was|are|is)\s+(?P<state>[A-Za-z-]+)[.]?$", stripped, re.I)
        if match:
            subject = match.group("subject").strip()
            aux = match.group("aux").lower()
            state = match.group("state").lower()
            groups.setdefault(subject.lower(), []).append((idx, subject, aux, state, "no"))
            continue
        match = re.match(r"^(?P<subject>.+?)\s+(?P<aux>were|was|are|is)\s+not\s+(?P<state>[A-Za-z-]+)[.]?$", stripped, re.I)
        if match:
            subject = match.group("subject").strip()
            aux = match.group("aux").lower()
            state = match.group("state").lower()
            groups.setdefault(subject.lower(), []).append((idx, subject, aux, state, "not"))

    replacements: dict[int, str] = {}
    consumed: set[int] = set()
    changes: list[dict] = []
    for _, entries in groups.items():
        states: list[str] = []
        for _, _, _, state, _ in entries:
            if state not in states:
                states.append(state)
        if len(entries) < 2 or (len(states) < 2 and {mode for *_, mode in entries} != {"no", "not"}):
            continue
        first_idx, subject, aux, _, _ = entries[0]
        modes = {mode for *_, mode in entries}
        if modes == {"not"}:
            replacement = f"{subject} {aux} not " + f" and {aux} not ".join(states) + "."
        else:
            replacement = f"No {subject} {aux} {join_or_list(states)}."
        if count_tokens(replacement) >= sum(count_tokens(sentences[idx]) for idx, *_ in entries):
            continue
        replacements[first_idx] = replacement
        consumed.update(idx for idx, *_ in entries)
        consumed.remove(first_idx)
        changes.append({
            "type": "negative_frame_list_compression",
            "section_label": label,
            "removed_sentence": " ".join(sentences[idx] for idx, *_ in entries),
            "kept_sentence": replacement,
            "reason": "same negative safety frame repeated across states",
        })

    if not replacements:
        return sentences, []
    output: list[str] = []
    for idx, sentence in enumerate(sentences):
        if idx in replacements:
            output.append(replacements[idx])
        elif idx not in consumed:
            output.append(sentence)
    return output, changes


NEGATED_FACT_RE = re.compile(
    r"^(?P<subject>.+?)\s+(?P<aux>is|are|was|were)\s+not\s+"
    r"(?P<predicate>exposed|deleted|modified|approved|migrated|rotated|patched|offline|disabled|included|ready|mapped)[.]?$",
    re.I,
)

NO_FACT_RE = re.compile(
    r"^No\s+(?P<subject>.+?)\s+(?P<aux>is|are|was|were)\s+"
    r"(?P<predicate>exposed|deleted|modified|approved|migrated|rotated|patched|offline|disabled|included|ready|mapped)[.]?$",
    re.I,
)

DO_NOT_CLAIM_FACT_RE = re.compile(
    r"^Do not claim\s+(?P<subject>.+?)\s+(?P<aux>is|are|was|were)\s+"
    r"(?P<predicate>exposed|deleted|modified|approved|migrated|rotated|patched|offline|disabled|included|ready|mapped)[.]?$",
    re.I,
)

FUSABLE_NEGATED_PREDICATES = {
    "exposed", "deleted", "modified", "migrated",
    "rotated", "patched", "offline", "disabled", "included", "mapped",
}

UNFUSIBLE_FACT_STATE_CONTEXT = {
    "unavailable", "pending", "approved", "denied", "active", "inactive",
}


def fact_warning_fusion_compress(sentences: list[str], label: str) -> tuple[list[str], list[dict]]:
    facts: list[tuple[int, str, str, str, str, str]] = []
    warnings: list[tuple[int, str, str, str]] = []
    for idx, sentence in enumerate(sentences):
        fact = _parse_negated_fact(sentence)
        if fact:
            subject, aux, predicate = fact
            if predicate in FUSABLE_NEGATED_PREDICATES:
                facts.append((idx, sentence, subject, aux, predicate, _entity_key(subject, sentence)))
            continue
        warning = DO_NOT_CLAIM_FACT_RE.match(sentence.strip())
        if warning:
            subject = warning.group("subject").strip()
            predicate = warning.group("predicate").lower()
            if predicate in FUSABLE_NEGATED_PREDICATES:
                warnings.append((idx, subject, warning.group("aux").lower(), predicate))

    replacements: dict[int, str] = {}
    consumed: set[int] = set()
    changes: list[dict] = []
    for fact_idx, fact_sentence, fact_subject, fact_aux, fact_predicate, fact_key in facts:
        if _is_conflict_record(fact_sentence):
            continue
        if _has_unfusible_fact_state_context(fact_sentence):
            continue
        for warning_idx, warning_subject, warning_aux, warning_predicate in warnings:
            if warning_idx in consumed or fact_idx == warning_idx:
                continue
            if fact_predicate != warning_predicate:
                continue
            if not _entity_keys_align(fact_key, fact_subject, warning_subject, sentences[warning_idx]):
                continue
            if _is_conflict_record(sentences[warning_idx]):
                continue
            pronoun = "they" if fact_aux in {"are", "were"} or _looks_plural_subject(fact_subject) else "it"
            fused_aux = "were" if pronoun == "they" else "was"
            replacement = f"{fact_sentence.rstrip('.;')}; do not claim {pronoun} {fused_aux} {warning_predicate}."
            if count_tokens(replacement) > count_tokens(fact_sentence) + count_tokens(sentences[warning_idx]):
                continue
            replacements[fact_idx] = replacement
            consumed.add(warning_idx)
            changes.append({
                "type": "fact_warning_fusion",
                "section_label": label,
                "removed_sentence": sentences[warning_idx],
                "kept_sentence": replacement,
                "reason": "factual negated state fused with matching do-not-claim warning",
                "fact_sentence": fact_sentence,
                "warning_sentence": sentences[warning_idx],
                "entity_key": fact_key,
                "predicate": "not_" + fact_predicate,
                "fact_role": "factual_state",
                "warning_role": "instruction_warning",
            })
            break

    if not replacements:
        return sentences, []
    output: list[str] = []
    for idx, sentence in enumerate(sentences):
        if idx in replacements:
            output.append(replacements[idx])
        elif idx not in consumed:
            output.append(sentence)
    return output, changes


def _parse_negated_fact(sentence: str) -> tuple[str, str, str] | None:
    stripped = sentence.strip()
    match = NEGATED_FACT_RE.match(stripped)
    if match:
        return match.group("subject").strip(), match.group("aux").lower(), match.group("predicate").lower()
    match = NO_FACT_RE.match(stripped)
    if match:
        return match.group("subject").strip(), match.group("aux").lower(), match.group("predicate").lower()
    return None


def _entity_key(subject: str, full_sentence: str) -> str:
    ids = sorted(identity_fact_keys(full_sentence))
    if ids:
        return "|".join(ids)
    return normalize_event_subject(subject)


def _entity_keys_align(fact_key: str, fact_subject: str, warning_subject: str, warning_sentence: str) -> bool:
    warning_key = _entity_key(warning_subject, warning_sentence)
    if fact_key and warning_key and fact_key == warning_key:
        return True
    fact_norm = normalize_text(fact_subject)
    warning_norm = normalize_text(warning_subject)
    return bool(fact_norm and warning_norm and (fact_norm in warning_norm or warning_norm in fact_norm))


def _looks_plural_subject(subject: str) -> bool:
    norm = normalize_text(subject)
    return norm.endswith("s") and not norm.endswith("ss")


def _is_conflict_record(sentence: str) -> bool:
    return bool(re.search(r"\b(conflict record|contradiction|conflict)\b", sentence, re.I))


def _has_unfusible_fact_state_context(sentence: str) -> bool:
    low = sentence.lower()
    return any(re.search(rf"\b{re.escape(state)}\b", low) for state in UNFUSIBLE_FACT_STATE_CONTEXT)


def can_remove_sentence(removed: str, kept: str) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    removed_role = sentence_role(removed)
    kept_role = sentence_role(kept)
    if removed_role == "factual_state" and kept_role == "instruction_warning":
        reasons.append("factual state cannot be satisfied by instruction warning")
    if kept_role == "instruction_warning" and removed_role != "instruction_warning":
        reasons.append("non-instruction fact cannot be satisfied by instruction warning")
    removed_date_frame = date_role_signature(removed)
    kept_date_frame = date_role_signature(kept)
    if removed_date_frame and kept_date_frame and removed_date_frame != kept_date_frame:
        reasons.append("different date role or predicate")
    removed_facts = canonical_fact_keys(removed)
    kept_facts = canonical_fact_keys(kept)
    if not removed_facts <= kept_facts:
        reasons.append("protected facts would be lost")
    removed_states = state_signatures(removed)
    kept_states = state_signatures(kept)
    if not removed_states <= kept_states:
        reasons.append("state signatures would be lost")
    removed_events = event_signatures(removed)
    kept_events = event_signatures(kept)
    if not removed_events <= kept_events:
        reasons.append("event signatures would be lost")
    removed_risks = risk_keys(removed)
    kept_risks = risk_keys(kept)
    if not removed_risks <= kept_risks:
        reasons.append("risk/negation markers would be lost")
    if identity_fact_keys(removed) and identity_fact_keys(removed) != identity_fact_keys(kept):
        reasons.append("different entity identities")
    return not reasons, reasons


def sentence_role(sentence: str) -> str:
    stripped = sentence.strip()
    if DO_NOT_CLAIM_FACT_RE.match(stripped) or re.match(r"^(Do not|Never)\b", stripped, re.I):
        return "instruction_warning"
    if _parse_negated_fact(stripped) or STATUS_PATTERN.match(stripped):
        return "factual_state"
    return "other"


def date_role_signature(sentence: str) -> str | None:
    if "__PROTECTED_" not in sentence:
        return None
    low = sentence.lower()
    if not re.search(r"\b(started|paused|deadline|expires?|expired|legal hold|release|filing)\b", low):
        return None
    skeleton = re.sub(r"__PROTECTED_\d+__", "DATE", sentence)
    skeleton = re.sub(r"\b(the|a|an)\b", "", skeleton, flags=re.I)
    return normalize_text(skeleton)


def validation_gate(original: str, compressed: str) -> dict:
    original_facts = canonical_fact_keys(original)
    compressed_facts = canonical_fact_keys(compressed)
    original_states = state_signatures(original)
    compressed_states = state_signatures(compressed)
    original_events = event_signatures(original)
    compressed_events = event_signatures(compressed)
    original_risks = risk_keys(original)
    compressed_risks = risk_keys(compressed)
    quality = grammar_issues(compressed)
    missing_facts = sorted(original_facts - compressed_facts)
    missing_states = sorted(original_states - compressed_states)
    missing_events = sorted(original_events - compressed_events)
    missing_risks = sorted(original_risks - compressed_risks)
    return {
        "passed": not missing_facts and not missing_states and not missing_events and not missing_risks and not quality,
        "missing_protected_fact_keys": missing_facts,
        "missing_state_signatures": missing_states,
        "missing_event_signatures": missing_events,
        "missing_risk_keys": missing_risks,
        "surface_quality_issues": quality,
        "original_fact_keys": sorted(original_facts),
        "compressed_fact_keys": sorted(compressed_facts),
        "original_state_signatures": sorted(original_states),
        "compressed_state_signatures": sorted(compressed_states),
        "original_event_signatures": sorted(original_events),
        "compressed_event_signatures": sorted(compressed_events),
    }


def analyze_text(text: str) -> dict:
    sections = parse_sections(text)
    sentences = split_sentences(text)
    counts = Counter(sentence_key(sentence) for sentence in sentences)
    exact_dupes = [{"sentence": sentence, "count": counts[sentence_key(sentence)]} for sentence in sentences if counts[sentence_key(sentence)] > 1]
    return {
        "tokens": count_tokens(text),
        "sentences": len(sentences),
        "sections": len(sections),
        "context_sections": sum(1 for section in sections if section.kind == "context"),
        "instruction_sections": sum(1 for section in sections if section.kind == "instruction"),
        "variable_sections": sum(1 for section in sections if section.kind == "variable"),
        "protected_facts_count": len(protected_facts(text)),
        "risk_markers": sorted(risk_markers(text)),
        "state_signatures": sorted(state_signatures(text)),
        "event_signatures": sorted(event_signatures(text)),
        "surface_quality_issues": grammar_issues(text),
        "exact_duplicates": exact_dupes[:10],
    }


def protected_facts(text: str) -> set[str]:
    patterns = [
        r"\bhttps?://[^\s\])}>]+",
        r"\b[\w.-]+@[\w.-]+\.\w+\b",
        r"\b[A-Z]{2,}-[A-Z0-9-]+\b",
        r"\b[A-Z]+-[0-9A-Z-]+\b",
        r"\b[a-z]{2}(?:-[a-z]+)+-\d+\b",
        r"\b[a-z][a-z0-9]+(?:_[a-z0-9]+)+\b",
        r"\$\d+(?:,\d{3})*(?:\.\d+)?",
        r"\b\d+(?:\.\d+)?%",
        r"\b\d+(?:\.\d+)?\s*percent\b",
        rf"\b{NUMBER_WORD_RE}\s*percent\b",
        r"\b\d+(?:\.\d+)?\s*(?:ms|milliseconds|seconds|minutes|hours|days)\b",
        rf"\b{NUMBER_WORD_RE}\s*(?:ms|milliseconds|seconds|minutes|hours|days)\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        r"\b\d{1,2}:\d{2}\s*(?:AM|PM)\b",
        r"\b(?:CUSIP|ISIN)\s+[A-Z0-9]+\b",
        r"\b(?:10-K|10-Q|T\+2|BB\+|Baa2|A/R|A/P|P&L|X-chromosome|X-inactivation|T-cell|B-cell|p53|IL-6|TNF-alpha)\b",
        r"__PROTECTED_\d+__",
    ]
    found: set[str] = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.I):
            value = match.group(0).strip(".,;:!?")
            if parse_number_words(value) is not None:
                continue
            found.add(value)
    return found


def canonical_fact_keys(text: str) -> set[str]:
    return {canonical_fact_key(fact) for fact in protected_facts(text)}


def canonical_fact_key(value: str) -> str:
    raw = value.strip()
    low = raw.lower().replace(",", "")
    percent = re.fullmatch(rf"(\d+(?:\.\d+)?|{NUMBER_WORD_RE})\s*(?:%|percent)", low)
    if percent:
        return "percent:" + normalize_number_text(percent.group(1))
    measure = re.fullmatch(rf"(\d+(?:\.\d+)?|{NUMBER_WORD_RE})\s*(ms|milliseconds)", low)
    if measure:
        return "measure:" + normalize_number_text(measure.group(1)) + ":ms"
    duration = re.fullmatch(rf"(\d+(?:\.\d+)?|{NUMBER_WORD_RE})\s*(seconds?|minutes?|hours?|days?)", low)
    if duration:
        unit = duration.group(2)
        unit = "second" if unit.startswith("second") else "minute" if unit.startswith("minute") else "hour" if unit.startswith("hour") else "day"
        return "duration:" + normalize_number_text(duration.group(1)) + ":" + unit
    return "raw:" + raw


def identity_fact_keys(text: str) -> set[str]:
    return {key for key in canonical_fact_keys(text) if key.startswith("raw:") and re.search(r"[@_\-/]|\b[A-Z]{2,}\b|\d", key[4:])}


def normalize_number_text(text: str) -> str:
    parsed = parse_number_words(text)
    if parsed is not None:
        return str(parsed)
    return text.replace(",", "").strip()


def parse_number_words(text: str) -> int | None:
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


def state_signatures(text: str) -> set[str]:
    low = text.lower()
    states: set[str] = set()
    for word in STATE_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", low):
            if re.search(rf"\bnot\s+{re.escape(word)}\b", low):
                states.add("not_" + word.replace("-", "_"))
            else:
                states.add(word.replace("-", "_"))
    if re.search(r"\bnot\s+deleted\b|\bno\b[^.!?;:]{0,80}\bdeleted\b", low):
        states.add("not_deleted")
    if re.search(r"\bnot\s+offline\b|\bno\b[^.!?;:]{0,80}\boffline\b", low):
        states.add("not_offline")
    for word in [
        "exposed", "modified", "approved", "migrated", "rotated",
        "patched", "disabled", "included", "ready", "mapped",
    ]:
        if re.search(rf"\bnot\s+{re.escape(word)}\b|\bno\b[^.!?;:]{{0,80}}\b{re.escape(word)}\b", low):
            states.add("not_" + word.replace("-", "_"))
    return states


EVENT_STATUS_PHRASES = {
    "is inactive": "inactive",
    "is active": "active",
    "is degraded": "degraded",
    "is read-only": "read_only",
    "status is unavailable": "unavailable",
    "is unavailable": "unavailable",
    "is not offline": "not_offline",
    "is not disabled": "not_disabled",
    "is not deleted": "not_deleted",
    "still exists": "exists",
    "is pending": "pending",
    "is not approved": "not_approved",
    "is stale": "stale",
    "is locked": "locked",
    "is not ready": "not_ready",
    "is blocked": "blocked",
    "was retained": "retained",
    "were retained": "retained",
    "was not deleted": "not_deleted",
    "were not deleted": "not_deleted",
    "was not migrated": "not_migrated",
    "were not migrated": "not_migrated",
    "was migrated": "migrated",
    "were migrated": "migrated",
    "was quarantined": "quarantined",
    "were quarantined": "quarantined",
    "was also quarantined": "quarantined",
    "were also quarantined": "quarantined",
    "was later paused": "paused",
    "were later paused": "paused",
    "was also placed on watchlist": "watchlist",
    "were also placed on watchlist": "watchlist",
    "was placed on watchlist": "watchlist",
    "were placed on watchlist": "watchlist",
    "was also flagged for legal review": "flagged_for_legal_review",
    "were also flagged for legal review": "flagged_for_legal_review",
    "was flagged for legal review": "flagged_for_legal_review",
    "were flagged for legal review": "flagged_for_legal_review",
}


def event_signatures(text: str) -> set[str]:
    signatures: set[str] = set()
    for sentence in split_sentences(text):
        raw = sentence.strip().strip(".")
        if not raw:
            continue
        low = raw.lower()
        if low.startswith("do not claim "):
            continue
        list_signatures = _list_event_signatures(raw)
        signatures.update(list_signatures)
        slot = parse_general_list_slot(raw) if not list_signatures else None
        if slot and slot.predicate != "logged":
            subject_parts = [slot.entity_type, slot.entity_id]
            if slot.owned_object:
                subject_parts.append(slot.owned_object)
            subject = normalize_event_subject(" ".join(part for part in subject_parts if part))
            if subject:
                signatures.add(f"event::{subject}::{slot.state}")

        legal_flag = re.match(r"^(?P<subject>.+?)\s+(?:is|was)\s+(?:also\s+)?flagged\s+for\s+(?P<flag>[^.]+)[.]?$", raw, re.I)
        if legal_flag and "not flagged" not in low:
            subject = normalize_event_subject(legal_flag.group("subject"))
            flag = normalize_text(legal_flag.group("flag"))
            if subject and flag:
                signatures.add(f"event::{subject}::flagged_for::{flag}")

        subject = _subject_before_event(raw, r"\b(?:started|began)\b")
        if subject:
            for value in _event_values(raw):
                if re.search(r"\b(started|began)\b", low):
                    signatures.add(f"event::{subject}::started::{value}")

        for verb in ["reached", "paused"]:
            subject = _subject_before_event(raw, rf"\b{verb}\b")
            if subject and re.search(rf"\b{verb}\b", low):
                for value in _event_values(raw):
                    signatures.add(f"event::{subject}::{verb}::{value}")

        for verb in ["decreased", "increased", "dropped", "improved"]:
            subject = _subject_before_event(raw, rf"\b{verb}\b")
            if not subject or not re.search(rf"\b{verb}\b", low):
                continue
            values = _event_values(raw)
            if len(values) >= 2:
                signatures.add(f"event::{subject}::{verb}::{values[0]}->{values[1]}")

        if re.search(r"\bfailed validation\b|\bstill fails validation\b", low):
            subject = _subject_before_event(raw, r"\b(?:failed validation|still fails validation)\b")
            if subject:
                signatures.add(f"event::{subject}::failed_validation")

        if "excluded from migration" in low:
            subject = re.sub(r"^the\s+", "", raw[:low.find("excluded from migration")].strip(), flags=re.I)
            subject = re.sub(r"\s+(is|are|was|were)$", "", subject, flags=re.I)
            subject = re.sub(r"\s+note pool$", "", subject, flags=re.I)
            if subject:
                signatures.add(f"event::note_pool:{normalize_text(subject)}::excluded")

        if not list_signatures:
            status_hits = [(low.find(phrase), phrase, event) for phrase, event in EVENT_STATUS_PHRASES.items() if phrase in low]
            status_hits = [(pos, phrase, event) for pos, phrase, event in status_hits if pos >= 0]
            if status_hits:
                first_pos = min(pos for pos, _, _ in status_hits)
                subject_text = raw[:first_pos].strip(" ,.;:")
                subject_text = re.sub(r"^(the|a|an)\s+", "", subject_text, flags=re.I)
                subject = normalize_event_subject(subject_text)
                if subject:
                    for _, _, event in status_hits:
                        signatures.add(f"event::{subject}::{event}")
    return signatures


def has_contradictory_events(text: str) -> bool:
    by_subject: dict[str, set[str]] = {}
    for signature in event_signatures(text):
        match = re.match(r"^event::(.+)::([^:]+)$", signature)
        if not match:
            continue
        by_subject.setdefault(match.group(1), set()).add(match.group(2))
    conflicts = [
        ("active", "inactive"),
        ("approved", "denied"),
        ("migrated", "not_migrated"),
        ("deleted", "not_deleted"),
        ("offline", "not_offline"),
        ("disabled", "not_disabled"),
        ("ready", "not_ready"),
        ("patched", "not_patched"),
    ]
    return any(left in events and right in events for events in by_subject.values() for left, right in conflicts)


def _list_event_signatures(sentence: str) -> set[str]:
    signatures: set[str] = set()
    signatures.update(_rendered_general_list_event_signatures(sentence))
    list_frames = {
        "Affected cohorts": ("affected", "cohort"),
        "Affected cells": ("affected", "cell"),
        "Affected airports": ("affected", "airport"),
        "Affected gates": ("affected", "gate"),
        "Operational flights": ("operational", "flight"),
        "Recovered passenger baggage": ("recovered", "passenger_baggage"),
        "Recovered baggage IDs": ("recovered", "baggage"),
        "Healthy biomarkers": ("healthy", "biomarker"),
        "Online services": ("online", "service"),
        "Active drugs": ("active", "drug"),
        "Approved findings": ("approved", "finding"),
        "Excluded note pools": ("excluded", "note_pool"),
    }
    for prefix, (event, kind) in list_frames.items():
        if not sentence.startswith(prefix + ":"):
            continue
        for item in _split_items(sentence.split(":", 1)[1].strip(" .")):
            signatures.add(f"event::{kind}:{normalize_text(item)}::{event}")

    suffix = re.match(r"^(?P<items>.+?)\s+routing tables were migrated[.]?$", sentence, re.I)
    if suffix:
        for item in _split_items(suffix.group("items")):
            signatures.add(f"event::routing_table:{normalize_text(item)}::migrated")

    simple_patterns = [
        (r"^Cohort\s+(?P<item>.+?)\s+was affected", "cohort", "affected"),
        (r"^Cell\s+(?P<item>.+?)\s+was affected", "cell", "affected"),
        (r"^Airport\s+(?P<item>.+?)\s+was affected", "airport", "affected"),
        (r"^Gate\s+(?P<item>.+?)\s+was affected", "gate", "affected"),
        (r"^Flight\s+(?P<item>.+?)\s+is operational", "flight", "operational"),
        (r"^Passenger\s+(?P<item>.+?)\s+baggage was recovered", "passenger_baggage", "recovered"),
        (r"^Baggage\s+(?P<item>.+?)\s+was recovered", "baggage", "recovered"),
        (r"^Biomarker\s+(?P<item>.+?)\s+(?:status is healthy|is healthy)", "biomarker", "healthy"),
        (r"^Service\s+(?P<item>.+?)\s+is online", "service", "online"),
        (r"^Drug\s+(?P<item>.+?)\s+is active", "drug", "active"),
        (r"^Approved\s+(?P<item>.+?)$", "finding", "approved"),
        (r"^(?P<item>Zone-[A-Z0-9-]+)\s+routing tables were migrated", "routing_table", "migrated"),
        (r"^(?P<item>Class-[A-Z0-9-]+)\s+seating records were migrated", "seating_record", "migrated"),
        (r"^(?P<item>Class-[A-Z0-9-]+)\s+seating records were not migrated", "seating_record", "not_migrated"),
    ]
    for pattern, kind, event in simple_patterns:
        match = re.match(pattern, sentence.strip(), re.I)
        if match:
            signatures.add(f"event::{kind}:{normalize_text(match.group('item'))}::{event}")
    return signatures


def _rendered_general_list_event_signatures(sentence: str) -> set[str]:
    if "," not in sentence and not re.search(r"\s+and\s+", sentence, re.I):
        return set()
    signatures: set[str] = set()
    for slot in parse_rendered_general_list_sentence(sentence):
        if slot.predicate == "logged":
            continue
        subject_parts = [slot.entity_type, slot.entity_id]
        if slot.owned_object:
            subject_parts.append(slot.owned_object)
        subject = normalize_event_subject(" ".join(part for part in subject_parts if part))
        if subject:
            signatures.add(f"event::{subject}::{slot.state}")
    return signatures


def _subject_before_event(sentence: str, event_pattern: str) -> str | None:
    match = re.search(event_pattern, sentence, flags=re.I)
    if not match:
        return None
    subject = sentence[:match.start()].strip(" ,.;:")
    subject = re.sub(r"^(the|a|an)\s+", "", subject, flags=re.I)
    return normalize_event_subject(subject) if subject else None


def normalize_event_subject(subject: str) -> str:
    normalized = normalize_text(re.sub(r"^(the|a|an)\s+", "", subject.strip(), flags=re.I))
    for kind in ["compliance control", "routing worker", "control", "policy", "region", "service", "scanner", "deployment", "node"]:
        if normalized.startswith(kind + " "):
            return f"{kind}:{normalized[len(kind) + 1:]}"
        if normalized.endswith(" " + kind):
            return f"{kind}:{normalized[:-(len(kind) + 1)]}"
    return normalized


def _event_values(sentence: str) -> list[str]:
    values: list[str] = []
    value_pattern = (
        rf"__PROTECTED_\d+__|"
        rf"\b\d{{4}}-\d{{2}}-\d{{2}}\b|"
        rf"(?:\b(?:\d+(?:\.\d+)?|{NUMBER_WORD_RE})\s*%|\b(?:\d+(?:\.\d+)?|{NUMBER_WORD_RE})\s*percent\b)|"
        rf"\b(?:\d+(?:\.\d+)?|{NUMBER_WORD_RE})\s*(?:ms|milliseconds|seconds|minutes|hours|days)\b"
    )
    for match in re.finditer(value_pattern, sentence, flags=re.I):
        key = canonical_fact_key(match.group(0))
        if key not in values:
            values.append(key)
    return sorted(set(values), key=values.index)


def _split_items(text: str) -> list[str]:
    items: list[str] = []
    for item in re.split(r",\s*|\s+and\s+|\s+or\s+", text):
        cleaned = re.sub(r"^(and|or)\s+", "", item.strip(" ."), flags=re.I)
        if cleaned:
            items.append(cleaned)
    return items


def risk_markers(text: str) -> set[str]:
    low = text.lower()
    found: set[str] = set()
    for marker in RISK_MARKERS:
        if " " in marker or "-" in marker:
            if re.search(rf"(?<![a-z0-9]){re.escape(marker)}(?![a-z0-9])", low):
                found.add(marker)
        elif re.search(rf"\b{re.escape(marker)}\b", low):
            found.add(marker)
    return found


def risk_keys(text: str) -> set[str]:
    return {"risk:" + marker for marker in risk_markers(text)}


def grammar_issues(text: str) -> list[str]:
    issues: list[str] = []
    if re.search(r"\s+[,.!?;:]", text):
        issues.append("broken_punctuation")
    if re.search(r"\bnot\s+[A-Za-z0-9_-]+\s+or\s+[A-Za-z0-9_-]+\b", text, re.I):
        issues.append("ambiguous_not_or")
    if "PREFACE: PREFACE" in text:
        issues.append("duplicated_preface_label")
    return sorted(set(issues))


def lexical_similarity(a: str, b: str) -> float:
    a_tokens = set(re.findall(r"[A-Za-z0-9$%@./:+-]+", a.lower()))
    b_tokens = set(re.findall(r"[A-Za-z0-9$%@./:+-]+", b.lower()))
    jaccard = len(a_tokens & b_tokens) / max(1, len(a_tokens | b_tokens))
    sequence = SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()
    fact_bonus = 0.1 if canonical_fact_keys(a) and canonical_fact_keys(a) == canonical_fact_keys(b) else 0.0
    return min(1.0, 0.6 * jaccard + 0.4 * sequence + fact_bonus)


def normalize_text(text: str) -> str:
    lowered = text.lower()
    lowered = _canonicalize_number_word_units(lowered)
    return re.sub(r"[^a-z0-9$%@./:+-]+", " ", lowered).strip()


def _canonicalize_number_word_units(text: str) -> str:
    unit_pattern = rf"\b({NUMBER_WORD_RE})\s*(percent|ms|milliseconds|seconds|minutes|hours|days)\b"

    def replace_unit(match: re.Match) -> str:
        number = parse_number_words(match.group(1))
        if number is None:
            return match.group(0)
        unit = match.group(2)
        if unit == "percent":
            return f"{number}%"
        unit = "ms" if unit == "milliseconds" else unit
        return f"{number} {unit}"

    text = re.sub(unit_pattern, replace_unit, text, flags=re.I)
    text = re.sub(r"(?<=[a-z])-(?=[a-z])", " ", text)
    text = re.sub(r"\b(were|was)\s+migrated\s+successfully\b", r"\1 migrated", text, flags=re.I)
    text = re.sub(r"\b(\d+(?:\.\d+)?)\s+percent\b", r"\1%", text, flags=re.I)
    text = re.sub(rf"\b({NUMBER_WORD_RE})\b", lambda match: str(parse_number_words(match.group(1))) if parse_number_words(match.group(1)) is not None else match.group(0), text, flags=re.I)
    return text


def sentence_key(text: str) -> str:
    return normalize_text(text)


def choose_keeper_index(sentences: list[str], i: int, j: int) -> tuple[int, int]:
    # Prefer the shorter sentence if it preserves the same facts; otherwise keep the earlier one.
    if canonical_fact_keys(sentences[i]) == canonical_fact_keys(sentences[j]):
        return (i, j) if count_tokens(sentences[i]) <= count_tokens(sentences[j]) else (j, i)
    return i, j


def join_list(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def join_or_list(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} or {items[1]}"
    return ", ".join(items[:-1]) + f", or {items[-1]}"


def join_predicates(predicates: list[str]) -> str:
    if len(predicates) == 1:
        return predicates[0]
    if len(predicates) == 2:
        return f"{predicates[0]} and {predicates[1]}"
    return ", ".join(predicates[:-1]) + f", and {predicates[-1]}"


def choose_final_action(before: dict, after: dict, accepted: bool) -> str:
    if not accepted:
        return "send_original_to_next_layer"
    if before["tokens"] == after["tokens"]:
        return "no_change"
    return "use_cheap_compressed_prompt"


def run_self_checks() -> list[dict]:
    checks = [
        ("Database write latency increased from 12 ms to 28 ms.", "Database write latency increased from twelve ms to twenty-eight ms.", True),
        ("The rollout paused at 83%.", "The rollout paused at eighty-three percent.", True),
        ("Region us-east-1 was affected.", "Region us-west-2 was affected.", False),
        ("Service CACHE-WARMER is degraded.", "Service CACHE-WARMER is not offline.", False),
        ("Ledger snapshot LS-9004 was retained.", "Ledger snapshot LS-9004 was not deleted.", False),
    ]
    failures: list[dict] = []
    for a, b, expected in checks:
        ok, reasons = can_remove_sentence(a, b)
        score = lexical_similarity(a, b)
        if canonical_fact_keys(a) and canonical_fact_keys(a) == canonical_fact_keys(b):
            score = max(score, 0.95)
        actual = ok and score > 0.75
        if actual != expected:
            failures.append({"a": a, "b": b, "expected": expected, "actual": actual, "reasons": reasons})
    return failures
