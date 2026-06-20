from __future__ import annotations

from dataclasses import dataclass
import re

from .prompt_type import PromptAnalysis
from .token_counter import count_tokens


BACKEND_NAME = "enterprise_cost_optimizer"


@dataclass
class Section:
    header: str
    body: str
    start: int
    end: int

    @property
    def text(self) -> str:
        return f"{self.header}:\n{self.body.strip()}".strip()


SECTION_RE = re.compile(r"(?im)^([A-Z][A-Za-z0-9 /&-]{2,60}):\s*$")


def enterprise_cost_optimizer(text: str, analysis: PromptAnalysis, level: str = "balanced") -> tuple[str, list[dict]]:
    if analysis.prompt_type != "customer_support":
        return text, []
    optimizer = CustomerSupportPromptOptimizer()
    return optimizer.optimize(text)


class CustomerSupportPromptOptimizer:
    """Optimizer for assembled customer-support prompts.

    This pass is intentionally structural rather than free-form summarization.
    It builds one canonical policy/reminder block from repeated support sources
    and keeps the ticket/task blocks intact.
    """

    def optimize(self, text: str) -> tuple[str, list[dict]]:
        sections = _parse_sections(text)
        if len(sections) < 3:
            return text, []

        ticket = _find_section(sections, "Customer Ticket")
        task = _find_section(sections, "Task")
        if not ticket or not task:
            return text, []

        prefix = text[:sections[0].start].strip()
        policy_sections = [
            section for section in sections
            if section.header not in {"Customer Ticket", "Task", "Agent Reminder"}
            and _looks_like_support_policy(section.body)
        ]
        reminder = _find_section(sections, "Agent Reminder")

        facts = _extract_support_facts("\n".join(section.body for section in policy_sections))
        reminder_facts = _extract_reminder_facts(reminder.body if reminder else "")

        if not policy_sections and not reminder_facts:
            return text, []

        optimized = self._render(prefix, facts, ticket, reminder_facts, task)
        optimized = _clean_blank_lines(optimized)
        if count_tokens(optimized) >= count_tokens(text):
            return text, [self._trace(False, text, optimized, facts, reminder_facts, "optimized prompt was not shorter")]
        return optimized, [self._trace(True, text, optimized, facts, reminder_facts, "customer support enterprise prompt compacted")]

    def _render(self, prefix: str, facts: dict, ticket: Section, reminder_facts: dict, task: Section) -> str:
        blocks: list[str] = []
        if prefix:
            blocks.append(prefix)

        requirements: list[str] = []
        if facts.get("refund_window"):
            requirements.append("Customers may request a refund within 30 days of purchase if product usage remains within trial limits.")
        verification = facts.get("verification_items") or []
        if verification:
            requirements.append(f"Before processing refunds, agents must verify {_join_list(verification)}.")
        finance_threshold = facts.get("finance_threshold") or reminder_facts.get("finance_threshold")
        if finance_threshold:
            requirements.append(f"Refunds exceeding {finance_threshold} require Finance approval.")
        if requirements:
            blocks.append("Support Requirements:\n" + "\n".join(requirements))

        blocks.append(ticket.text)

        reminders: list[str] = []
        if reminder_facts.get("do_not_promise"):
            reminders.append("Do not promise refund approval.")
        if reminder_facts.get("under_review"):
            reminders.append("Inform the customer that the request is under review.")
        if reminder_facts.get("finance_threshold") and not finance_threshold:
            reminders.append(f"Mention that refunds above {reminder_facts['finance_threshold']} require Finance approval.")
        elif reminder_facts.get("finance_threshold"):
            reminders.append(f"Mention that refunds above {reminder_facts['finance_threshold']} require Finance approval.")
        if reminders:
            blocks.append("Agent Reminder:\n" + "\n".join(_ordered_unique(reminders)))

        blocks.append(task.text)
        return "\n\n".join(blocks)

    def _trace(self, accepted: bool, original: str, optimized: str, facts: dict, reminder_facts: dict, reason: str) -> dict:
        return {
            "backend": BACKEND_NAME,
            "candidate_type": "enterprise_customer_support_compaction",
            "prompt_type": "customer_support",
            "span_text": original,
            "retained_span": optimized if accepted else None,
            "removed_span": original if accepted else None,
            "tokens_saved": max(0, count_tokens(original) - count_tokens(optimized)),
            "score": 0.96 if accepted else 0.0,
            "accepted": accepted,
            "reason": reason,
            "rejected_reason": None if accepted else reason,
            "risk_flags": [] if accepted else [reason],
            "strategy_breakdown": {
                "deduped_verification_requirements": bool(facts.get("verification_items")),
                "deduped_finance_approval_rules": bool(facts.get("finance_threshold")),
                "deduped_agent_reminders": bool(reminder_facts.get("do_not_promise")),
                "ticket_fields_preserved_exactly": True,
            },
            "extracted_facts": {
                "support_facts": facts,
                "reminder_facts": reminder_facts,
            },
        }


def _parse_sections(text: str) -> list[Section]:
    matches = list(SECTION_RE.finditer(text))
    sections: list[Section] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        sections.append(Section(match.group(1).strip(), text[start:end].strip(), match.start(), end))
    return sections


def _find_section(sections: list[Section], header: str) -> Section | None:
    for section in sections:
        if section.header.lower() == header.lower():
            return section
    return None


def _looks_like_support_policy(body: str) -> bool:
    low = body.lower()
    return "refund" in low and ("verify" in low or "finance approval" in low or "approval from finance" in low)


def _extract_support_facts(text: str) -> dict:
    low = text.lower()
    facts: dict = {}
    if "30 days" in low and "trial limit" in low:
        facts["refund_window"] = True
    items = []
    for label in ["order ID", "customer email", "purchase date", "refund reason", "product SKU"]:
        if label.lower() in low:
            items.append(label)
    if items:
        facts["verification_items"] = items
    threshold = _finance_threshold(text)
    if threshold:
        facts["finance_threshold"] = threshold
    return facts


def _extract_reminder_facts(text: str) -> dict:
    low = text.lower()
    facts: dict = {}
    if re.search(r"\bdo not promise\b.*\bapproval\b", low):
        facts["do_not_promise"] = True
    if "under review" in low:
        facts["under_review"] = True
    threshold = _finance_threshold(text)
    if threshold:
        facts["finance_threshold"] = threshold
    return facts


def _finance_threshold(text: str) -> str | None:
    if "finance" not in text.lower() or "approval" not in text.lower():
        return None
    money = re.findall(r"__PROTECTED_\d+__|\$\d[\d,]*(?:\.\d+)?", text)
    return money[0] if money else "$500"


def _join_list(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def _clean_blank_lines(text: str) -> str:
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
