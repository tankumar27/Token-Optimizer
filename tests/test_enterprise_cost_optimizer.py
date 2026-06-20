from app.models import ChatMessage
from optimizer.pipeline import optimize_messages
from optimizer.prompt_type import detect_prompt_type


ENTERPRISE_SUPPORT_PROMPT = """You are a customer support assistant. Use only the information provided below.

Company Refund Policy:
Customers may request a refund within 30 days of purchase if product usage remains within trial limits.
Refund requests must include order ID, customer email, purchase date, refund reason, and product SKU.
Refunds exceeding $500 require Finance approval.

Internal Support Handbook:
Before processing a refund request, agents must verify order ID, customer email, purchase date, refund reason, and product SKU.
For refund amounts greater than $500, Finance approval is required before issuing a refund.

Support Escalation Guide:
Agents should verify order ID, customer email, purchase date, refund reason, and product SKU before processing refunds.
Refund requests above $500 require approval from Finance.

Knowledge Base Article:
Refunds over $500 require Finance approval.
Support representatives should verify order ID, customer email, purchase date, refund reason, and product SKU before handling a refund.

Customer Ticket:
Ticket ID: SUP-99124
Customer Email: maya.chen@example.com
Order ID: ORD-900184
Product SKU: PRO-ANALYTICS-TEAM
Purchase Date: 2026-05-18
Refund Amount: $749
Reason: The annual plan was purchased by mistake instead of the monthly plan.
Usage: Trial dashboard opened twice. No exports generated.

Agent Reminder:
Do not promise refund approval.
Do not promise refund approval.
Do not promise refund approval.
Inform the customer that the request is under review.
Mention that refunds above $500 require Finance approval.

Task:
Write a customer-facing response."""


def test_prompt_type_detects_customer_support_workload():
    analysis = detect_prompt_type(ENTERPRISE_SUPPORT_PROMPT)
    assert analysis.prompt_type == "customer_support"
    assert analysis.cacheability == "high"
    assert "customer_identifiers" in analysis.sensitivity_flags


def test_enterprise_support_prompt_compacts_repeated_requirements_and_preserves_ticket():
    result = optimize_messages(
        [ChatMessage(role="user", content=ENTERPRISE_SUPPORT_PROMPT)],
        "balanced",
        "dry-run",
        "dry-run",
    )
    output = result["optimized_messages"][0].content
    assert result["savings_percent"] >= 25
    assert "enterprise_cost_optimizer" in result["backend_used"]
    assert output.count("Do not promise refund approval.") == 1
    assert output.count("Finance approval") >= 1
    assert "Before processing refunds, agents must verify order ID, customer email, purchase date, refund reason, and product SKU." in output
    for fact in [
        "SUP-99124",
        "maya.chen@example.com",
        "ORD-900184",
        "PRO-ANALYTICS-TEAM",
        "2026-05-18",
        "$749",
        "The annual plan was purchased by mistake instead of the monthly plan.",
        "Trial dashboard opened twice. No exports generated.",
        "Write a customer-facing response.",
    ]:
        assert fact in output
    assert result["quality_gate_status"]["accepted"]


def test_enterprise_strategy_trace_explains_selected_work():
    result = optimize_messages(
        [ChatMessage(role="user", content=ENTERPRISE_SUPPORT_PROMPT)],
        "balanced",
        "dry-run",
        "dry-run",
    )
    strategy = result["traces"]["strategy_decision"][0]
    trace = next(item for item in result["removed_or_changed_text"] if item.get("backend") == "enterprise_cost_optimizer")
    assert strategy["prompt_type"] == "customer_support"
    assert "customer_support_policy_dedupe" in strategy["chosen_strategies"]
    assert trace["strategy_breakdown"]["deduped_verification_requirements"]
    assert trace["strategy_breakdown"]["deduped_finance_approval_rules"]
    assert trace["strategy_breakdown"]["deduped_agent_reminders"]
