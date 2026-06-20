from app.models import ChatMessage
from optimizer.pipeline import optimize_messages


def _result(text: str):
    return optimize_messages([ChatMessage(role="user", content=text)], "balanced", "dry-run", "dry-run")


def _out(text: str) -> str:
    return _result(text)["optimized_messages"][0].content


def test_reliability_constraints_aggregate_unique_concepts():
    text = (
        "System reliability must be preserved. "
        "Service availability must remain high. "
        "Performance degradation is not acceptable. "
        "Customer experience should not suffer."
    )
    output = _out(text)
    assert output == "The system must preserve reliability, availability, performance, and customer experience."


def test_cost_reliability_constraints_keep_cost_frame():
    text = (
        "Cost savings should not reduce reliability. "
        "Performance degradation is not acceptable. "
        "Customer experience must not suffer."
    )
    output = _out(text)
    assert output == "Cost reductions must not compromise reliability, performance, and customer experience."


def test_business_outcomes_aggregate_margin_and_financial_performance():
    text = (
        "The company reduced cloud spending. "
        "Lower infrastructure expenses improved operating margins. "
        "Reduced spending contributed positively to financial performance."
    )
    output = _out(text)
    assert output == "The company lowered cloud spending, improving operating margins and financial performance."


def test_support_requirements_aggregate_verification_slots():
    text = (
        "Agents must verify order ID. "
        "Agents must verify customer email. "
        "Agents must verify purchase date. "
        "Agents must verify refund reason."
    )
    output = _out(text)
    assert output == "Agents must verify order ID, customer email, purchase date, and refund reason."


def test_product_benefits_aggregate_actions():
    text = (
        "The platform saves time. "
        "The platform reduces manual work. "
        "The platform improves reporting accuracy."
    )
    output = _out(text)
    assert output == "The platform saves time, reduces manual work, and improves reporting accuracy."


def test_concept_aggregation_does_not_merge_contradiction():
    text = "Costs decreased. Costs increased."
    assert _out(text) == text


def test_concept_aggregation_preserves_numbers_and_contrast():
    text = "Cloud costs decreased. However, customer acquisition costs increased by 11%."
    output = _out(text)
    assert "11%" in output
    assert "However" in output
    assert "customer acquisition costs increased" in output


def test_long_enterprise_report_aggregates_cost_and_constraints():
    text = """You are an enterprise operations assistant.

Executive Summary:
The company reduced cloud infrastructure costs through compute optimization initiatives. By improving compute efficiency, the organization lowered cloud spending. Reduced compute waste contributed significantly to lower infrastructure expenses. One of the primary business outcomes this quarter was a reduction in cloud costs through optimization efforts.

Engineering Review:
Engineering teams reduced unnecessary compute workloads across production systems. By eliminating redundant processing, the company improved efficiency and reduced infrastructure expenses. Optimization of compute-intensive services lowered overall cloud costs. Reducing wasteful computation was a major contributor to cost savings.

Finance Review:
Cloud expenditures decreased because infrastructure resources were used more efficiently. Compute optimization initiatives lowered operational spending. Reduced infrastructure waste helped decrease cloud-related expenses. Improvements in compute efficiency contributed to lower cloud costs.

Operations Review:
The organization improved resource utilization across its cloud environment. Better utilization of compute resources reduced infrastructure spending. Increased efficiency helped lower cloud expenditures. Optimization efforts reduced operational costs associated with cloud workloads.

Business Impact:
The company successfully reduced cloud costs through efficiency improvements. Lower infrastructure expenses improved operating margins. Reduced spending on cloud resources contributed positively to financial performance.

Risk Assessment:
Aggressive cost reduction efforts must not reduce system reliability. Critical services must maintain performance and availability. Cost savings should not come at the expense of customer experience.

Reliability Requirements:
System reliability must be preserved. Service availability must remain high. Performance degradation is not acceptable even when pursuing cost reductions.

Additional Observation:
Customer acquisition costs increased by 11% during the same quarter.

Task:
Summarize the most important operational and business outcomes from this report."""
    result = _result(text)
    output = result["optimized_messages"][0].content
    assert result["optimized_tokens"] < result["original_tokens"]
    assert "compute optimization" in output or "compute efficiency" in output
    assert "operating margins" in output
    assert "financial performance" in output
    assert "reliability" in output
    assert "availability" in output
    assert "performance" in output
    assert "customer experience" in output
    assert "11%" in output
    assert any(backend in result["backend_used"] for backend in ["information_representation_backend", "concept_aggregation_backend"])
