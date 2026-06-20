from app.self_discovery import run_self_discovery
from app.models import ChatMessage
from optimizer.information_graph import InformationGraph
from optimizer.pipeline import optimize_messages
from optimizer.proposition_extractor import PropositionExtractor


def test_proposition_extractor_builds_information_units():
    text = "System reliability must be preserved. Service availability must remain high."
    units = PropositionExtractor().extract(text)
    assert len(units) == 2
    assert units[0].subject == "System reliability"
    assert units[0].theme == "reliability_constraint"
    assert "reliability" in units[0].concepts
    assert units[0].modality == "must"


def test_information_graph_groups_by_concept_theme():
    units = PropositionExtractor().extract(
        "Agents must verify order ID. Agents must verify customer email. Agents must verify purchase date."
    )
    clusters = InformationGraph(units).compressible_clusters()
    assert len(clusters) == 1
    assert clusters[0].theme == "support_verification"
    assert clusters[0].concepts == ["order ID", "customer email", "purchase date"]


def test_information_representation_backend_emits_minimum_trace():
    result = optimize_messages(
        [ChatMessage(role="user", content="The platform saves time. The platform reduces manual work. The platform improves reporting accuracy.")],
        "balanced",
        "dry-run",
        "dry-run",
    )
    assert "information_representation_backend" in result["backend_used"]
    trace = next(item for item in result["removed_or_changed_text"] if item.get("backend") == "information_representation_backend")
    assert trace["accepted"] is True
    assert trace["information_recall"] == 1.0
    assert trace["generated_minimum_representation"] == "The platform saves time, reduces manual work, and improves reporting accuracy."


def test_self_discovery_has_low_failure_rate():
    report = run_self_discovery(seed=3, cases=40)
    assert report["failure_rate"] <= 0.05
    assert report["average_savings_percent"] > 10


def test_information_representation_preserves_distinct_security_secrets():
    result = optimize_messages(
        [ChatMessage(role="user", content="Do not expose API keys. Never reveal access tokens. Secrets must not be included in customer-facing responses. Credentials should remain confidential.")],
        "balanced",
        "dry-run",
        "dry-run",
    )
    output = result["optimized_messages"][0].content
    for concept in ["API keys", "access tokens", "secrets", "credentials"]:
        assert concept.lower() in output.lower()
    assert "information_representation_backend" in result["backend_used"]


def test_information_representation_preserves_operational_metrics_benefit():
    result = optimize_messages(
        [ChatMessage(role="user", content="The platform saves time. The platform reduces manual work. The platform improves reporting accuracy. The platform helps teams monitor operational metrics.")],
        "balanced",
        "dry-run",
        "dry-run",
    )
    output = result["optimized_messages"][0].content
    assert "operational metrics" in output
    assert "reporting accuracy" in output
    assert result["optimized_tokens"] < result["original_tokens"]
