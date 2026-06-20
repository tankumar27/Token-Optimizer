from fastapi.testclient import TestClient

from app.main import app
from app.models import ChatMessage
from demo import company_pilot_sim
from optimizer.pipeline import optimize_messages
from optimizer.routing import route_request
from storage.semantic_cache import protected_facts_match, semantic_cache_lookup


client = TestClient(app)


def test_exact_same_prompt_hits_cache():
    payload = {
        "provider": "gemini",
        "mode": "dry-run",
        "messages": [{"role": "user", "content": "hello hello can you help"}],
        "temperature": 0,
    }
    first = client.post("/v1/chat/completions", json=payload)
    second = client.post("/v1/chat/completions", json=payload)
    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["middleware"]["traces"]["cache_hit"] is True


def test_semantic_cache_rejects_time_sensitive_prompt():
    result = semantic_cache_lookup("what is the latest price today", "gemini", "gemini-1.5-flash")
    assert result["hit"] is None
    assert result["trace"]["rejection_reason"] == "time_sensitive_prompt"


def test_different_ids_and_money_do_not_match_cache_safety():
    assert protected_facts_match("Order ORD-1 costs $10.", "Order ORD-2 costs $10.") is False
    assert protected_facts_match("Order ORD-1 costs $10.", "Order ORD-1 costs $20.") is False


def test_routing_signals():
    assert route_request([ChatMessage(role="user", content="hi there")], "gemini")["route_decision"] == "cheaper_model"
    assert route_request([ChatMessage(role="user", content="fix ```python\nprint(1)\n```")], "gemini")["route_decision"] == "stronger_model"
    assert route_request([ChatMessage(role="user", content="legal contract Clause 4.2")], "openai")["route_decision"] == "stronger_model"
    assert route_request([ChatMessage(role="user", content="hi")], "gemini", "custom-model")["route_decision"] == "user_override"


def test_grammar_does_not_leave_orphan_is():
    result = optimize_messages([ChatMessage(role="user", content="Acme Cloud API is our provider. Acme Cloud API is reliable.")], "balanced", "gemini", "dry-run")
    output = result["optimized_messages"][0].content
    assert ". is reliable" not in output.lower()
    assert result["traces"]["grammar_validity"] is True


def test_company_pilot_score_bounded():
    summary = company_pilot_sim("safe", "dry-run")["summary"]
    assert 0 <= summary["production_readiness_score"] <= 85
