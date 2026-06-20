from app.config import get_settings
from app.models import ChatMessage
import optimizer.llm_lingua_layer as lingua_layer
from optimizer.llm_lingua_layer import llm_lingua_backend, validate_llm_lingua_candidate
from optimizer.pipeline import optimize_messages


def test_llm_lingua_layer_compresses_safe_low_information_filler():
    result = optimize_messages(
        [ChatMessage(role="user", content="Please please kindly kindly answer for ORD-900184.")],
        "balanced",
        "dry-run",
        "dry-run",
    )
    output = result["optimized_messages"][0].content
    assert output == "Please answer for ORD-900184."
    assert "llm_lingua_backend" in result["backend_used"]
    assert result["optimized_tokens"] < result["original_tokens"]
    trace = next(item for item in result["removed_or_changed_text"] if item.get("backend") == "llm_lingua_backend" and item.get("accepted"))
    assert trace["generator"] == "deterministic_token_cleanup"
    assert trace["facts_preserved"] is True


def test_llm_lingua_validator_rejects_lost_event_signature():
    original = "Remediation reached 98%. Remediation paused at 98%. Control __PROTECTED_1__ is blocked."
    candidate = "Remediation reached 98%. Control __PROTECTED_1__ is blocked."
    accepted, reason, validation = validate_llm_lingua_candidate(original, candidate, "balanced")
    assert accepted is False
    assert "cheap-layer invariant gate failed" in reason
    assert any("paused" in item for item in validation["missing_event_signatures"])


def test_llm_lingua_validator_rejects_missing_protected_placeholder():
    original = "Ticket __PROTECTED_1__ is blocked. Please please answer."
    candidate = "Ticket is blocked. Please answer."
    accepted, reason, validation = validate_llm_lingua_candidate(original, candidate, "balanced")
    assert accepted is False
    assert "protected placeholders missing" in reason
    assert validation["missing_placeholders"] == ["__PROTECTED_1__"]


def test_llm_lingua_gemini_backend_falls_back_without_key(monkeypatch):
    monkeypatch.setenv("ENABLE_LLM_LINGUA", "1")
    monkeypatch.setenv("LLM_LINGUA_BACKEND", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    get_settings.cache_clear()
    try:
        output, traces = llm_lingua_backend("Please please answer for __PROTECTED_1__.", "balanced", "gemini", "dry-run")
    finally:
        get_settings.cache_clear()
    assert output == "Please answer for __PROTECTED_1__."
    assert any(trace.get("generator") == "gemini_llm_lingua" and not trace.get("accepted") for trace in traces)
    assert any(trace.get("generator") == "deterministic_token_cleanup" and trace.get("accepted") for trace in traces)


def test_true_llmlingua2_backend_candidate_is_validated(monkeypatch):
    class FakeCompressor:
        def compress_prompt(self, context, **kwargs):
            assert kwargs["force_reserve_digit"] is True
            assert "__PROTECTED_1__" in kwargs["force_tokens"]
            return {
                "compressed_prompt": "Answer for __PROTECTED_1__.",
                "origin_tokens": 6,
                "compressed_tokens": 4,
                "ratio": "1.5x",
                "rate": "66.7%",
            }

    monkeypatch.setenv("ENABLE_LLM_LINGUA", "1")
    monkeypatch.setenv("LLM_LINGUA_BACKEND", "llmlingua2")
    get_settings.cache_clear()
    monkeypatch.setattr(lingua_layer, "_get_llmlingua2_compressor", lambda model, device: FakeCompressor())
    try:
        output, traces = llm_lingua_backend("Please please answer for __PROTECTED_1__.", "balanced", "gemini", "dry-run")
    finally:
        get_settings.cache_clear()
    assert output == "Answer for __PROTECTED_1__."
    accepted = [trace for trace in traces if trace.get("accepted")]
    assert accepted
    assert accepted[-1]["generator"] == "llmlingua2"
    assert accepted[-1]["facts_preserved"] is True
