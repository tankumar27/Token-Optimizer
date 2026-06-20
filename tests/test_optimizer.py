from app.models import ChatMessage
from optimizer.pipeline import optimize_messages


def optimize(text: str, level: str = "safe"):
    return optimize_messages([ChatMessage(role="user", content=text)], level, "gemini", "dry-run")


def test_repeated_sentence_compressed():
    result = optimize("Do not promise approval. Do not promise approval.")
    assert result["optimized_tokens"] < result["original_tokens"]


def test_repeated_filler_compressed():
    result = optimize("please please please kindly kindly help help me")
    assert result["optimized_tokens"] < result["original_tokens"]


def test_domain_repetitions_preserved():
    for text in ["x-chromosome x-chromosome", "New York New York", "had had"]:
        result = optimize(text)
        assert result["optimized_messages"][0].content == text


def test_optimized_never_longer():
    result = optimize("Hi")
    assert result["optimized_tokens"] <= result["original_tokens"]


def test_phrase_level_redundancy_balanced_example():
    text = "hello hello hello my name is tanishq. hello bro my name is tanishq what can i do, hello"
    result = optimize(text, "balanced")
    assert result["optimized_messages"][0].content == "hello my name is tanishq. bro what can i do?"
    assert result["optimized_tokens"] < result["original_tokens"]


def test_repeated_greetings_compress_to_one_retained_copy():
    result = optimize("hey hey hey can you help")
    assert result["optimized_messages"][0].content == "hey can you help"


def test_repeated_phrase_my_name_is_x_preserves_entity_once():
    result = optimize("my name is tanishq and I need help. my name is tanishq and I need help.", "balanced")
    output = result["optimized_messages"][0].content
    assert "tanishq" in output
    assert output.count("my name is tanishq") == 1


def test_trailing_repeated_filler_removed():
    result = optimize("please review this please", "balanced")
    assert result["optimized_messages"][0].content == "please review this"


def test_repeated_paragraph_compresses():
    paragraph = "This onboarding policy applies to all contractors and vendors.\n\n"
    result = optimize(paragraph + paragraph, "balanced")
    assert result["optimized_tokens"] < result["original_tokens"]


def test_named_entity_preserved_when_redundancy_removed():
    result = optimize("Alice can help with onboarding. Alice can help with onboarding.", "balanced")
    assert "Alice" in result["optimized_messages"][0].content


def test_math_code_json_unchanged():
    text = "Keep this: ```python\nprint('ORD-900184')\n``` and {\"id\":\"SUP-44891\"}. y = x^2 - x - 23."
    result = optimize(text, "balanced")
    assert result["optimized_messages"][0].content == text


def test_numbers_dates_money_ids_preserved():
    text = "ORD-900184 is due on 2026-06-01 for $10,000. please please help"
    result = optimize(text, "balanced")
    output = result["optimized_messages"][0].content
    for value in ["ORD-900184", "2026-06-01", "$10,000"]:
        assert value in output


def test_trace_explains_accepted_and_rejected_candidates():
    result = optimize("x-chromosome x-chromosome please please help", "balanced")
    candidates = [item for item in result["removed_or_changed_text"] if item.get("candidate_type")]
    assert any(item["accepted"] for item in candidates)
    assert any(not item["accepted"] and item["risk_flags"] for item in candidates)
    for item in candidates:
        assert "span_text" in item
        assert "score" in item
        assert "tokens_saved" in item
        assert "reason" in item


def test_product_name_partial_ngram_does_not_break_sentence():
    result = optimize("Acme Cloud API is our provider. Acme Cloud API is reliable. please please help", "balanced")
    output = result["optimized_messages"][0].content
    assert "Acme Cloud API is reliable" in output
    assert "Acme reliable" not in output


def test_semantic_comparative_claims_compact_meaningfully():
    text = "ACT is very easy compared to SAT. ACT is easy because it more easier than SAT. ACT is very less hard because it might be easier than ACT."
    result = optimize(text, "balanced")
    assert result["optimized_messages"][0].content == "ACT is easier than SAT."
    assert result["optimized_tokens"] < result["original_tokens"]
    assert any(item.get("backend") == "semantic_claim_compactor" and item.get("accepted") for item in result["removed_or_changed_text"])


def test_semantic_comparative_equivalent_reverse_claims_compact():
    text = "ACT is very easy than SAT. ACT is easy than SAT. SAT is harder than ACT. ACT is astonishingly easier than SAT. ACT is very very easier than SAT for some reason. ACT is very very easy."
    result = optimize(text, "balanced")
    assert result["optimized_messages"][0].content == "ACT is easier than SAT."
    assert result["savings_percent"] > 70


def test_semantic_comparative_compactor_generalizes():
    examples = [
        (
            "GRE is less hard compared to GMAT. GRE is easy because it is easier than GMAT. GRE is not hard versus GMAT.",
            "GRE is easier than GMAT.",
        ),
        (
            "Plan A is cheap versus Plan B. Plan A is affordable compared to Plan B. Plan A is cheaper than Plan B.",
            "PLAN A is cheaper than PLAN B.",
        ),
        (
            "Model X is fast compared to Model Y. Model X is quick versus Model Y. Model X is faster than Model Y.",
            "MODEL X is faster than MODEL Y.",
        ),
    ]
    for text, expected in examples:
        result = optimize(text, "balanced")
        assert result["optimized_messages"][0].content == expected


def test_semantic_comparative_contradiction_not_compacted():
    text = "ACT is easier than SAT. ACT is harder than SAT."
    result = optimize(text, "balanced")
    assert result["optimized_messages"][0].content == text
    rejected = [
        item for item in result["removed_or_changed_text"]
        if item.get("backend") == "semantic_claim_compactor" and not item.get("accepted")
    ]
    assert rejected
    assert any("opposite semantic claim" in item.get("rejected_reason", "") for item in rejected)


def test_lowercase_reversed_preference_comparison_compacts_safely():
    text = "cr is better than messi. messi is not better than cr"
    result = optimize(text, "balanced")
    assert result["optimized_messages"][0].content == "CR is better than MESSI."
    assert "messi is not cr" not in result["optimized_messages"][0].content.lower()
    assert any(item.get("backend") == "semantic_claim_compactor" and item.get("accepted") for item in result["removed_or_changed_text"])


def test_token_backend_does_not_remove_comparison_relation_phrase():
    text = "alpha is better than beta. gamma is better than delta"
    result = optimize(text, "balanced")
    assert result["optimized_messages"][0].content == text
    rejected = [
        item for item in result["removed_or_changed_text"]
        if item.get("span_text", "").lower() == "better than"
    ]
    assert rejected
    assert any("semantic_relation_phrase" in item.get("risk_flags", []) for item in rejected)


def test_semantic_quality_claims_compact_across_paraphrases():
    examples = [
        ("Nimbus API is reliable. Nimbus API is stable. Nimbus API is very reliable.", "Nimbus API is reliable."),
        ("Checkout Flow is simple. Checkout Flow is easy. Checkout Flow is not hard.", "Checkout Flow is easy."),
    ]
    for text, expected in examples:
        result = optimize(text, "balanced")
        assert result["optimized_messages"][0].content == expected
        assert result["optimized_tokens"] < result["original_tokens"]


def test_semantic_requirement_claims_compact_across_paraphrases():
    text = "Refund Policy must require manager approval. Refund Policy requires manager approval. Refund Policy should require manager approval."
    result = optimize(text, "balanced")
    assert result["optimized_messages"][0].content == "Refund Policy must require manager approval."


def test_semantic_capability_claims_compact_across_paraphrases():
    text = "Search API can filter results. Search API supports filter results. Search API allows filter results."
    result = optimize(text, "balanced")
    assert result["optimized_messages"][0].content == "Search API can filter results."


def test_semantic_identity_claims_compact_across_paraphrases():
    text = "My name is Tanishq. Tanishq is my name. Tanishq is the name which I got."
    result = optimize(text, "balanced")
    assert result["optimized_messages"][0].content == "My name is Tanishq."
    assert any(
        item.get("backend") == "semantic_claim_compactor"
        and item.get("candidate_type") == "semantic_duplicate_identity_claim"
        and item.get("accepted")
        for item in result["removed_or_changed_text"]
    )


def test_semantic_identity_claims_reject_conflicting_names():
    text = "My name is Tanishq. My name is Alex."
    result = optimize(text, "balanced")
    assert result["optimized_messages"][0].content == text


def test_grammar_gate_rejects_orphan_auxiliary_end():
    text = "Tanishq is."
    result = optimize(text, "balanced")
    assert result["optimized_messages"][0].content == text
    assert not result["quality_gate_status"]["accepted"]


def test_semantic_sentence_cluster_compacts_non_rule_paraphrases():
    text = "Our onboarding checklist is simple for new hires. The onboarding checklist is easy for new hires to follow. New hires can follow the onboarding checklist without much difficulty."
    result = optimize(text, "balanced")
    assert result["optimized_messages"][0].content == "Our onboarding checklist is simple for new hires."
    assert "semantic_sentence_cluster" in result["backend_used"] or "semantic_optimizer_backend" in result["backend_used"]


def test_semantic_sentence_cluster_preserves_fact_differences():
    text = "Order ORD-123 costs $500. Order ORD-456 costs $500."
    result = optimize(text, "balanced")
    assert result["optimized_messages"][0].content == text


def test_grammar_gate_rejects_missing_subject_modal():
    text = "The onboarding checklist is easy for new hires to follow. can follow without much difficulty."
    result = optimize(text, "balanced")
    assert result["optimized_messages"][0].content == text


def test_natural_language_need_purpose_paraphrases_compact():
    text = "to visit there we might need train. we might need train to visit there. train is what we need to visit there."
    result = optimize(text, "balanced")
    assert result["optimized_messages"][0].content == "We might need a train to visit there."
    assert any(item.get("candidate_type") == "semantic_duplicate_need_claim" and item.get("accepted") for item in result["removed_or_changed_text"])


def test_natural_language_event_and_quality_paraphrases_compact():
    text = (
        "we might have met before. you know we might have met before. before we might have met. "
        "when we met before it was cheap. when we met before it was not expensive."
    )
    result = optimize(text, "balanced")
    assert result["optimized_messages"][0].content == "We might have met before. When we met before it was cheap."
    assert result["optimized_tokens"] < result["original_tokens"]


def test_natural_language_filler_modifiers_compress():
    text = "This is very very very important and really really useful"
    result = optimize(text, "balanced")
    output = result["optimized_messages"][0].content
    assert "very very" not in output
    assert "really really" not in output
    assert result["optimized_tokens"] < result["original_tokens"]


def test_natural_language_entity_preservation_without_object_loss():
    text = "Tanishq Kumar built the tool. Tanishq Kumar tested the tool."
    result = optimize(text, "balanced")
    assert result["optimized_messages"][0].content == text
    assert "Tanishq Kumar" in result["optimized_messages"][0].content
    assert "tested the tool" in result["optimized_messages"][0].content


def test_natural_language_contrast_not_overcompressed():
    text = "The ACT is easier than the SAT, but the SAT has fewer sections."
    result = optimize(text, "balanced")
    assert result["optimized_messages"][0].content == text


def test_natural_language_opposing_claims_preserved():
    for text in [
        "ACT is easier than SAT. SAT is easier than ACT.",
        "AI reduces cost. AI increases cost in some cases.",
    ]:
        result = optimize(text, "balanced")
        assert result["optimized_messages"][0].content == text


def test_natural_language_long_paragraph_preserves_unique_claims():
    text = (
        "The onboarding guide is simple for new employees. "
        "New employees can follow the onboarding guide without much difficulty. "
        "The onboarding guide is easy for new employees to follow. "
        "Managers must review access requests. "
        "Managers must review access requests. "
        "The payroll system remains separate from onboarding."
    )
    result = optimize(text, "balanced")
    output = result["optimized_messages"][0].content
    assert result["optimized_tokens"] < result["original_tokens"]
    assert "Managers must review access requests" in output
    assert "payroll system remains separate" in output
    assert any(
        backend in result["backend_used"]
        for backend in ["semantic_optimizer_backend", "semantic_sentence_cluster", "semantic_claim_compactor"]
    )


def test_natural_language_generator_compresses_comparison_paragraph():
    text = (
        "The ACT is easier than the SAT. Many students find the ACT less difficult than the SAT. "
        "Compared with the SAT, the ACT is generally considered the easier exam. "
        "The SAT is often viewed as harder than the ACT."
    )
    result = optimize(text, "balanced")
    output = result["optimized_messages"][0].content
    assert output == "The ACT is generally considered easier than the SAT."
    assert "semantic_optimizer_backend" in result["backend_used"]
    trace = next(item for item in result["removed_or_changed_text"] if item.get("backend") == "semantic_optimizer_backend")
    assert trace["accepted"] is True
    assert trace["grammar_validity"] is True
    assert trace["facts_preserved"] is True


def test_natural_language_generator_compresses_cost_paragraph_with_contrast():
    text = (
        "Our AI middleware reduces token costs by removing redundant context. "
        "The platform lowers AI spending by decreasing unnecessary token usage. "
        "By compressing repeated prompt content, the system helps companies reduce LLM API expenses. "
        "The main benefit of the middleware is lowering AI-related costs while preserving important context. "
        "However, the middleware should not remove code, IDs, dates, prices, or important business rules because those details may affect accuracy."
    )
    result = optimize(text, "balanced")
    output = result["optimized_messages"][0].content
    assert output == (
        "Our AI middleware lowers AI-related costs by compressing redundant prompt context while preserving important context. "
        "However, the middleware should not remove code, IDs, dates, prices, or important business rules because those details may affect accuracy."
    )
    assert "semantic_optimizer_backend" in result["backend_used"]
    assert "However" in output
    assert "IDs" in output
    assert result["optimized_tokens"] < result["original_tokens"]
