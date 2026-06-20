from app.models import ChatMessage
from optimizer.pipeline import optimize_messages


def _opt(text: str) -> str:
    result = optimize_messages([ChatMessage(role="user", content=text)], "balanced", "dry-run", "dry-run")
    return result["optimized_messages"][0].content


def test_direction_reversal_equivalence_and_contradiction():
    assert _opt("The ACT is easier than the SAT. The SAT is harder than the ACT.") == "The ACT is easier than the SAT."
    contradictory = "The ACT is easier than the SAT. The SAT is easier than the ACT."
    assert _opt(contradictory) == contradictory


def test_modality_equivalence_preserves_uncertainty_but_not_certainty_shift():
    compressible = (
        "We might need additional funding next quarter. "
        "Additional funding may be required next quarter. "
        "It is possible we will need more funding next quarter."
    )
    assert _opt(compressible) == "We might need additional funding next quarter."
    mixed = "We might need additional funding next quarter. We will need additional funding next quarter."
    assert _opt(mixed) == mixed


def test_negated_evidence_equivalence_and_contradiction():
    assert _opt(
        "The medicine has not been proven effective. "
        "There is currently no evidence that the medicine is effective."
    ) == "The medicine has not been proven effective."
    contradictory = "The medicine has not been proven effective. The medicine has been proven effective."
    assert _opt(contradictory) == contradictory


def test_hidden_numeric_unit_equivalence_compacts_without_losing_units():
    output = _opt(
        "The server responded in under 100 milliseconds. "
        "Requests completed in less than 0.1 seconds. "
        "Latency remained below one tenth of a second."
    )
    assert output == "Latency remained below 100 milliseconds (0.1 seconds)."


def test_entity_swap_and_same_entity_representation():
    assert _opt("Messi plays for Argentina. Messi represents Argentina internationally.") == "Messi represents Argentina internationally."
    different = "Messi plays for Argentina. Ronaldo plays for Portugal."
    assert _opt(different) == different


def test_causation_equivalence_but_not_different_effects():
    assert _opt("Removing redundant context reduces token costs. Lower token counts help reduce API expenses.") == "Removing redundant context reduces token costs."
    different = "Removing redundant context reduces token costs. Lower token counts cause hallucinations."
    assert _opt(different) == different


def test_company_prompt_compresses_without_dropping_critical_constraints():
    text = (
        "The platform lowers AI costs by reducing unnecessary token usage. "
        "By removing redundant prompt content, the system decreases API expenses. "
        "The middleware helps organizations spend less on large language model inference by compressing repeated context. "
        "One of the primary benefits of the platform is reducing AI-related costs while preserving useful information. "
        "However, the system must preserve important business rules, identifiers, dates, and constraints because removing them may reduce response quality. "
        "Furthermore, preserving critical information is essential to maintain reliability."
    )
    output = _opt(text)
    assert len(output.split()) < len(text.split())
    for phrase in ["AI-related costs", "redundant prompt context", "business rules", "identifiers", "dates", "constraints", "reliability"]:
        assert phrase in output

