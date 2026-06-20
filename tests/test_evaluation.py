from evaluation import evaluate_quality


def test_evaluation_runs():
    result = evaluate_quality("safe", "dry-run", "gemini")
    assert result["results"][0]["structural_validation_only"] is True
