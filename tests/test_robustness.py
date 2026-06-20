from robustness import run_robustness


def test_robustness_runs():
    result = run_robustness("safe")
    assert len(result["rows"]) >= 18
