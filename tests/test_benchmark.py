from benchmark import run_benchmark


def test_benchmark_runs():
    result = run_benchmark("safe")
    assert len(result["rows"]) >= 12
    assert result["total"]["original_tokens"] >= result["total"]["optimized_tokens"]
