from pathlib import Path


def test_dashboard_contains_main_panels():
    html = Path("dashboard/dashboard.html").read_text(encoding="utf-8")
    js = Path("dashboard/dashboard.js").read_text(encoding="utf-8")
    for text in ["Prompt Optimizer", "Duplicate Chunk Graph", "Candidate Inspection", "Semantic Cache", "Provider And Cost", "Benchmark", "Quality Evaluation", "Robustness", "Company Pilot"]:
        assert text in html
    for endpoint in ["/optimize", "/benchmark", "/evaluate-quality", "/robustness-test", "/company-pilot-sim", "/analytics", "/traces"]:
        assert endpoint in js
