from __future__ import annotations

from pathlib import Path
from benchmark import run_benchmark
from evaluation import evaluate_quality
from robustness import run_robustness
from storage.analytics import analytics_summary


def readiness_score(benchmark: dict, quality: dict, robustness: dict, analytics: dict, live_validated: bool = False) -> dict:
    savings = benchmark["total"].get("savings_percent", 0)
    quality_failures = sum(1 for row in quality["results"] if not row.get("numeric_answer_preservation", True))
    robustness_failures = robustness.get("failures", 0)
    score = 55 + min(20, savings) - quality_failures * 10 - robustness_failures * 4
    score += min(10, analytics.get("cache_hit_rate", 0) / 10)
    if live_validated:
        score += 10
    score = max(0, min(85 if not live_validated else 100, round(score)))
    if score >= 90:
        label = "pilot_ready"
    elif score >= 75:
        label = "staging_only"
    elif score >= 50:
        label = "prototype_only"
    else:
        label = "unsafe"
    return {"score": score, "recommendation": label}


def company_pilot_sim(compression_level: str = "safe", mode: str = "dry-run") -> dict:
    bench = run_benchmark(compression_level)
    quality = evaluate_quality(compression_level, mode, "gemini")
    robust = run_robustness(compression_level)
    analytics = analytics_summary()
    readiness = readiness_score(bench, quality, robust, analytics, live_validated=mode == "live")
    return {
        "summary": {
            "status": "completed",
            "production_readiness_score": readiness["score"],
            "recommendation": readiness["recommendation"],
            "token_savings_percent": bench["total"]["savings_percent"],
            "estimated_cost_savings": bench["total"].get("cost_saved", 0.0),
            "quality_failures": sum(1 for row in quality["results"] if not row.get("numeric_answer_preservation", True)),
            "protected_failures": analytics.get("protected_failures", 0),
            "grammar_failures": sum(row.get("grammar_failures", 0) for row in bench["rows"]),
            "semantic_failures": sum(row.get("semantic_failures", 0) for row in bench["rows"]),
            "robustness_failures": robust["failures"],
            "dry_run_or_live": mode,
        },
        "benchmark": bench,
        "quality": quality,
        "robustness": robust,
        "rollout_recommendation": {
            "recommended_compression_level": "safe",
            "live_provider_test_needed": mode != "live",
            "staging_safe": readiness["score"] >= 75,
            "production_blocked": mode != "live" or readiness["score"] < 90,
        },
    }


def generate_demo_report(payload: dict) -> str:
    reports = Path("reports")
    reports.mkdir(exist_ok=True)
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>AI Cost Optimization Demo Report</title>
<style>body{{font-family:Arial,sans-serif;margin:40px;color:#172033}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #d6dbe6;padding:8px;text-align:left}}.warn{{background:#fff4d6;padding:12px;border:1px solid #e3bb4f}}</style></head>
<body><h1>AI Cost Optimization Middleware Demo Report</h1>
<div class="warn">Dry-run validates middleware mechanics only. It does not prove real model output quality.</div>
<h2>Executive summary</h2><pre>{payload["summary"]}</pre>
<h2>Rollout recommendation</h2><pre>{payload["rollout_recommendation"]}</pre>
<h2>Benchmark</h2><table><tr><th>Category</th><th>Original</th><th>Optimized</th><th>Savings</th></tr>
{''.join(f'<tr><td>{r["category"]}</td><td>{r["original_tokens"]}</td><td>{r["optimized_tokens"]}</td><td>{r["savings_percent"]}%</td></tr>' for r in payload["benchmark"]["rows"])}
</table><h2>Limitations</h2><p>No LLM judge is used. Dry-run mode cannot prove live model output quality.</p></body></html>"""
    path = reports / "demo_report.html"
    path.write_text(html, encoding="utf-8")
    return str(path)
