from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.models import ChatMessage, OptimizeRequest, OptimizeResponse
from benchmark import run_benchmark
from demo import company_pilot_sim, generate_demo_report, readiness_score
from evaluation import evaluate_quality
from optimizer.pipeline import optimize_messages
from optimizer.cost_model import live_cost_report
from optimizer.token_counter import count_tokens
from providers.dry_run import DryRunProvider
from providers.gemini_provider import GeminiProvider
from providers.openai_provider import OpenAIProvider
from robustness import run_robustness
from storage.analytics import analytics_summary
from storage.db import (
    cache_get,
    cache_set,
    init_db,
    list_evaluations,
    list_robustness,
    recent_traces,
    save_evaluation,
    save_robustness,
    save_trace,
)
from storage.semantic_cache import semantic_cache_lookup
from optimizer.semantic_validator import SemanticValidator


settings = get_settings()
app = FastAPI(title=settings.app_name, version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

root = settings.project_root
reports_dir = root / "reports"
reports_dir.mkdir(exist_ok=True)
app.mount("/reports", StaticFiles(directory=str(reports_dir)), name="reports")
app.mount("/static", StaticFiles(directory=str(root / "dashboard")), name="static")


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.middleware("http")
async def request_limits(request: Request, call_next):
    size = request.headers.get("content-length")
    if size and int(size) > settings.request_size_limit_bytes:
        return JSONResponse(status_code=413, content={"error": {"message": "request too large", "code": "request_too_large"}})
    try:
        return await call_next(request)
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": {"message": type(exc).__name__, "code": "internal_error"}})


def _messages_to_dict(messages: list[ChatMessage]) -> list[dict[str, str]]:
    return [message.model_dump() for message in messages]


def _cache_key(req: OptimizeRequest, optimized_messages: list[ChatMessage], selected_model: str) -> str:
    system = "\n".join(m.content for m in req.messages if m.role == "system")
    payload = {
        "provider": req.provider,
        "model": selected_model,
        "optimized_prompt_hash": hashlib.sha256(json.dumps(_messages_to_dict(optimized_messages), sort_keys=True).encode()).hexdigest(),
        "system_prompt_hash": hashlib.sha256(system.encode()).hexdigest(),
        "compression_level": req.compression_level,
        "temperature": req.temperature,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _provider_for(provider: str, mode: str):
    if mode == "dry-run" or provider == "dry-run":
        return DryRunProvider()
    if provider == "openai":
        return OpenAIProvider()
    return GeminiProvider()


def _public_optimize(req: OptimizeRequest) -> dict[str, Any]:
    result = optimize_messages(req.messages, req.compression_level, req.provider, req.mode)
    result["warning"] = settings.dry_run_warning if req.mode == "dry-run" else None
    result["cache"] = {"hit": False}
    save_trace(result["traces"])
    return result


@app.get("/")
def index() -> dict:
    return {
        "name": settings.app_name,
        "routes": [
            "/health",
            "/optimize",
            "/v1/chat/completions",
            "/benchmark",
            "/evaluate-quality",
            "/evaluations",
            "/robustness-test",
            "/robustness-results",
            "/demo-report",
            "/company-pilot-sim",
            "/analytics",
            "/traces",
            "/dashboard",
        ],
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/validator-status")
def validator_status() -> dict:
    validator = SemanticValidator()
    similarity = validator.similarity("ACT is easier than SAT.", "ACT is simple compared to SAT.")
    transformer_active = validator.embedding_backend != "lexical"
    return {
        "enable_local_transformers": os.getenv("ENABLE_LOCAL_TRANSFORMERS") == "1",
        "packages": {
            "sentence_transformers": importlib.util.find_spec("sentence_transformers") is not None,
            "transformers": importlib.util.find_spec("transformers") is not None,
            "torch": importlib.util.find_spec("torch") is not None,
            "numpy": importlib.util.find_spec("numpy") is not None,
            "spacy": importlib.util.find_spec("spacy") is not None,
        },
        "active_embedding_backend": "transformer" if transformer_active else "lexical",
        "embedding_backend_detail": validator.embedding_backend,
        "fallback_used": not transformer_active,
        "local_generator_active": os.getenv("ENABLE_LOCAL_GENERATOR", "0") == "1",
        "active_ner_backend": validator.ner_backend,
        "active_grammar_backend": validator.grammar_backend,
        "sample_similarity": similarity,
    }


@app.post("/optimize")
def optimize(req: OptimizeRequest) -> dict:
    return _public_optimize(req)


@app.post("/v1/chat/completions")
async def chat_completions(payload: dict) -> dict:
    messages = [ChatMessage(role=m.get("role", "user"), content=m.get("content", "")) for m in payload.get("messages", [])]
    req = OptimizeRequest(
        messages=messages,
        compression_level=payload.get("compression_level", "safe"),
        provider=payload.get("provider", "gemini"),
        mode=payload.get("mode", "live"),
        model=payload.get("model"),
        temperature=payload.get("temperature", 0),
    )
    opt = _public_optimize(req)
    selected_model = req.model or opt["route_decision"]["selected_model"]
    system_prompt = "\n".join(m.content for m in req.messages if m.role == "system")
    optimized_prompt = json.dumps(_messages_to_dict(opt["optimized_messages"]), sort_keys=True)
    key = _cache_key(req, opt["optimized_messages"], selected_model)
    cached = cache_get(key)
    if cached and not (req.mode == "live" and cached.get("provider") != req.provider):
        opt["traces"]["cache_hit"] = True
        opt["traces"]["exact_hit"] = True
        return {"id": opt["request_id"], "object": "chat.completion", "middleware": opt, **cached}

    semantic = semantic_cache_lookup(optimized_prompt, req.provider, selected_model)
    opt["traces"]["semantic_cache"] = semantic["trace"]

    provider = _provider_for(req.provider, req.mode)
    provider_result = await provider.complete(_messages_to_dict(opt["optimized_messages"]), selected_model, req.temperature)
    usage = dict(provider_result).get("usage") or {}
    completion_text = ""
    choices = dict(provider_result).get("choices") or []
    if choices:
        completion_text = choices[0].get("message", {}).get("content", "") or ""
    output_tokens = usage.get("completion_tokens") or count_tokens(completion_text)
    prompt_tokens = usage.get("prompt_tokens")
    actual_provider = dict(provider_result).get("provider", req.provider)
    opt["cost"] = live_cost_report(
        actual_provider,
        selected_model,
        opt["original_tokens"],
        opt["optimized_tokens"],
        output_tokens,
        prompt_tokens,
    )
    opt["traces"].update(opt["cost"])
    opt["traces"]["provider_usage"] = usage
    opt["traces"]["live_provider"] = actual_provider
    if actual_provider == req.provider and not dict(provider_result).get("fallback_reason"):
        cache_set(key, dict(provider_result), max(0, opt["original_tokens"] - opt["optimized_tokens"]))
    return {"id": opt["request_id"], "object": "chat.completion", "middleware": opt, **provider_result}


@app.post("/benchmark")
async def benchmark(payload: dict | None = None) -> dict:
    return run_benchmark((payload or {}).get("compression_level", "safe"))


@app.post("/evaluate-quality")
async def evaluate_quality_endpoint(payload: dict | None = None) -> dict:
    body = payload or {}
    result = evaluate_quality(body.get("compression_level", "safe"), body.get("mode", "dry-run"), "gemini")
    save_evaluation(result)
    return result


@app.get("/evaluations")
def evaluations() -> dict:
    return {"items": list_evaluations()}


@app.post("/robustness-test")
async def robustness_endpoint(payload: dict | None = None) -> dict:
    result = run_robustness((payload or {}).get("compression_level", "safe"))
    save_robustness(result)
    return result


@app.get("/robustness-results")
def robustness_results() -> dict:
    return {"items": list_robustness()}


@app.post("/company-pilot-sim")
async def company_pilot_endpoint(payload: dict | None = None) -> dict:
    body = payload or {}
    return company_pilot_sim(body.get("compression_level", "safe"), body.get("mode", "dry-run"))


@app.post("/demo-report")
async def demo_report(payload: dict | None = None) -> dict:
    body = payload or {}
    report = company_pilot_sim(body.get("compression_level", "safe"), body.get("mode", "dry-run"))
    path = generate_demo_report(report)
    return {"report_path": path, "report_url": "/reports/demo_report.html", **report}


@app.get("/analytics")
def analytics() -> dict:
    summary = analytics_summary()
    readiness = readiness_score(
        {"total": {"savings_percent": summary["savings_percent"]}},
        {"results": []},
        {"failures": 0},
        summary,
        False,
    )
    return {**summary, "readiness_score": readiness["score"], "readiness_recommendation": readiness["recommendation"]}


@app.get("/traces")
def traces(limit: int = 50) -> dict:
    return {"items": recent_traces(limit)}


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    html = (root / "dashboard" / "dashboard.html").read_text(encoding="utf-8")
    return HTMLResponse(html)
