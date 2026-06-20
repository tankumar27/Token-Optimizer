from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_optimize_endpoint():
    response = client.post("/optimize", json={
        "messages": [{"role": "user", "content": "Refunds over $500 require Finance approval. Refunds over $500 require Finance approval."}],
        "compression_level": "safe",
        "provider": "gemini",
        "mode": "dry-run",
    })
    assert response.status_code == 200
    data = response.json()
    assert data["optimized_tokens"] <= data["original_tokens"]
    assert data["request_id"]


def test_main_endpoints():
    for path in ["/benchmark", "/evaluate-quality", "/robustness-test", "/company-pilot-sim", "/demo-report"]:
        response = client.post(path, json={"compression_level": "safe", "mode": "dry-run"})
        assert response.status_code == 200
    assert client.get("/evaluations").status_code == 200
    assert client.get("/robustness-results").status_code == 200
    assert client.get("/traces").status_code == 200
    assert client.get("/analytics").status_code == 200


def test_dashboard_renders():
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "Prompt Optimizer" in response.text
    assert "Company Pilot" in response.text


def test_chat_falls_back_without_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    response = client.post("/v1/chat/completions", json={
        "model": "gemini-1.5-flash",
        "messages": [{"role": "user", "content": "hello hello"}],
        "mode": "live",
    })
    assert response.status_code == 200
    data = response.json()
    assert data["provider"] in {"dry-run", "gemini"}
    assert "choices" in data


def test_openai_missing_key_falls_back(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    response = client.post("/v1/chat/completions", json={
        "provider": "openai",
        "messages": [{"role": "user", "content": "hello hello"}],
        "mode": "live",
    })
    assert response.status_code == 200
    assert response.json()["provider"] == "dry-run"
