from app.main import app
from fastapi.testclient import TestClient


client = TestClient(app)


def test_api_keys_never_exposed():
    secret = "AQ.TEST_DUMMY_KEY_0000000000000000000000"
    response = client.post("/optimize", json={
        "messages": [{"role": "user", "content": f"Protect this key {secret}. Protect this key {secret}."}],
        "provider": "gemini",
        "mode": "dry-run",
    })
    assert response.status_code == 200
    text = response.text
    assert "GEMINI_API_KEY" not in text
    assert "TEST_DUMMY_KEY" in text
