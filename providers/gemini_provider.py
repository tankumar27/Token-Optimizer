from __future__ import annotations

import httpx
from app.config import get_settings
from .base import BaseProvider, ProviderResult
from .dry_run import DryRunProvider


class GeminiProvider(BaseProvider):
    name = "gemini"

    async def complete(self, messages: list[dict], model: str | None = None, temperature: float | None = 0) -> ProviderResult:
        settings = get_settings()
        if not settings.gemini_api_key:
            fallback = await DryRunProvider().complete(messages, model, temperature)
            fallback["provider"] = "dry-run"
            fallback["fallback_reason"] = "GEMINI_API_KEY is not configured"
            return ProviderResult(fallback)

        model_name = model or settings.gemini_model
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        prompt = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature or 0},
        }
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(url, params={"key": settings.gemini_api_key}, json=payload)
                response.raise_for_status()
                data = response.json()
        except Exception as exc:
            fallback = await DryRunProvider().complete(messages, model_name, temperature)
            fallback["provider"] = "dry-run"
            fallback["fallback_reason"] = f"Gemini request failed: {type(exc).__name__}"
            return ProviderResult(fallback)

        text = ""
        candidates = data.get("candidates") or []
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(part.get("text", "") for part in parts)
        return ProviderResult({
            "provider": self.name,
            "model": model_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        })
