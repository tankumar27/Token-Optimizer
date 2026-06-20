from __future__ import annotations

import httpx
from app.config import get_settings
from .base import BaseProvider, ProviderResult
from .dry_run import DryRunProvider


class OpenAIProvider(BaseProvider):
    name = "openai"

    async def complete(self, messages: list[dict], model: str | None = None, temperature: float | None = 0) -> ProviderResult:
        api_key = getattr(get_settings(), "openai_api_key", None)
        if not api_key:
            fallback = await DryRunProvider().complete(messages, model, temperature)
            fallback["provider"] = "dry-run"
            fallback["fallback_reason"] = "OPENAI_API_KEY is not configured"
            return ProviderResult(fallback)
        model_name = model or "gpt-4o-mini"
        payload = {"model": model_name, "messages": messages, "temperature": temperature or 0}
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
        except Exception as exc:
            fallback = await DryRunProvider().complete(messages, model_name, temperature)
            fallback["provider"] = "dry-run"
            fallback["fallback_reason"] = f"OpenAI request failed: {type(exc).__name__}"
            return ProviderResult(fallback)
        data["provider"] = "openai"
        return ProviderResult(data)
