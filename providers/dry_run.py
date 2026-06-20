from __future__ import annotations

from app.config import get_settings
from .base import BaseProvider, ProviderResult


class DryRunProvider(BaseProvider):
    name = "dry-run"

    async def complete(self, messages: list[dict], model: str | None = None, temperature: float | None = 0) -> ProviderResult:
        content = (
            "[dry-run structural validation only] Request passed through optimization middleware. "
            "No real model inference was performed."
        )
        return ProviderResult({
            "provider": self.name,
            "model": model or "dry-run",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "warning": get_settings().dry_run_warning,
        })
