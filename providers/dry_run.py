from __future__ import annotations

from app.config import get_settings
from optimizer.token_counter import count_message_tokens, count_tokens
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
            "usage": {
                "prompt_tokens": count_message_tokens(messages),
                "completion_tokens": count_tokens(content),
                "total_tokens": count_message_tokens(messages) + count_tokens(content),
            },
            "warning": get_settings().dry_run_warning,
        })
