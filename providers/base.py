from __future__ import annotations

from abc import ABC, abstractmethod


class ProviderResult(dict):
    pass


class BaseProvider(ABC):
    name: str

    @abstractmethod
    async def complete(self, messages: list[dict], model: str | None = None, temperature: float | None = 0) -> ProviderResult:
        raise NotImplementedError
