from typing import Any, Literal
from pydantic import BaseModel, Field


CompressionLevel = Literal["safe", "balanced", "aggressive"]
ProviderName = Literal["dry-run", "gemini", "openai"]
ModeName = Literal["dry-run", "live"]


class ChatMessage(BaseModel):
    role: str
    content: str


class OptimizeRequest(BaseModel):
    messages: list[ChatMessage]
    compression_level: CompressionLevel = "safe"
    provider: ProviderName = "gemini"
    mode: ModeName = "dry-run"
    model: str | None = None
    temperature: float | None = 0


class OptimizeResponse(BaseModel):
    request_id: str
    original_messages: list[ChatMessage]
    optimized_messages: list[ChatMessage]
    original_tokens: int
    optimized_tokens: int
    savings_percent: float
    backend_used: list[str]
    protected_region_status: dict[str, Any]
    quality_gate_status: dict[str, Any]
    traces: dict[str, Any]
    removed_or_changed_text: list[dict[str, Any]]
    duplicate_chunk_graph: list[dict[str, Any]]
    cache: dict[str, Any] = Field(default_factory=dict)
    cost: dict[str, Any] = Field(default_factory=dict)
    route_decision: dict[str, Any] = Field(default_factory=dict)
    grammar_status: list[dict[str, Any]] = Field(default_factory=list)
    semantic_status: list[dict[str, Any]] = Field(default_factory=list)
    warning: str | None = None
