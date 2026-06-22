from functools import lru_cache
from pathlib import Path
from pydantic import BaseModel, Field
import os


def _default_database_path() -> str:
    if os.getenv("VERCEL"):
        return "/tmp/app.sqlite3"
    return "data/app.sqlite3"


class Settings(BaseModel):
    app_name: str = "AI Cost Optimization Middleware"
    database_path: str = Field(default_factory=lambda: os.getenv("DATABASE_PATH", _default_database_path()))
    request_size_limit_bytes: int = Field(default_factory=lambda: int(os.getenv("REQUEST_SIZE_LIMIT_BYTES", "1000000")))
    cors_origins: list[str] = ["*"]
    gemini_api_key: str | None = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    gemini_model: str = Field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    openai_api_key: str | None = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    openai_model: str = Field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    enable_llm_lingua: bool = Field(default_factory=lambda: os.getenv("ENABLE_LLM_LINGUA", "0") == "1")
    llm_lingua_backend: str = Field(default_factory=lambda: os.getenv("LLM_LINGUA_BACKEND", "llmlingua2"))
    llm_lingua_model: str = Field(default_factory=lambda: os.getenv("LLM_LINGUA_MODEL", os.getenv("GEMINI_MODEL", "gemini-1.5-flash")))
    llm_lingua2_model: str = Field(default_factory=lambda: os.getenv("LLM_LINGUA2_MODEL", "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"))
    llm_lingua2_device: str = Field(default_factory=lambda: os.getenv("LLM_LINGUA2_DEVICE", "cpu"))
    llm_lingua_timeout_seconds: float = Field(default_factory=lambda: float(os.getenv("LLM_LINGUA_TIMEOUT_SECONDS", "20")))
    default_provider: str = "gemini"
    dry_run_warning: str = (
        "Dry-run validates middleware mechanics only. It does not prove real model output quality."
    )

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[1]


@lru_cache
def get_settings() -> Settings:
    return Settings()

