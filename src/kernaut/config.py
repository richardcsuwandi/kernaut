from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from kernaut.extensions import extension_names

API_PROVIDERS = {"openai", "openai-compatible", "openrouter", "anthropic"}
CLI_PROVIDERS = {"claude-code", "codex"}  # use the logged-in CLI subscription, no API key
DEFAULT_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}
OPENROUTER_URL = "https://openrouter.ai/api/v1"


class ModelConfig(BaseModel):
    provider: str = "openai-compatible"
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    timeout_seconds: float = Field(default=180.0, gt=0)
    max_retries: int = Field(default=4, ge=0)
    extra_body: dict[str, Any] = Field(default_factory=dict)
    weight: float = Field(default=1.0, gt=0)

    @model_validator(mode="after")
    def require_compatible_url(self) -> ModelConfig:
        if self.provider not in API_PROVIDERS | CLI_PROVIDERS | set(
            extension_names("kernaut.models")
        ):
            raise ValueError(
                f"unknown provider {self.provider!r}; expected one of "
                f"{sorted(API_PROVIDERS | CLI_PROVIDERS)}"
            )
        if self.provider == "openrouter" and not self.base_url:
            self.base_url = OPENROUTER_URL
        if self.provider == "openai-compatible" and not self.base_url:
            raise ValueError("openai-compatible requires base_url")
        if self.api_key_env is None:
            self.api_key_env = DEFAULT_KEY_ENV.get(self.provider)
        return self

    @property
    def needs_api_key(self) -> bool:
        return self.provider in API_PROVIDERS or self.api_key_env is not None

    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) if self.api_key_env else None


class BanditConfig(BaseModel):
    window_size: int = Field(default=20, ge=1)
    min_samples: int = Field(default=2, ge=1)
    alpha_prior: float = Field(default=1.0, gt=0)
    beta_prior: float = Field(default=1.0, gt=0)


class CampaignConfig(BaseModel):
    max_rounds: int = Field(default=20, ge=1)
    max_tool_calls: int = Field(default=100, ge=1)
    seed: int = 0


class ExecutionConfig(BaseModel):
    timeout_seconds: float = Field(default=20.0, gt=0)
    memory_mb: int = Field(default=1024, ge=128)
    cpu_seconds: int = Field(default=15, ge=1)
    max_output_bytes: int = Field(default=1_000_000, ge=1024)


class AppConfig(BaseModel):
    llm: ModelConfig | None = None
    ensemble: list[ModelConfig] = Field(default_factory=list)
    bandit: BanditConfig = Field(default_factory=BanditConfig)
    campaign: CampaignConfig = Field(default_factory=CampaignConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)

    @model_validator(mode="before")
    @classmethod
    def inherit_connection_settings(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        llm = data.get("llm")
        ensemble = data.get("ensemble")
        if isinstance(llm, dict) and isinstance(ensemble, list):
            inherited = {
                key: llm[key]
                for key in ("provider", "base_url", "api_key_env", "timeout_seconds", "max_retries")
                if key in llm and llm[key] is not None
            }
            for member in ensemble:
                if isinstance(member, dict):
                    for key, value in inherited.items():
                        member.setdefault(key, value)
        return data

    @model_validator(mode="after")
    def require_model_source(self) -> AppConfig:
        if self.llm is None and not self.ensemble:
            raise ValueError("configuration requires [llm] or at least one [[ensemble]] entry")
        return self

    @classmethod
    def from_toml(cls, path: str | Path) -> AppConfig:
        with Path(path).open("rb") as handle:
            return cls.model_validate(tomllib.load(handle))
