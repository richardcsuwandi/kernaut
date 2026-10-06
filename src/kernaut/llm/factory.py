from __future__ import annotations

from kernaut.config import AppConfig, ModelConfig
from kernaut.extensions import load_extension

from .base import LanguageModel
from .openai_compatible import OpenAICompatibleModel


def create_model(config: ModelConfig) -> LanguageModel:
    if config.provider in {"openai", "openai-compatible", "openrouter"}:
        return OpenAICompatibleModel(
            config.model,
            api_key=config.api_key(),
            base_url=config.base_url,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
            extra_body=config.extra_body,
        )
    if config.provider == "anthropic":
        from .anthropic_model import AnthropicModel

        return AnthropicModel(
            config.model, api_key=config.api_key(), timeout_seconds=config.timeout_seconds
        )
    if config.provider in {"claude-code", "codex"}:
        from .cli_model import SubscriptionCLIModel

        return SubscriptionCLIModel(
            config.provider,
            config.model,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
        )
    model = load_extension("kernaut.models", config.provider)(config)
    if not isinstance(model, LanguageModel):
        raise TypeError("Model factories must return a LanguageModel instance")
    return model


def build_model(config: AppConfig) -> LanguageModel:
    """Single model from [llm], or a Thompson-sampled ensemble from [[ensemble]]."""
    if not config.ensemble:
        if config.llm is None:
            raise ValueError("configuration requires [llm] or at least one [[ensemble]] entry")
        return create_model(config.llm)
    from .ensemble import EnsembleModel

    return EnsembleModel(
        [create_model(member) for member in config.ensemble],
        config.ensemble,
        seed=config.campaign.seed,
        bandit=config.bandit,
    )
