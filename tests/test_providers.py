from pathlib import Path

import pytest

from kernaut.config import AppConfig, ModelConfig
from kernaut.llm.factory import create_model
from kernaut.llm.openai_compatible import OpenAICompatibleModel

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*.toml")), ids=lambda p: p.name)
def test_example_configs_parse(path: Path) -> None:
    AppConfig.from_toml(path)


def test_provider_defaults() -> None:
    router = ModelConfig(provider="openrouter", model="qwen/qwen3-coder")
    assert router.base_url == "https://openrouter.ai/api/v1"
    assert router.api_key_env == "OPENROUTER_API_KEY" and router.needs_api_key
    assert ModelConfig(provider="openai", model="m").api_key_env == "OPENAI_API_KEY"
    assert ModelConfig(provider="anthropic", model="m").api_key_env == "ANTHROPIC_API_KEY"
    assert not ModelConfig(provider="claude-code", model="").needs_api_key
    with pytest.raises(ValueError, match="unknown provider"):
        ModelConfig(provider="gemini-cli", model="m")


def test_openrouter_builds_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    model = create_model(ModelConfig(provider="openrouter", model="qwen/qwen3-coder"))
    assert isinstance(model, OpenAICompatibleModel)
    assert str(model.client.base_url).startswith("https://openrouter.ai/api/v1")
