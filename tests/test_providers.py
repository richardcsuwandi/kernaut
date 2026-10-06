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


def test_openai_retry_budget_and_retry_after_header():
    from collections import UserDict
    from types import SimpleNamespace

    model = create_model(ModelConfig(provider="openai", model="test", max_retries=0))
    assert model.client.max_retries == 0
    error = SimpleNamespace(response=SimpleNamespace(headers=UserDict({"retry-after": "7"})))
    assert model._backoff_delay(0, error) == 7.0


def test_anthropic_uses_configured_retry_budget(monkeypatch):
    import sys
    from types import SimpleNamespace

    captured = {}

    def client(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=client))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    create_model(ModelConfig(provider="anthropic", model="test", max_retries=0))
    assert captured["max_retries"] == 0


@pytest.mark.parametrize("status", [408, 409, 429, 503, 401])
def test_openai_retries_only_transient_status_errors(status, monkeypatch):
    from types import SimpleNamespace

    import openai

    from kernaut.llm import openai_compatible
    from kernaut.llm.base import AssistantReply

    model = create_model(ModelConfig(provider="openai", model="test", max_retries=1))
    calls = []
    error = openai.APIStatusError(
        "test failure",
        response=SimpleNamespace(status_code=status, headers={}, request=None),
        body=None,
    )

    def consume(*args):
        calls.append(1)
        if len(calls) == 1:
            raise error
        return AssistantReply(content="Recovered")

    monkeypatch.setattr(model, "_consume_stream", consume)
    monkeypatch.setattr(openai_compatible.time, "sleep", lambda delay: None)
    if status == 401:
        with pytest.raises(openai.APIStatusError):
            model.complete([], [], "test")
        assert len(calls) == 1
    else:
        assert model.complete([], [], "test").content == "Recovered"
        assert len(calls) == 2


def test_anthropic_preserves_tool_call_ids_and_usage(monkeypatch):
    import sys
    from types import SimpleNamespace

    from kernaut.llm.base import ConversationMessage, RequestedTool

    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="tool_use", id="next-id", name="query_archive", input={"view": "recent"}
                )
            ],
            stop_reason="tool_use",
            usage=SimpleNamespace(input_tokens=12, output_tokens=3),
        )

    monkeypatch.setitem(
        sys.modules,
        "anthropic",
        SimpleNamespace(
            Anthropic=lambda **kw: SimpleNamespace(messages=SimpleNamespace(create=create))
        ),
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    model = create_model(ModelConfig(provider="anthropic", model="test"))
    call = RequestedTool(call_id="previous-id", name="query_archive", arguments={"view": "recent"})
    messages = [
        ConversationMessage(role="user", content="go"),
        ConversationMessage(role="assistant", tool_calls=[call]),
        ConversationMessage(role="tool", tool_call_id="previous-id", content="[]"),
    ]
    reply = model.complete(messages, [], "system")
    assert captured["messages"][2]["content"][0]["tool_use_id"] == "previous-id"
    assert reply.tool_calls[0].call_id == "next-id"
    assert reply.tool_calls[0].arguments == {"view": "recent"}
    assert reply.usage == {"input_tokens": 12, "output_tokens": 3}
