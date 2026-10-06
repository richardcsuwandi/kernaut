from kernaut.llm import ConversationMessage
from kernaut.models import CandidateBundle


def test_candidate_id_is_content_addressed_but_ignores_rationale() -> None:
    first = CandidateBundle(
        name="demo", contract="feature_map", source="def feature_point(x, parameters): return x"
    )
    second = CandidateBundle(
        name="demo", contract="feature_map", source=first.source, rationale="new explanation"
    )
    assert first.candidate_id == second.candidate_id


def test_stream_disconnect_is_retried_and_recovers() -> None:
    from types import SimpleNamespace

    httpx2 = __import__("httpx2")

    class FlakyClient:
        def __init__(self) -> None:
            self.attempts = 0

        @property
        def chat(self):
            return self

        @property
        def completions(self):
            return self

        def create(self, **kwargs):
            self.attempts += 1
            if self.attempts == 1:
                raise httpx2.RemoteProtocolError("peer closed connection")
            chunk = SimpleNamespace(
                usage=None,
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="ok", tool_calls=None),
                        finish_reason="stop",
                    )
                ],
            )
            return iter([chunk])

    from kernaut.config import ModelConfig
    from kernaut.llm.openai_compatible import OpenAICompatibleModel

    config = ModelConfig(model="stub", base_url="https://example.invalid/v1", max_retries=3)
    model = OpenAICompatibleModel(
        config.model,
        api_key=None,
        base_url=config.base_url,
        timeout_seconds=5,
        max_retries=3,
    )
    model.client = FlakyClient()
    reply = model.complete([ConversationMessage(role="user", content="hi")], [], "system")
    assert reply.content == "ok"
    assert model.client.attempts == 2
