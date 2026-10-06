from __future__ import annotations

from typing import Any

from kernaut.agent import HarnessTools, SynthesisController
from kernaut.archive import CandidateStore
from kernaut.config import ExecutionConfig
from kernaut.evaluation import Dataset, GaussianProcessEvaluator
from kernaut.execution import SubprocessExecutor
from kernaut.llm import AssistantReply, ConversationMessage, LanguageModel, RequestedTool
from kernaut.models import CandidateBundle, EvidenceTier
from kernaut.verification import VerificationPolicy, Verifier

SOURCE = "def feature_point(x, parameters):\n    return x\n"


class ScriptedModel(LanguageModel):
    def __init__(self, candidate_id: str) -> None:
        self.candidate_id = candidate_id
        self.turn = 0

    def complete(
        self,
        messages: list[ConversationMessage],
        tools: list[dict[str, Any]],
        system_prompt: str,
    ) -> AssistantReply:
        scripts = [
            RequestedTool(
                call_id="submit-1",
                name="submit_candidate",
                arguments={"name": "linear_features", "contract": "feature_map", "source": SOURCE},
            ),
            RequestedTool(
                call_id="verify-1",
                name="verify_candidate",
                arguments={"candidate_id": self.candidate_id},
            ),
            RequestedTool(
                call_id="evaluate-1",
                name="evaluate_candidate",
                arguments={"candidate_id": self.candidate_id},
            ),
        ]
        if self.turn < len(scripts):
            reply = AssistantReply(
                tool_calls=[scripts[self.turn]], usage={"input_tokens": 1, "output_tokens": 1}
            )
        else:
            reply = AssistantReply(content="Campaign complete.")
        self.turn += 1
        return reply


class CompletesAfterRecoveryModel(LanguageModel):
    def complete(
        self,
        messages: list[ConversationMessage],
        tools: list[dict[str, Any]],
        system_prompt: str,
    ) -> AssistantReply:
        assert messages[-1].role == "tool"
        assert "retry the tool call" in messages[-1].content
        assert messages[-1].tool_call_id == "interrupted-1"
        return AssistantReply(content="Recovered campaign complete.")


class FailIfCalledModel(LanguageModel):
    def complete(
        self,
        messages: list[ConversationMessage],
        tools: list[dict[str, Any]],
        system_prompt: str,
    ) -> AssistantReply:
        raise AssertionError("completed campaigns must not call the model when resumed")


def test_offline_agent_campaign(tmp_path) -> None:
    candidate = CandidateBundle(name="linear_features", contract="feature_map", source=SOURCE)
    store = CandidateStore(tmp_path / "archive.sqlite")
    executor = SubprocessExecutor(
        ExecutionConfig(timeout_seconds=10, memory_mb=1024, cpu_seconds=8)
    )
    verifier = Verifier(
        executor, VerificationPolicy(trials=1, points_per_trial=4, input_dimension=1)
    )
    dataset = Dataset(x=[[-1.0], [0.0], [1.0]], y=[1.0, 0.0, 1.0])
    tools = HarnessTools(store, verifier, GaussianProcessEvaluator(executor), dataset)
    result = SynthesisController(ScriptedModel(candidate.candidate_id), tools, store).run(
        "Find a kernel", run_id="test-run"
    )
    assert result.final_message == "Campaign complete."
    assert result.tool_calls == 3
    assert store.latest_evidence(candidate.candidate_id).tier == EvidenceTier.CONTRACT_CERTIFIED
    assert store.latest_evaluation(candidate.candidate_id) is not None
    assert len(store.load_events("test-run")) == 8


def test_campaign_recovers_a_tool_call_with_no_checkpointed_result(tmp_path) -> None:
    store = CandidateStore(tmp_path / "archive.sqlite")
    store.append_event(
        "interrupted-run",
        0,
        ConversationMessage(role="user", content="Find a kernel").model_dump(mode="json"),
    )
    store.append_event(
        "interrupted-run",
        1,
        ConversationMessage(
            role="assistant",
            tool_calls=[
                RequestedTool(
                    call_id="interrupted-1",
                    name="optimize_parameters",
                    arguments={"draft_id": "draft-1"},
                )
            ],
        ).model_dump(mode="json"),
    )
    executor = SubprocessExecutor()
    verifier = Verifier(executor)
    dataset = Dataset(x=[[-1.0], [1.0]], y=[1.0, 1.0])
    tools = HarnessTools(store, verifier, GaussianProcessEvaluator(executor), dataset)

    result = SynthesisController(CompletesAfterRecoveryModel(), tools, store).run(
        "ignored when resuming", run_id="interrupted-run"
    )

    events = store.load_events("interrupted-run")
    assert result.final_message == "Recovered campaign complete."
    assert events[2]["role"] == "tool"
    assert events[2]["tool_call_id"] == "interrupted-1"
    assert events[3]["content"] == "Recovered campaign complete."


def test_campaign_resume_skips_a_completed_checkpoint(tmp_path) -> None:
    store = CandidateStore(tmp_path / "archive.sqlite")
    store.append_event(
        "completed-run",
        0,
        ConversationMessage(role="user", content="Find a kernel").model_dump(mode="json"),
    )
    store.append_event(
        "completed-run",
        1,
        ConversationMessage(role="assistant", content="Campaign complete.").model_dump(mode="json"),
    )
    executor = SubprocessExecutor()
    verifier = Verifier(executor)
    dataset = Dataset(x=[[-1.0], [1.0]], y=[1.0, 1.0])
    tools = HarnessTools(store, verifier, GaussianProcessEvaluator(executor), dataset)

    result = SynthesisController(FailIfCalledModel(), tools, store).run(
        "ignored when resuming", run_id="completed-run"
    )

    assert result.final_message == "Campaign complete."
    assert result.rounds == 1
    assert result.tool_calls == 0
    assert len(store.load_events("completed-run")) == 2
