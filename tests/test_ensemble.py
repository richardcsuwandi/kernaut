from __future__ import annotations

import json
import random
from typing import Any

import pytest

from kernaut.agent import HarnessTools
from kernaut.archive import CandidateStore
from kernaut.config import AppConfig, BanditConfig, ExecutionConfig, ModelConfig
from kernaut.evaluation import Dataset, GaussianProcessEvaluator
from kernaut.execution import SubprocessExecutor
from kernaut.llm import (
    AssistantReply,
    ConversationMessage,
    EnsembleModel,
    LanguageModel,
    RequestedTool,
)
from kernaut.verification import VerificationPolicy, Verifier

SOURCE = "def feature_point(x, parameters):\n    return x\n"


class ScriptedModel(LanguageModel):
    """Submits and evaluates one candidate, then finishes."""

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


class SubmitEvaluateModel(LanguageModel):
    """Submits a candidate, then verifies and evaluates the submitted ID."""

    def __init__(self) -> None:
        self.turn = 0
        self.candidate_id = ""

    def complete(
        self,
        messages: list[ConversationMessage],
        tools: list[dict[str, Any]],
        system_prompt: str,
    ) -> AssistantReply:
        if self.candidate_id == "" and messages[-1].role == "tool":
            payload = json.loads(messages[-1].content)
            self.candidate_id = str(payload.get("candidate_id", ""))
        if self.turn == 0:
            call = RequestedTool(
                call_id="submit-1",
                name="submit_candidate",
                arguments={"name": "linear_features", "contract": "feature_map", "source": SOURCE},
            )
        elif self.turn in (1, 2):
            name = "verify_candidate" if self.turn == 1 else "evaluate_candidate"
            call = RequestedTool(
                call_id=f"{name.split('_')[0]}-1",
                name=name,
                arguments={"candidate_id": self.candidate_id},
            )
        else:
            return AssistantReply(content="Campaign complete.", usage={})
        self.turn += 1
        return AssistantReply(tool_calls=[call], usage={"input_tokens": 1, "output_tokens": 1})


def _bandit(**overrides: Any) -> BanditConfig:
    values: dict[str, Any] = {"window_size": 5, "min_samples": 2}
    values.update(overrides)
    return BanditConfig(**values)


def _member(name: str) -> ModelConfig:
    return ModelConfig(model=name, base_url="https://example.invalid/v1")


def test_warmup_round_robin_then_thompson_converges() -> None:
    sampler = EnsembleModel(
        [ScriptedModel("a"), ScriptedModel("b"), ScriptedModel("c")],
        [_member("a"), _member("b"), _member("c")],
        seed=0,
        bandit=_bandit(),
    ).sampler
    rng = random.Random(0)

    picks = []
    for _ in range(6):
        arm = sampler.select(rng)
        picks.append(arm)
        sampler.update(arm, 0.5)
    assert picks[:3] == [0, 1, 2]
    assert all(count == 2 for count in (len(window) for window in sampler.history))

    for _ in range(20):
        for arm in range(3):
            sampler.update(arm, {0: 0.9, 1: 0.1, 2: 0.5}[arm])
    dominant = [sampler.select(rng) for _ in range(60)]
    assert dominant.count(0) > dominant.count(1)


def test_sliding_window_forgets_old_rewards() -> None:
    sampler = EnsembleModel(
        [ScriptedModel("a"), ScriptedModel("b")],
        [_member("a"), _member("b")],
        seed=0,
        bandit=_bandit(window_size=3),
    ).sampler
    for arm in range(2):
        sampler.update(arm, 1.0 if arm == 0 else 0.0)

    assert sampler.posterior_mean(0) > sampler.posterior_mean(1)
    for _ in range(10):
        sampler.update(0, 0.0)
        sampler.update(1, 1.0)

    assert sampler.posterior_mean(1) > sampler.posterior_mean(0)
    assert len(sampler.history[0]) == 3


def test_score_normalization_tracks_running_extremes() -> None:
    ensemble = EnsembleModel(
        [ScriptedModel("a")],
        [_member("a")],
        seed=0,
        bandit=_bandit(),
    )
    ensemble.end_campaign(0, -12.0)
    assert ensemble.sampler.history[0][0] == pytest.approx(0.5)
    ensemble.end_campaign(0, -8.0)
    assert ensemble.sampler.history[0][-1] == pytest.approx(1.0)
    ensemble.end_campaign(0, -16.0)
    assert ensemble.sampler.history[0][-1] == pytest.approx(0.0)
    ensemble.end_campaign(0, None)
    assert ensemble.sampler.history[0][-1] == pytest.approx(0.0)


def test_complete_delegates_to_active_campaign_model() -> None:
    first, second = ScriptedModel("a"), ScriptedModel("b")
    ensemble = EnsembleModel(
        [first, second],
        [_member("a"), _member("b")],
        seed=0,
        bandit=_bandit(),
    )
    index, child = ensemble.begin_campaign()
    ensemble.complete([ConversationMessage(role="user", content="go")], [], "system")
    assert child.turn == 1
    other = second if index == 0 else first
    assert other.turn == 0
    ensemble.end_campaign(index, 0.4)
    ensemble.complete([ConversationMessage(role="user", content="go")], [], "system")
    assert other.turn == 1 or child.turn == 2


def test_appconfig_inherits_llm_connection_into_ensemble() -> None:
    config = AppConfig.model_validate(
        {
            "llm": {
                "model": "Qwen/Qwen3.8-Max",
                "base_url": "https://api-inference.modelscope.ai/v1",
                "api_key_env": "MODELSCOPE_API_KEY",
            },
            "ensemble": [
                {"model": "zai-org/GLM-5.2", "weight": 0.7},
                {"model": "deepseek-ai/DeepSeek-V4-Pro"},
            ],
        }
    )
    assert config.llm is not None and config.ensemble[0].base_url == config.llm.base_url
    assert config.ensemble[1].weight == 1.0
    assert config.ensemble[1].api_key_env == "MODELSCOPE_API_KEY"
    assert config.bandit.window_size == 20

    with pytest.raises(ValueError):
        AppConfig.model_validate({"campaign": {"seed": 1}})


def test_evolutionary_loop_assigns_and_rewards_per_campaign(tmp_path) -> None:
    store = CandidateStore(tmp_path / "archive.sqlite")
    executor = SubprocessExecutor(ExecutionConfig(timeout_seconds=10, memory_mb=1024))
    verifier = Verifier(
        executor, VerificationPolicy(trials=1, points_per_trial=4, input_dimension=1)
    )
    dataset = Dataset(x=[[-1.0], [0.0], [1.0]], y=[1.0, 0.0, 1.0])
    tools = HarnessTools(store, verifier, GaussianProcessEvaluator(executor), dataset)

    from kernaut.agent import EvolutionarySynthesisController

    models = {
        "alpha": SubmitEvaluateModel(),
        "beta": SubmitEvaluateModel(),
    }
    ensemble = EnsembleModel(
        list(models.values()),
        [_member(name) for name in models],
        seed=11,
        bandit=_bandit(min_samples=1),
    )
    controller = EvolutionarySynthesisController(
        ensemble,
        tools,
        store,
        iterations=3,
        rounds_per_iteration=8,
        tool_calls_per_iteration=10,
        root_probability=1.0,
        exploration_probability=0.0,
        seed=5,
    )
    result = controller.run("Find a kernel", run_id="ensemble-test")

    assert result.completed_iterations == 3
    assert sum(ensemble.picks) == 3
    assert ensemble.picks[0] >= 1 and ensemble.picks[1] >= 1
    rewards = [reward for window in ensemble.sampler.history for reward in window]
    assert any(reward == pytest.approx(0.5) for reward in rewards)
    assert len(rewards) == 3
