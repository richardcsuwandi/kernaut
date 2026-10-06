from __future__ import annotations

from importlib.metadata import EntryPoint

import pytest

from kernaut import extensions
from kernaut.agent.controller import CampaignResult
from kernaut.archive import CandidateStore
from kernaut.config import CampaignConfig, ModelConfig
from kernaut.evaluation.gp import Dataset, GaussianProcessEvaluator
from kernaut.execution import SubprocessExecutor
from kernaut.llm.base import AssistantReply, LanguageModel
from kernaut.llm.factory import create_model
from kernaut.models import CandidateBundle, CandidateOrigin
from kernaut.tasks import Task, run_task
from kernaut.verification import VerificationPolicy


class QuietModel(LanguageModel):
    def complete(self, messages, tools, system_prompt):
        return AssistantReply(content="Finished without a provider call.")


def make_model(config):
    return QuietModel()


def test_discovery_does_not_import_and_duplicate_names_fail(monkeypatch):
    entries = [EntryPoint(name="demo", value="missing_module:factory", group="kernaut.tasks")]
    monkeypatch.setattr(extensions, "entry_points", lambda **kw: entries)
    assert extensions.extension_names("kernaut.tasks") == ("demo",)
    entries.append(entries[0])
    with pytest.raises(ValueError, match="Multiple packages"):
        extensions.load_extension("kernaut.tasks", "demo")


def test_unknown_extension_has_actionable_error(monkeypatch):
    monkeypatch.setattr(extensions, "entry_points", lambda **kw: [])
    with pytest.raises(ValueError, match="Install its package"):
        extensions.load_extension("kernaut.tasks", "absent")


def test_installed_provider_can_be_configured_and_constructed(monkeypatch):
    entries = [
        EntryPoint(name="test-provider", value="test_extensions:make_model", group="kernaut.models")
    ]
    monkeypatch.setattr(extensions, "entry_points", lambda **kw: entries)
    config = ModelConfig(provider="test-provider", model="test", api_key_env="TEST_PROVIDER_KEY")
    assert isinstance(create_model(config), QuietModel)
    assert config.needs_api_key


def test_non_model_factory_is_rejected(monkeypatch):
    entry = EntryPoint(name="bad-provider", value="builtins:dict", group="kernaut.models")
    monkeypatch.setattr(extensions, "entry_points", lambda **kw: [entry])
    with pytest.raises(TypeError, match="LanguageModel"):
        create_model(ModelConfig(provider="bad-provider", model="test"))


def task_and_executor():
    executor = SubprocessExecutor()
    task = Task(
        dataset=Dataset(x=[[-1.0], [0.0], [1.0]], y=[1.0, 0.0, 1.0]),
        context="Find a kernel for the training data.",
        evaluator=GaussianProcessEvaluator(executor),
        verification_policy=VerificationPolicy(trials=1, points_per_trial=3, input_dimension=1),
    )
    return task, executor


def test_baseline_is_verified_scored_and_archived(tmp_path):
    task, executor = task_and_executor()
    store = CandidateStore(tmp_path / "archive.sqlite")
    candidate = CandidateBundle(
        name="linear", contract="feature_map", source="def feature_point(x, parameters): return x"
    )
    result = run_task(
        task,
        QuietModel(),
        store,
        executor,
        campaign=CampaignConfig(max_rounds=1),
        baselines=[candidate],
    )
    assert isinstance(result, CampaignResult)
    assert store.get_candidate(candidate.candidate_id).origin == CandidateOrigin.BASELINE
    assert store.latest_evidence(candidate.candidate_id).accepted
    assert store.latest_evaluation(candidate.candidate_id) is not None


def test_rejected_baseline_never_reaches_evaluator_or_model(tmp_path):
    task, executor = task_and_executor()

    class MustNotRun:
        def evaluate(self, *args):
            pytest.fail("rejected baseline reached evaluator")

        def complete(self, *args):
            pytest.fail("model called after a failed baseline")

    invalid = CandidateBundle(name="invalid", contract="feature_map", source="x = 1")
    guarded = Task(dataset=task.dataset, context=task.context, evaluator=MustNotRun())
    store = CandidateStore(tmp_path / "archive.sqlite")
    with pytest.raises(ValueError, match="failed verification"):
        run_task(guarded, MustNotRun(), store, executor, baselines=[invalid])
    assert store.get_candidate(invalid.candidate_id) is None


def test_init_preserves_existing_experiment_files(tmp_path):
    from kernaut.cli import main

    target = tmp_path / "context.md"
    target.write_text("My existing task")
    with pytest.raises(FileExistsError):
        main(["init", str(tmp_path)])
    assert target.read_text() == "My existing task"
    assert not (tmp_path / "campaign.toml").exists()
