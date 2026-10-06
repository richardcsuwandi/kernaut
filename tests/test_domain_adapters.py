"""Small offline checks for optional benchmark adapters, using synthetic observations."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from kernaut.evaluation.chembench import ChemBenchKernelEvaluator, chembench_episodes
from kernaut.evaluation.glucose import GLUCOSE_SPLITS, GlucoseKernelEvaluator, gp_forecast
from kernaut.execution import SubprocessExecutor
from kernaut.models import CandidateBundle


def reference():
    return CandidateBundle(
        name="reference",
        contract="closure",
        source="def closure_tree(parameters): return parameters['tree']",
        parameters={"tree": {"op": "base", "kind": "rbf"}},
    )


class SyntheticOracle:
    def __init__(self, domain, **kwargs):
        self.domain = domain

    def run(self, params):
        return SimpleNamespace(measurement=np.log(params["C_A"]) + params["Enz"] ** 0.5)


def test_chemistry_adapter_is_deterministic_and_scores_without_external_package(tmp_path):
    kwargs = dict(train_points=5, test_points=3, oracle_factory=SyntheticOracle)
    train = chembench_episodes(tmp_path, "train", **kwargs)
    replay = chembench_episodes(tmp_path, "train", **kwargs)
    validation = chembench_episodes(tmp_path, "validation", **kwargs)
    heldout = chembench_episodes(tmp_path, "test", **kwargs)
    assert {e.domain for e in train}.isdisjoint(e.domain for e in heldout)
    for first, again, other in zip(train, replay, validation, strict=True):
        np.testing.assert_array_equal(first.train_targets, again.train_targets)
        assert first.seed != other.seed
        assert first.train_inputs.shape == (5, 7)
        assert first.test_inputs.shape == (3, 7)
        assert abs(first.train_targets.mean()) < 1e-10
    evaluator = ChemBenchKernelEvaluator(
        SubprocessExecutor(),
        tmp_path,
        episodes=train[:1],
        amplitude_grid=(1.0,),
        noise_grid=(0.01,),
    )
    result = evaluator.evaluate(reference())
    assert result.metadata["successful_tasks"] == 1
    assert result.metadata["failed_tasks"] == 0
    assert np.isfinite(result.score)
    assert result.score == -result.metadata["mean_crps"]


@pytest.fixture
def glucose_data(tmp_path):
    episodes = []
    for j in range(3):
        episodes.append(
            {
                "times_min": [0, 15, 30, 45, 60, 75],
                "cgm_mg_dl": [100 + j, 101 + j, 103 + j, 106 + j, 109 + j, 107 + j],
                "minute_inputs": [[0, 0, 0, 0], [15, 10 * j, 0, 0.1 * j]],
            }
        )
    payload = {p: {"training": episodes} for p in GLUCOSE_SPLITS["train"]}
    path = tmp_path / "training.json"
    path.write_text(json.dumps(payload))
    return path


def test_glucose_adapter_scores_all_folds_from_training_export(glucose_data):
    evaluator = GlucoseKernelEvaluator(
        SubprocessExecutor(), glucose_data, amplitude_grid=(1.0,), noise_grid=(0.01,)
    )
    result = evaluator.evaluate(reference())
    assert result.metadata["successful_tasks"] == 30
    assert result.metadata["failed_tasks"] == 0
    assert np.isfinite(result.metadata["geomean_cgm_nmse"])
    assert result.score == -result.metadata["mean_crps"]


def test_glucose_forecast_does_not_use_query_targets():
    x = np.arange(6, dtype=float)
    gram = np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2)
    y = np.array([100.0, 101.0, 103.0, 107.0, 108.0, 109.0])
    fit, condition, query = np.array([0, 1, 2]), np.array([0, 1, 2, 3]), np.array([4, 5])
    mean, variance, params = gp_forecast(gram, y, fit, condition, query, (1.0,), (0.01,))
    changed = y.copy()
    changed[query] = [-1e6, 1e6]
    other_mean, other_variance, other_params = gp_forecast(
        gram, changed, fit, condition, query, (1.0,), (0.01,)
    )
    np.testing.assert_array_equal(mean, other_mean)
    np.testing.assert_array_equal(variance, other_variance)
    assert params == other_params


@pytest.mark.parametrize(
    "command",
    [
        "meta-run",
        "meta-benchmark",
        "ts-run",
        "ts-benchmark",
        "chem-run",
        "chem-benchmark",
        "glucose-run",
        "glucose-benchmark",
    ],
)
def test_domain_commands_complete_offline(command, tmp_path, monkeypatch, capsys, glucose_data):
    from kernaut import cli
    from kernaut.archive import CandidateStore
    from kernaut.llm.base import AssistantReply, LanguageModel
    from kernaut.models import CandidateOrigin

    class QuietModel(LanguageModel):
        def complete(self, messages, tools, system_prompt):
            return AssistantReply(content="Offline command check complete.")

    candidate = reference().model_copy(update={"origin": CandidateOrigin.BASELINE})
    for name in (
        "meta_baseline_candidates",
        "greenhouse_baseline_candidates",
        "chembench_baseline_candidates",
        "glucose_baseline_candidates",
    ):
        monkeypatch.setattr(cli, name, lambda: [candidate])
    monkeypatch.setattr(cli, "build_model", lambda config: QuietModel())
    # Use the real evaluators with one amplitude/noise pair to keep this command test small.
    for name in (
        "MetaKernelEvaluator",
        "MetaBOBenchmark",
        "GreenhouseKernelEvaluator",
        "ChemBenchKernelEvaluator",
        "GlucoseKernelEvaluator",
    ):
        original = getattr(cli, name)

        def small_evaluator(*args, _original=original, **kwargs):
            kwargs.update(amplitude_grid=(1.0,), noise_grid=(0.01,))
            if _original is ChemBenchKernelEvaluator:
                kwargs["oracle_factory"] = SyntheticOracle
            return _original(*args, **kwargs)

        monkeypatch.setattr(cli, name, small_evaluator)
    config = tmp_path / "config.toml"
    config.write_text('[llm]\nprovider="openai"\nmodel="offline"\n[campaign]\nmax_rounds=1\n')
    context = tmp_path / "context.md"
    context.write_text("Find a kernel for these training data.")
    archive = tmp_path / (command + ".sqlite")
    args = [command, "--archive", str(archive)]
    if command.endswith("-run"):
        args += ["--config", str(config), "--context", str(context)]
    else:
        args += ["--split", "validation"]
    if command.startswith(("meta-", "ts-")):
        args += ["--episodes-per-family", "1"]
    if command == "meta-benchmark":
        args += ["--bo-steps", "1", "--candidate-pool-size", "8"]
    if command.startswith("chem-"):
        args += ["--chembench-root", str(tmp_path), "--train-points", "5", "--test-points", "3"]
    if command.startswith("glucose-"):
        # Use the same synthetic observations for each age group in this routing test.
        payload = json.loads(glucose_data.read_text())
        row = next(iter(payload.values()))
        payload.update({p: row for p in GLUCOSE_SPLITS["validation"]})
        glucose_data.write_text(json.dumps(payload))
        args += ["--data", str(glucose_data)]
    assert cli.main(args) == 0
    report = json.loads(capsys.readouterr().out)
    if command.endswith("-run"):
        assert report["final_message"] == "Offline command check complete."
        assert CandidateStore(archive).recent()[0]["accepted"]
    else:
        assert len(report) == 1
        result = report[0]["splits"]["validation"]
        if "predictive" in result:
            assert result["bo"]["failed_tasks"] == 0
            result = result["predictive"]
        assert result["failed_tasks"] == 0
        assert result["successful_tasks"] > 0
