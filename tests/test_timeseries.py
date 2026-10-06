from __future__ import annotations

import math

import numpy as np

from kernaut.config import ExecutionConfig
from kernaut.evaluation import (
    TEST_GASES,
    TRAIN_GASES,
    GreenhouseKernelEvaluator,
    forecast_episodes,
    greenhouse_baseline_candidates,
    load_gas_series,
)
from kernaut.execution import SubprocessExecutor


def test_gas_series_load_from_bundled_records() -> None:
    for gas in (*TRAIN_GASES, *TEST_GASES):
        series = load_gas_series(gas)
        assert series.name == gas
        assert series.value.size > 200
        assert np.all(np.diff(series.time_years) > 0)
        assert np.all(np.isfinite(series.value))


def test_forecast_splits_are_deterministic_and_family_disjoint() -> None:
    train = forecast_episodes("train", episodes_per_family=2)
    validation = forecast_episodes("validation", episodes_per_family=2)
    test = forecast_episodes("test", episodes_per_family=1)

    replay = forecast_episodes("train", episodes_per_family=2)
    for original, repeated in zip(train, replay, strict=True):
        assert (original.gas, original.seed) == (repeated.gas, repeated.seed)
        assert np.array_equal(original.train_targets, repeated.train_targets)
        assert np.array_equal(original.train_inputs, repeated.train_inputs)
        assert np.array_equal(original.test_targets, repeated.test_targets)
        assert np.array_equal(original.test_inputs, repeated.test_inputs)
    assert {episode.gas for episode in train} == set(TRAIN_GASES)
    assert {episode.gas for episode in test} == set(TEST_GASES)
    assert {episode.gas for episode in train}.isdisjoint(episode.gas for episode in test)
    assert {episode.seed for episode in train}.isdisjoint(episode.seed for episode in validation)


def test_forecast_episodes_have_contiguous_standardized_windows() -> None:
    episodes = (
        forecast_episodes("train", 1)
        + forecast_episodes("validation", 1)
        + forecast_episodes("test", 1)
    )
    for episode in episodes:
        total = episode.train_inputs.shape[0] + episode.test_inputs.shape[0]
        record = load_gas_series(episode.gas)
        assert total <= record.value.size
        assert episode.train_months >= 216
        assert episode.horizon_months == 48
        # Inputs are normalized to [0, 1] and the forecast window starts exactly at the
        # end of the training window.
        assert abs(float(episode.train_inputs[0, 0]) - 0.0) < 1e-12
        assert float(episode.test_inputs[-1, 0]) <= 1.0 + 1e-12
        gap = float(episode.test_inputs[0, 0] - episode.train_inputs[-1, 0])
        step = float(episode.train_inputs[-1, 0] - episode.train_inputs[-2, 0])
        assert math.isclose(gap, step, rel_tol=1e-9)
        assert abs(float(np.std(episode.test_targets))) > 0.0


def test_greenhouse_evaluator_scores_reference_kernels() -> None:
    candidates = {candidate.name: candidate for candidate in greenhouse_baseline_candidates()}
    executor = SubprocessExecutor(ExecutionConfig(max_output_bytes=8 * 1024 * 1024))
    evaluator = GreenhouseKernelEvaluator(
        executor,
        split="train",
        episodes_per_family=1,
        horizon_months=24,
        min_train_months=240,
        amplitude_grid=(1.0,),
        noise_grid=(1e-2,),
    )
    result = evaluator.evaluate(candidates["gas_rbf_ls0_25"])

    assert math.isfinite(result.score)
    assert result.metadata["benchmark"] == "greenhouse_forecasting"
    assert result.metadata["successful_tasks"] == len(TRAIN_GASES)
    assert result.metadata["failed_tasks"] == 0
    assert math.isclose(result.score, -result.metadata["mean_crps"])


def test_greenhouse_baseline_candidates_are_unique_closure_references() -> None:
    candidates = greenhouse_baseline_candidates()
    names = [candidate.name for candidate in candidates]

    assert len(names) == len(set(names))
    kinds = {str(candidate.parameters["tree"]["kind"]) for candidate in candidates}
    assert {"rbf", "matern32", "matern52", "linear", "rq", "periodic"} <= kinds
