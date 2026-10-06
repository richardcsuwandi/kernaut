from __future__ import annotations

import math

import numpy as np

from kernaut.evaluation import (
    TEST_TASKS,
    TRAIN_TASKS,
    MetaBOBenchmark,
    MetaKernelEvaluator,
    MetaSearchEvaluator,
    meta_baseline_candidates,
    procedural_tasks,
)
from kernaut.evaluation.novelty import FunctionalNoveltyEvaluator
from kernaut.execution import SubprocessExecutor
from kernaut.models import CandidateBundle


def test_meta_splits_are_deterministic_and_family_disjoint() -> None:
    train = procedural_tasks("train", episodes_per_family=2)
    validation = procedural_tasks("validation", episodes_per_family=2)
    test = procedural_tasks("test", episodes_per_family=1)

    assert train == procedural_tasks("train", episodes_per_family=2)
    assert {task.spec.name for task in train} == {task.name for task in TRAIN_TASKS}
    assert {task.spec.name for task in test} == {task.name for task in TEST_TASKS}
    assert {task.spec.name for task in train}.isdisjoint(task.spec.name for task in test)
    assert {task.seed for task in train}.isdisjoint(task.seed for task in validation)


def test_procedural_tasks_are_finite_and_respect_known_maxima() -> None:
    for task in procedural_tasks("train", 1) + procedural_tasks("test", 1):
        rng = np.random.default_rng(task.seed)
        values = task.evaluate(rng.uniform(size=(16, task.dimension)))
        assert values.shape == (16,)
        assert np.all(np.isfinite(values))
        assert float(values.max()) <= task.maximum + 1e-7


def test_meta_predictive_and_bo_evaluators_return_finite_scores() -> None:
    candidate = next(
        candidate for candidate in meta_baseline_candidates() if candidate.name == "meta_rbf_ls0_5"
    )
    executor = SubprocessExecutor()
    predictive = MetaKernelEvaluator(
        executor,
        split="train",
        episodes_per_family=1,
        train_points_base=2,
        test_points=8,
        amplitude_grid=(1.0,),
        noise_grid=(1e-2,),
    ).evaluate(candidate)

    assert math.isfinite(predictive.score)
    assert predictive.metadata["successful_tasks"] == len(TRAIN_TASKS)
    assert predictive.metadata["failed_tasks"] == 0

    bo = MetaBOBenchmark(
        executor,
        episodes_per_family=1,
        bo_steps=1,
        candidate_pool_size=8,
        amplitude_grid=(1.0,),
        noise_grid=(1e-2,),
    ).evaluate(candidate, "test")

    assert math.isfinite(bo["regret_auc"])
    assert bo["successful_tasks"] == len(TEST_TASKS)
    assert bo["failed_tasks"] == 0


def test_novelty_shortfall_uses_soft_penalty() -> None:
    executor = SubprocessExecutor()
    identity = CandidateBundle(
        name="identity_warp",
        contract="input_transform",
        source="def transform_point(x, parameters):\n    return x\n",
        parameters={"base_kernel": {"kind": "matern52", "lengthscale": 0.5}},
    )
    evaluator = MetaKernelEvaluator(
        executor,
        episodes_per_family=1,
        train_points_base=2,
        test_points=4,
        amplitude_grid=(1.0,),
        noise_grid=(1e-2,),
        novelty_evaluator=FunctionalNoveltyEvaluator(
            executor, probe_dimensions=(2,), probe_points=6
        ),
    )
    result = evaluator.evaluate(identity)
    assert not result.metadata["novelty_passed"]
    assert 0.0 < result.metadata["novelty_penalty"] <= 0.1
    assert math.isclose(
        result.score,
        -result.metadata["mean_crps"] - result.metadata["novelty_penalty"],
    )


def test_full_search_evaluator_adds_training_bo_metrics() -> None:
    executor = SubprocessExecutor()
    candidate = next(
        candidate for candidate in meta_baseline_candidates() if candidate.name == "meta_rbf_ls0_5"
    )
    evaluator = MetaSearchEvaluator(
        MetaKernelEvaluator(
            executor,
            episodes_per_family=1,
            train_points_base=2,
            test_points=4,
            amplitude_grid=(1.0,),
            noise_grid=(1e-2,),
        ),
        MetaBOBenchmark(
            executor,
            episodes_per_family=1,
            bo_steps=1,
            candidate_pool_size=8,
            amplitude_grid=(1.0,),
            noise_grid=(1e-2,),
        ),
    )
    result = evaluator.evaluate(candidate)
    assert result.metadata["bo"]["split"] == "train"
    assert math.isfinite(result.metadata["bo"]["regret_auc"])
