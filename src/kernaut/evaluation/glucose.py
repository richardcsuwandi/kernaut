"""GlucoseBench adapter: CGM forecasting of meal and insulin interventions.

GlucoseBench (a modification of the simglucose UVA/Padova type-1 diabetes model)
asks a learner to forecast one simulated patient's CGM response to a new meal/bolus
intervention from that patient's six training episodes and a 45-minute prefix of the query.

A GP input is one CGM reading, encoded as five coordinates in [0, 1]: reading time and the
episode's delivered intervention (meal grams, meal start, bolus units, bolus start). The
kernel therefore decides how readings share strength across time and across interventions.
Search and validation never touch sealed test outcomes: each patient contributes
leave-one-training-episode-out folds, so every held-out target is a public training episode.
Kernels are searched on children and validated on adolescents. Sealed adult testing requires
the external GlucoseBench evaluator and is separate from this training-data adapter.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from kernaut.evaluation.meta_bo import (
    MetaSplit,
    PredictiveTaskResult,
    _fit_gp_scale,
    _gp_predict,
    _mean_crps,
    _mean_or_penalty,
    meta_baseline_candidates,
)
from kernaut.evaluation.novelty import FunctionalNoveltyEvaluator
from kernaut.execution import KernelExecutor
from kernaut.models import CandidateBundle, CandidateOrigin, EvaluationRecord

GLUCOSE_FEATURES: tuple[str, ...] = ("time", "meal_g", "meal_minute", "bolus_U", "bolus_minute")
GLUCOSE_INPUT_DIMENSION = len(GLUCOSE_FEATURES)
EPISODE_MINUTES = 360.0
MAX_MEAL_G = 60.0
MAX_BOLUS_U = 1.5
FORECAST_CUT_MIN = 45
GLUCOSE_SPLITS: dict[str, tuple[str, ...]] = {
    "train": tuple(f"child#{i:03d}" for i in range(1, 11)),
    "validation": tuple(f"adolescent#{i:03d}" for i in range(1, 11)),
    "test": tuple(f"adult#{i:03d}" for i in range(1, 11)),
}


def intervention(episode: dict[str, Any]) -> tuple[float, float, float, float]:
    """Delivered meal grams, meal start, bolus units, bolus start from the public input log.

    With no bolus the start time is irrelevant, so it is set to the meal start rather than
    left arbitrary.
    """
    inputs = np.asarray(episode["minute_inputs"], dtype=np.float64)
    minutes, meal, bolus = inputs[:, 0], inputs[:, 1], inputs[:, 3]
    meal_g, bolus_u = float(meal.sum()), float(bolus.sum())
    meal_start = float(minutes[meal > 0][0]) if meal_g > 0 else 0.0
    bolus_start = float(minutes[bolus > 0][0]) if bolus_u > 0 else meal_start
    return meal_g, meal_start, bolus_u, bolus_start


def encode(episode: dict[str, Any], times: NDArray[np.float64]) -> NDArray[np.float64]:
    meal_g, meal_start, bolus_u, bolus_start = intervention(episode)
    row = [
        meal_g / MAX_MEAL_G,
        meal_start / EPISODE_MINUTES,
        bolus_u / MAX_BOLUS_U,
        bolus_start / EPISODE_MINUTES,
    ]
    times = np.asarray(times, dtype=np.float64)
    return np.column_stack([times / EPISODE_MINUTES, np.tile(row, (times.size, 1))])


def passive_scale(training: list[dict[str, Any]]) -> float:
    """GlucoseBench CGM normalizer: variance over the two passive episodes, floored at 1."""
    values = np.concatenate([np.asarray(e["cgm_mg_dl"], dtype=np.float64) for e in training[:2]])
    return max(float(values.var()), 1.0)


def gp_forecast(
    gram: NDArray[np.float64],
    y: NDArray[np.float64],
    fit_index: NDArray[np.intp],
    condition_index: NDArray[np.intp],
    query_index: NDArray[np.intp],
    amplitude_grid: tuple[float, ...],
    noise_grid: tuple[float, ...],
) -> tuple[NDArray[np.float64], NDArray[np.float64], tuple[float, float, float, float]]:
    """Fit amplitude/noise on full training episodes only, then condition on them plus the prefix.

    Targets are standardized by the fitting episodes, so the query prefix updates the
    predictive state without tuning anything, as the GlucoseBench protocol requires.
    Returns mean and variance on the original scale plus the fit tuple.
    """
    center, spread = float(y[fit_index].mean()), max(float(y[fit_index].std()), 1e-6)
    z = (y - center) / spread
    fit = _fit_gp_scale(
        gram[np.ix_(fit_index, fit_index)], z[fit_index], amplitude_grid, noise_grid
    )
    order = np.concatenate([condition_index, query_index])
    mean, variance = _gp_predict(
        gram[np.ix_(order, order)],
        condition_index.size,
        z[condition_index],
        amplitude=fit[1],
        noise=fit[2],
    )
    return mean * spread + center, variance * spread**2, fit


def load_training(path: Path) -> dict[str, list[dict[str, Any]]]:
    return {patient: row["training"] for patient, row in json.loads(Path(path).read_text()).items()}


class GlucoseKernelEvaluator:
    """Leave-one-training-episode-out CGM forecasting across GlucoseBench patients."""

    def __init__(
        self,
        executor: KernelExecutor,
        data_path: Path,
        *,
        split: MetaSplit = "train",
        amplitude_grid: tuple[float, ...] = (0.1, 0.3, 1.0, 3.0),
        noise_grid: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1),
        failure_penalty: float = 2.0,
        novelty_evaluator: FunctionalNoveltyEvaluator | None = None,
        novelty_failure_penalty: float = 0.1,
    ) -> None:
        self.executor = executor
        self.split = split
        training = load_training(data_path)
        self.patients = {p: training[p] for p in GLUCOSE_SPLITS[split]}
        self.amplitude_grid = amplitude_grid
        self.noise_grid = noise_grid
        self.failure_penalty = failure_penalty
        self.novelty_evaluator = novelty_evaluator
        self.novelty_failure_penalty = novelty_failure_penalty
        self.task_count = sum(len(episodes) for episodes in self.patients.values())

    def evaluate(self, candidate: CandidateBundle, _dataset: object = None) -> EvaluationRecord:
        started = time.monotonic()
        results: list[PredictiveTaskResult] = []
        nmse: list[float] = []
        failures: list[dict[str, str]] = []
        for patient, episodes in self.patients.items():
            try:
                rows = self._evaluate_patient(candidate, patient, episodes)
            except Exception as error:  # A kernel failure costs every fold of that patient.
                failures += [
                    {"task": f"{patient}:loto-{j}", "error": f"{type(error).__name__}: {error}"}
                    for j in range(len(episodes))
                ]
                continue
            for result, value in rows:
                results.append(result)
                nmse.append(value)

        count = self.task_count
        aggregate_crps = (
            sum(r.crps for r in results) + self.failure_penalty * len(failures)
        ) / count
        mean_nll = (
            sum(r.train_nll for r in results) + self.failure_penalty * len(failures)
        ) / count
        score = -aggregate_crps
        novelty_metadata: dict[str, object] = {}
        if self.novelty_evaluator is not None and candidate.origin == CandidateOrigin.DISCOVERED:
            novelty = self.novelty_evaluator.evaluate(candidate)
            shortfall = max(
                0.0,
                (self.novelty_evaluator.minimum_distance - novelty.distance)
                / self.novelty_evaluator.minimum_distance,
            )
            penalty = self.novelty_failure_penalty * shortfall
            score -= penalty
            novelty_metadata = {
                "functional_novelty": novelty.distance,
                "novelty_threshold": self.novelty_evaluator.minimum_distance,
                "novelty_passed": novelty.passed,
                "nearest_reference_id": novelty.nearest_candidate_id,
                "nearest_reference_name": novelty.nearest_name,
                "novelty_penalty": penalty,
            }
        return EvaluationRecord(
            candidate_id=candidate.candidate_id,
            score=score,
            negative_log_likelihood=mean_nll,
            runtime_seconds=time.monotonic() - started,
            jitter=float(np.mean([r.noise for r in results])) if results else 1.0,
            condition_number=max((r.condition_number for r in results), default=float("inf")),
            metadata={
                "objective": "negative_mean_heldout_crps",
                "benchmark": "glucosebench_loto_cgm",
                "split": self.split,
                "input_features": list(GLUCOSE_FEATURES),
                "forecast_cut_min": FORECAST_CUT_MIN,
                "task_count": count,
                "successful_tasks": len(results),
                "failed_tasks": len(failures),
                "mean_crps": aggregate_crps,
                # GlucoseBench aggregation: arithmetic within patient, geometric across patients.
                "geomean_cgm_nmse": _patient_geomean(results, nmse),
                "mean_nlpd": _mean_or_penalty(results, "nlpd", self.failure_penalty, count),
                "mean_rmse": _mean_or_penalty(results, "rmse", self.failure_penalty, count),
                "tasks": [r.__dict__ for r in results],
                "failures": failures,
                **novelty_metadata,
            },
        )

    def _evaluate_patient(
        self, candidate: CandidateBundle, patient: str, episodes: list[dict[str, Any]]
    ) -> list[tuple[PredictiveTaskResult, float]]:
        blocks = [encode(e, np.asarray(e["times_min"])) for e in episodes]
        y = np.concatenate([np.asarray(e["cgm_mg_dl"], dtype=np.float64) for e in episodes])
        offsets = np.cumsum([0] + [b.shape[0] for b in blocks])
        gram = self.executor.gram(candidate, np.vstack(blocks)).as_array()
        scale = passive_scale(episodes)
        out = []
        for j in range(len(episodes)):
            started = time.monotonic()
            times = np.asarray(episodes[j]["times_min"])
            own = np.arange(offsets[j], offsets[j + 1])
            prefix, suffix = own[times <= FORECAST_CUT_MIN], own[times > FORECAST_CUT_MIN]
            others = np.concatenate(
                [np.arange(offsets[k], offsets[k + 1]) for k in range(len(episodes)) if k != j]
            )
            mean, variance, fit = gp_forecast(
                gram,
                y,
                others,
                np.concatenate([others, prefix]),
                suffix,
                self.amplitude_grid,
                self.noise_grid,
            )
            spread = max(float(y[others].std()), 1e-6)
            target = y[suffix]
            crps = _mean_crps(mean / spread, variance / spread**2, target / spread)
            nlpd = float(
                np.mean(
                    0.5 * np.log(2 * math.pi * variance) + 0.5 * (target - mean) ** 2 / variance
                )
            )
            rmse = float(np.sqrt(np.mean((target - mean) ** 2)))
            out.append(
                (
                    PredictiveTaskResult(
                        task=f"{patient}:loto-{j}",
                        crps=crps,
                        nlpd=nlpd,
                        rmse=rmse,
                        train_nll=fit[0],
                        noise=fit[2],
                        amplitude=fit[1],
                        condition_number=fit[3],
                        runtime_seconds=time.monotonic() - started,
                    ),
                    rmse**2 / scale,
                )
            )
        return out


def _patient_geomean(results: list[PredictiveTaskResult], nmse: list[float]) -> float | None:
    if not results:
        return None
    by_patient: dict[str, list[float]] = {}
    for result, value in zip(results, nmse, strict=False):
        by_patient.setdefault(result.task.split(":")[0], []).append(value)
    return float(np.exp(np.mean([np.log(np.mean(v)) for v in by_patient.values()])))


def glucose_baseline_candidates() -> list[CandidateBundle]:
    """Normalized-input reference kernels for the GlucoseBench adapter."""
    prefix, rationale = "glucose_", "Fixed normalized-input reference for GlucoseBench."
    return [
        candidate.model_copy(
            update={"name": candidate.name.replace("meta_", prefix, 1), "rationale": rationale}
        )
        for candidate in meta_baseline_candidates()
    ]
