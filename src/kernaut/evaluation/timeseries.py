"""Greenhouse-gas forecasting benchmark over bundled NOAA GML monthly means.

Each family is one gas record (global monthly mean CO2, CH4, N2O, SF6). An episode
standardizes a training window of monthly values and forecasts a fixed horizon of
held-out months inside the normalized record span, so every input lives in [0, 1].
The predictive fitness is negative mean held-out CRPS, reusing the meta-BO scoring
helpers so both benchmarks remain directly comparable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from importlib import resources
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
)
from kernaut.evaluation.novelty import FunctionalNoveltyEvaluator
from kernaut.execution import KernelExecutor
from kernaut.models import (
    CandidateBundle,
    CandidateOrigin,
    ContractKind,
    EvaluationRecord,
)

GAS_DATA_DIR = "data/greenhouse"
SERIES_FILES = {"co2": "co2.csv", "ch4": "ch4.csv", "n2o": "n2o.csv", "sf6": "sf6.csv"}
TRAIN_GASES: tuple[str, ...] = ("co2", "ch4", "n2o")
TEST_GASES: tuple[str, ...] = ("sf6",)
_SEED_OFFSETS = {"train": 0, "validation": 10_000, "test": 20_000}
_NOAA_ATTRIBUTION = (
    "Monthly global means from NOAA Global Monitoring Laboratory trends data "
    "(gml.noaa.gov/ccgg/trends/); missing months removed."
)


@dataclass(frozen=True)
class GasSeries:
    """A bundled monthly greenhouse-gas record indexed by decimal year."""

    name: str
    time_years: NDArray[np.float64]
    value: NDArray[np.float64]


def load_gas_series(gas: str) -> GasSeries:
    """Load one bundled series; values are unmodified NOAA monthly means."""
    if gas not in SERIES_FILES:
        raise ValueError(f"unknown gas {gas!r}; expected one of {sorted(SERIES_FILES)}")
    text = resources.files("kernaut").joinpath(GAS_DATA_DIR, f"{gas}.csv").read_text()
    rows: list[tuple[float, float]] = []
    for line in text.splitlines():
        parts = line.split(",")
        if len(parts) != 2:
            continue
        rows.append((float(parts[0]), float(parts[1])))
    if len(rows) < 24:
        raise ValueError(f"bundled series {gas!r} is unexpectedly short")
    array = np.asarray(rows, dtype=np.float64)
    return GasSeries(gas, array[:, 0].copy(), array[:, 1].copy())


@dataclass(frozen=True)
class ForecastEpisode:
    """One deterministic train/forecast split of a single gas record."""

    gas: str
    seed: int
    train_months: int
    horizon_months: int
    flipped: bool
    observation_noise: float
    attribution: str
    train_inputs: NDArray[np.float64]
    train_targets: NDArray[np.float64]
    test_inputs: NDArray[np.float64]
    test_targets: NDArray[np.float64]

    @property
    def name(self) -> str:
        return f"{self.gas}:seed-{self.seed}"


def forecast_episodes(
    split: MetaSplit,
    episodes_per_family: int = 3,
    *,
    horizon_months: int = 48,
    min_train_months: int = 216,
) -> list[ForecastEpisode]:
    """Build deterministic forecasting episodes for one benchmark split."""
    if episodes_per_family < 1:
        raise ValueError("episodes_per_family must be positive")
    gases = TEST_GASES if split == "test" else TRAIN_GASES
    offset = _SEED_OFFSETS[split]
    episodes: list[ForecastEpisode] = []
    for family_index, gas in enumerate(gases):
        series = load_gas_series(gas)
        total = int(series.value.size)
        if total < min_train_months + horizon_months:
            raise ValueError(f"record {gas!r} is too short for the requested protocol")
        for episode in range(episodes_per_family):
            seed = offset + family_index * 1_000 + episode
            rng = np.random.default_rng(seed)
            n_train = int(rng.integers(min_train_months, total - horizon_months + 1))
            flipped = bool(rng.integers(0, 2))
            noise_fraction = float(rng.uniform(0.0, 0.04))
            values = series.value[::-1].copy() if flipped else series.value.copy()
            inputs = np.linspace(0.0, 1.0, total, dtype=np.float64)[:, None]
            y_mean = float(values[:n_train].mean())
            y_scale = max(float(values[:n_train].std()), 1e-8)
            train_targets = (values[:n_train] - y_mean) / y_scale
            # Observation noise is added to training targets only; the held-out future stays clean.
            sigma = noise_fraction * float(np.std(train_targets))
            if sigma > 0.0:
                train_targets = train_targets + rng.normal(0.0, sigma, size=n_train)
            episodes.append(
                ForecastEpisode(
                    gas=gas,
                    seed=seed,
                    train_months=n_train,
                    horizon_months=horizon_months,
                    flipped=flipped,
                    observation_noise=noise_fraction,
                    attribution=_NOAA_ATTRIBUTION,
                    train_inputs=inputs[:n_train],
                    train_targets=train_targets,
                    test_inputs=inputs[n_train : n_train + horizon_months],
                    test_targets=(values[n_train : n_train + horizon_months] - y_mean) / y_scale,
                )
            )
    return episodes


class GreenhouseKernelEvaluator:
    """Cross-series predictive fitness used during kernel evolution.

    Mirrors :class:`MetaKernelEvaluator`: fits only an outer covariance amplitude and
    diagonal noise on every episode, scores held-out CRPS against the clean future of
    the record, and returns negative mean CRPS so higher scores are better.
    """

    def __init__(
        self,
        executor: KernelExecutor,
        *,
        split: MetaSplit = "train",
        episodes_per_family: int = 3,
        horizon_months: int = 48,
        min_train_months: int = 216,
        amplitude_grid: tuple[float, ...] = (0.1, 0.3, 1.0, 3.0),
        noise_grid: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1),
        failure_penalty: float = 2.0,
        novelty_evaluator: FunctionalNoveltyEvaluator | None = None,
        novelty_failure_penalty: float = 0.1,
    ) -> None:
        self.executor = executor
        self.split = split
        self.episodes = forecast_episodes(
            split,
            episodes_per_family,
            horizon_months=horizon_months,
            min_train_months=min_train_months,
        )
        self.amplitude_grid = amplitude_grid
        self.noise_grid = noise_grid
        self.failure_penalty = failure_penalty
        self.novelty_evaluator = novelty_evaluator
        self.novelty_failure_penalty = novelty_failure_penalty

    def evaluate(self, candidate: CandidateBundle, _dataset: object = None) -> EvaluationRecord:
        started = time.monotonic()
        results: list[PredictiveTaskResult] = []
        failures: list[dict[str, str]] = []
        for episode in self.episodes:
            try:
                results.append(self._evaluate_episode(candidate, episode))
            except Exception as error:  # Candidate failures are part of benchmark fitness.
                failures.append({"task": episode.name, "error": f"{type(error).__name__}: {error}"})

        task_count = len(self.episodes)
        aggregate_crps = (
            sum(result.crps for result in results) + self.failure_penalty * len(failures)
        ) / task_count
        mean_nll = (
            sum(result.train_nll for result in results) + self.failure_penalty * len(failures)
        ) / task_count
        max_condition = max((result.condition_number for result in results), default=float("inf"))
        mean_noise = float(np.mean([result.noise for result in results])) if results else 1.0
        score = -aggregate_crps
        novelty_metadata: dict[str, object] = {}
        if self.novelty_evaluator is not None and candidate.origin == CandidateOrigin.DISCOVERED:
            novelty = self.novelty_evaluator.evaluate(candidate)
            novelty_shortfall = max(
                0.0,
                (self.novelty_evaluator.minimum_distance - novelty.distance)
                / self.novelty_evaluator.minimum_distance,
            )
            novelty_penalty = self.novelty_failure_penalty * novelty_shortfall
            novelty_metadata = {
                "functional_novelty": novelty.distance,
                "novelty_threshold": self.novelty_evaluator.minimum_distance,
                "novelty_passed": novelty.passed,
                "nearest_reference_id": novelty.nearest_candidate_id,
                "nearest_reference_name": novelty.nearest_name,
                "skipped_novelty_references": list(novelty.skipped_candidate_ids),
                "novelty_penalty": novelty_penalty,
            }
            score -= novelty_penalty
        return EvaluationRecord(
            candidate_id=candidate.candidate_id,
            score=score,
            negative_log_likelihood=mean_nll,
            runtime_seconds=time.monotonic() - started,
            jitter=mean_noise,
            condition_number=max_condition,
            metadata={
                "objective": "negative_mean_heldout_crps",
                "benchmark": "greenhouse_forecasting",
                "split": self.split,
                "task_count": task_count,
                "successful_tasks": len(results),
                "failed_tasks": len(failures),
                "mean_crps": aggregate_crps,
                "mean_nlpd": _mean_or_penalty(results, "nlpd", self.failure_penalty, task_count),
                "mean_rmse": _mean_or_penalty(results, "rmse", self.failure_penalty, task_count),
                "tasks": [result.__dict__ for result in results],
                "failures": failures,
                **novelty_metadata,
            },
        )

    def _evaluate_episode(
        self, candidate: CandidateBundle, episode: ForecastEpisode
    ) -> PredictiveTaskResult:
        started = time.monotonic()
        n_train = episode.train_inputs.shape[0]
        gram = self.executor.gram(
            candidate, np.vstack([episode.train_inputs, episode.test_inputs])
        ).as_array()
        fit = _fit_gp_scale(
            gram[:n_train, :n_train], episode.train_targets, self.amplitude_grid, self.noise_grid
        )
        mean, variance = _gp_predict(
            gram, n_train, episode.train_targets, amplitude=fit[1], noise=fit[2]
        )
        crps = _mean_crps(mean, variance, episode.test_targets)
        nlpd = float(
            np.mean(
                0.5 * np.log(2 * np.pi * variance)
                + 0.5 * (episode.test_targets - mean) ** 2 / variance
            )
        )
        rmse = float(np.sqrt(np.mean((episode.test_targets - mean) ** 2)))
        return PredictiveTaskResult(
            task=episode.name,
            crps=crps,
            nlpd=nlpd,
            rmse=rmse,
            train_nll=fit[0],
            amplitude=fit[1],
            noise=fit[2],
            condition_number=fit[3],
            runtime_seconds=time.monotonic() - started,
        )


def _gas_spectral_mixture_tree(rng: np.random.Generator) -> dict[str, Any]:
    """One seeded multi-start whose frequencies cover trend plus the annual band."""
    return {
        "op": "base",
        "kind": "spectral_mixture",
        "variance": 1.0,
        "weights": rng.uniform(0.5, 2.0, size=4).tolist(),
        # Frequencies are cycles per normalized span; records span roughly 25-47 years,
        # so the annual cycle sits near frequency span_years and its harmonic near twice that.
        "means": [
            rng.uniform(0.0, 3.0),
            rng.uniform(8.0, 16.0),
            rng.uniform(24.0, 48.0),
            rng.uniform(48.0, 80.0),
        ],
        "scales": rng.uniform(0.5, 4.0, size=4).tolist(),
    }


def greenhouse_baseline_candidates() -> list[CandidateBundle]:
    """Fixed closure-tree references with forecasting-appropriate grids."""
    source = 'def closure_tree(parameters):\n    return parameters["tree"]\n'
    rationale = "Fixed normalized-time reference for the greenhouse-gas benchmark."
    candidates: list[CandidateBundle] = []
    for kind in ("rbf", "matern32", "matern52"):
        for lengthscale in (0.05, 0.1, 0.25, 0.5, 1.0):
            candidates.append(
                CandidateBundle(
                    name=f"gas_{kind}_ls{str(lengthscale).replace('.', '_')}",
                    contract=ContractKind.CLOSURE,
                    origin=CandidateOrigin.BASELINE,
                    source=source,
                    parameters={
                        "tree": {
                            "op": "base",
                            "kind": kind,
                            "variance": 1.0,
                            "lengthscale": lengthscale,
                        }
                    },
                    rationale=rationale,
                )
            )
    candidates.append(
        CandidateBundle(
            name="gas_linear",
            contract=ContractKind.CLOSURE,
            origin=CandidateOrigin.BASELINE,
            source=source,
            parameters={"tree": {"op": "base", "kind": "linear", "variance": 1.0}},
            rationale=rationale,
        )
    )
    for lengthscale in (0.1, 0.5):
        for alpha in (1.0, 4.0):
            candidates.append(
                CandidateBundle(
                    name=(
                        f"gas_rq_ls{str(lengthscale).replace('.', '_')}"
                        f"_a{str(alpha).replace('.', '_')}"
                    ),
                    contract=ContractKind.CLOSURE,
                    origin=CandidateOrigin.BASELINE,
                    source=source,
                    parameters={
                        "tree": {
                            "op": "base",
                            "kind": "rq",
                            "variance": 1.0,
                            "lengthscale": lengthscale,
                            "alpha": alpha,
                        }
                    },
                    rationale=rationale,
                )
            )
    for period in (0.02, 0.022, 0.024, 0.03, 0.036, 0.04, 0.05):
        for lengthscale in (0.005, 0.01, 0.05):
            candidates.append(
                CandidateBundle(
                    name=(
                        f"gas_periodic_p{str(period).replace('.', '_')}"
                        f"_ls{str(lengthscale).replace('.', '_')}"
                    ),
                    contract=ContractKind.CLOSURE,
                    origin=CandidateOrigin.BASELINE,
                    source=source,
                    parameters={
                        "tree": {
                            "op": "base",
                            "kind": "periodic",
                            "variance": 1.0,
                            "lengthscale": lengthscale,
                            "period": period,
                        }
                    },
                    rationale=(
                        "Annual-cycle grid: records span 25-47 years, so yearly periodicity "
                        "lies near periods 0.021-0.040 in normalized time."
                    ),
                )
            )
    ard_patterns: tuple[tuple[str, dict[str, Any]], ...] = (
        ("uniform0_1", {"lengthscales": 0.1}),
        ("uniform1", {"lengthscales": 1.0}),
        ("ramp_up", {"lengthscales": {"mode": "geometric_ramp", "min": 0.25, "max": 4.0}}),
        ("ramp_down", {"lengthscales": {"mode": "geometric_ramp", "min": 4.0, "max": 0.25}}),
    )
    for kind in ("ard_rbf", "ard_matern32", "ard_matern52"):
        for label, pattern in ard_patterns:
            candidates.append(
                CandidateBundle(
                    name=f"gas_{kind}_{label}",
                    contract=ContractKind.CLOSURE,
                    origin=CandidateOrigin.BASELINE,
                    source=source,
                    parameters={"tree": {"op": "base", "kind": kind, "variance": 1.0, **pattern}},
                    rationale=rationale,
                )
            )
    rng = np.random.default_rng(2026_08_22)
    for start in range(8):
        candidates.append(
            CandidateBundle(
                name=f"gas_sm_start{start:02d}",
                contract=ContractKind.CLOSURE,
                origin=CandidateOrigin.BASELINE,
                source=source,
                parameters={"tree": _gas_spectral_mixture_tree(rng)},
                rationale="Seeded spectral-mixture start covering trend and annual frequencies.",
            )
        )
    return candidates


__all__ = [
    "ForecastEpisode",
    "GasSeries",
    "GreenhouseKernelEvaluator",
    "TEST_GASES",
    "TRAIN_GASES",
    "forecast_episodes",
    "greenhouse_baseline_candidates",
    "load_gas_series",
]
