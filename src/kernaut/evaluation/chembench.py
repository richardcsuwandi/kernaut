"""ChemBench adapter for certified kernel discovery.

ChemBench (from the LLM-AutoSciLab release, scientific-discovery org) is an
enzyme-kinetics active-experimentation benchmark: 100 mechanism domains
(``c0``..``c99``), each a distinct rate law over 7 continuous inputs
(substrate/inhibitor/product concentrations, enzyme loading, temperature,
pH), returning a single observed reaction rate ``r0``. The adapter only ever
calls the oracle's public ``run(params) -> OracleResult`` method; it never
imports the internal rate-law functions or parameter tables.

This is a first-stage surrogate benchmark: it tests whether a frozen kernel
is a useful prior for regressing enzyme kinetics from scattered experiments,
not that the kernel itself is a recovered rate law.
"""

from __future__ import annotations

import contextlib
import io
import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
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

# The 10 base single-mechanism domains: each isolates exactly one kinetic
# effect (saturation, one inhibition type, temperature, pH, or bisubstrate).
CHEM_TRAIN_DOMAINS: tuple[str, ...] = (
    "c0_michaelis_menten",
    "c1_competitive_inhibition",
    "c2_product_inhibition",
    "c3_arrhenius_temperature",
    "c4_ph_activity",
    "c5_pingpong_bisubstrate",
    "c6_uncompetitive_inhibition",
    "c7_substrate_inhibition",
    "c8_hill_cooperativity",
    "c9_noncompetitive_inhibition",
)
# Genuinely novel single-mechanism domains, structurally distinct from every
# c0-c9 training mechanism (not a recombination of them) -- held out entirely.
CHEM_TEST_DOMAINS: tuple[str, ...] = (
    "c65_ordered_bi_bi",
    "c66_reversible_mm",
    "c67_allosteric_act",
    "c69_fractal_kinetics",
    "c73_metal_activation",
)
CHEM_INPUT_DIMENSION = 7
CHEM_INPUT_VARS: tuple[str, ...] = ("C_A", "C_I", "C_B", "C_P", "Enz", "T", "pH")
CHEM_INPUT_BOUNDS: dict[str, tuple[float, float]] = {
    "C_A": (0.01, 100.0),
    "C_I": (0.0, 50.0),
    "C_B": (0.01, 100.0),
    "C_P": (0.0, 20.0),
    "Enz": (0.01, 10.0),
    "T": (278.0, 368.0),
    "pH": (4.0, 10.0),
}
CHEM_LOG_VARS: frozenset[str] = frozenset({"C_A", "C_B", "Enz"})

OracleFactory = Callable[..., Any]


@dataclass(frozen=True)
class ChemEpisode:
    """One domain with disjoint observed and held-out experiments."""

    domain: str
    seed: int
    train_inputs: NDArray[np.float64]
    train_targets: NDArray[np.float64]
    test_inputs: NDArray[np.float64]
    test_targets: NDArray[np.float64]
    target_mean: float
    target_scale: float

    @property
    def name(self) -> str:
        return f"{self.domain}:seed-{self.seed}"


def _encode(params: dict[str, float]) -> NDArray[np.float64]:
    """Normalize a raw 7-variable parameter dict to [0, 1]^7."""
    row = np.empty(CHEM_INPUT_DIMENSION, dtype=np.float64)
    for index, var in enumerate(CHEM_INPUT_VARS):
        low, high = CHEM_INPUT_BOUNDS[var]
        value = float(params[var])
        if var in CHEM_LOG_VARS:
            denom = math.log(high) - math.log(low)
            row[index] = np.clip((math.log(max(value, 1e-9)) - math.log(low)) / denom, 0.0, 1.0)
        else:
            row[index] = np.clip((value - low) / (high - low), 0.0, 1.0)
    return row


def _sample_params(rng: np.random.Generator) -> dict[str, float]:
    params: dict[str, float] = {}
    for var in CHEM_INPUT_VARS:
        low, high = CHEM_INPUT_BOUNDS[var]
        if var in CHEM_LOG_VARS:
            params[var] = float(np.exp(rng.uniform(math.log(low), math.log(high))))
        else:
            params[var] = float(rng.uniform(low, high))
    return params


def chembench_episodes(
    root: Path,
    split: MetaSplit,
    episodes_per_domain: int = 1,
    *,
    train_points: int = 20,
    test_points: int = 20,
    noise_level: float = 0.01,
    difficulty: str = "easy",
    oracle_factory: OracleFactory | None = None,
) -> list[ChemEpisode]:
    """Generate deterministic rate-regression episodes through the public oracle API.

    Training and validation use the same domains with disjoint experiment
    seeds. Test episodes use only held-out domains. Every episode also has
    disjoint within-domain train and test experiments.
    """
    if episodes_per_domain < 1:
        raise ValueError("episodes_per_domain must be positive")
    if train_points < 1 or test_points < 1:
        raise ValueError("train_points and test_points must be positive")
    if noise_level < 0.0:
        raise ValueError("noise_level must be nonnegative")

    factory = oracle_factory or _load_oracle_factory(root)
    domains = CHEM_TEST_DOMAINS if split == "test" else CHEM_TRAIN_DOMAINS
    offset = {"train": 0, "validation": 10_000, "test": 20_000}[split]
    episodes: list[ChemEpisode] = []
    for domain_index, domain in enumerate(domains):
        for episode_index in range(episodes_per_domain):
            seed = offset + domain_index * 1_000 + episode_index
            law_version = ("v0", "v1", "v2")[episode_index % 3]
            oracle = factory(
                domain,
                difficulty=difficulty,
                noise_level=noise_level,
                law_version=law_version,
            )
            rng = np.random.default_rng(seed)
            train_rows: list[NDArray[np.float64]] = []
            train_targets: list[float] = []
            for _ in range(train_points):
                params = _sample_params(rng)
                train_rows.append(_encode(params))
                train_targets.append(float(oracle.run(params).measurement))
            test_rows: list[NDArray[np.float64]] = []
            test_targets: list[float] = []
            for _ in range(test_points):
                params = _sample_params(rng)
                test_rows.append(_encode(params))
                test_targets.append(float(oracle.run(params).measurement))

            x_train = np.stack(train_rows)
            y_train_raw = np.asarray(train_targets, dtype=np.float64)
            x_test = np.stack(test_rows)
            y_test_raw = np.asarray(test_targets, dtype=np.float64)
            if not np.all(np.isfinite(x_train)) or not np.all(np.isfinite(y_train_raw)):
                raise ValueError("oracle produced non-finite training data")
            if not np.all(np.isfinite(x_test)) or not np.all(np.isfinite(y_test_raw)):
                raise ValueError("oracle produced non-finite held-out data")

            target_mean = float(y_train_raw.mean())
            target_scale = max(float(y_train_raw.std()), 0.1 * abs(target_mean), 1e-6)
            episodes.append(
                ChemEpisode(
                    domain=domain,
                    seed=seed,
                    train_inputs=x_train,
                    train_targets=(y_train_raw - target_mean) / target_scale,
                    test_inputs=x_test,
                    test_targets=(y_test_raw - target_mean) / target_scale,
                    target_mean=target_mean,
                    target_scale=target_scale,
                )
            )
    return episodes


class ChemBenchKernelEvaluator:
    """Held-out rate-law prediction across ChemBench mechanism domains."""

    def __init__(
        self,
        executor: KernelExecutor,
        root: Path,
        *,
        split: MetaSplit = "train",
        episodes_per_domain: int = 1,
        train_points: int = 20,
        test_points: int = 20,
        noise_level: float = 0.01,
        difficulty: str = "easy",
        amplitude_grid: tuple[float, ...] = (0.1, 0.3, 1.0, 3.0),
        noise_grid: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1),
        failure_penalty: float = 2.0,
        novelty_evaluator: FunctionalNoveltyEvaluator | None = None,
        novelty_failure_penalty: float = 0.1,
        worst_domain_weight: float = 0.0,
        episodes: list[ChemEpisode] | None = None,
        oracle_factory: OracleFactory | None = None,
    ) -> None:
        self.executor = executor
        self.split = split
        self.episodes = episodes or chembench_episodes(
            root,
            split,
            episodes_per_domain,
            train_points=train_points,
            test_points=test_points,
            noise_level=noise_level,
            difficulty=difficulty,
            oracle_factory=oracle_factory,
        )
        self.amplitude_grid = amplitude_grid
        self.noise_grid = noise_grid
        self.failure_penalty = failure_penalty
        self.novelty_evaluator = novelty_evaluator
        self.novelty_failure_penalty = novelty_failure_penalty
        self.worst_domain_weight = worst_domain_weight

    def evaluate(self, candidate: CandidateBundle, _dataset: object = None) -> EvaluationRecord:
        started = time.monotonic()
        results: list[PredictiveTaskResult] = []
        failures: list[dict[str, str]] = []
        for episode in self.episodes:
            try:
                results.append(self._evaluate_episode(candidate, episode))
            except Exception as error:
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
        worst_domain_crps = max((result.crps for result in results), default=self.failure_penalty)
        if failures:
            worst_domain_crps = max(worst_domain_crps, self.failure_penalty)
        score = -(aggregate_crps + self.worst_domain_weight * worst_domain_crps)
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
            jitter=mean_noise,
            condition_number=max_condition,
            metadata={
                "objective": (
                    "negative_mean_heldout_crps"
                    if self.worst_domain_weight == 0.0
                    else "negative_mean_plus_worst_domain_heldout_crps"
                ),
                "benchmark": "chembench_rate_surrogate",
                "split": self.split,
                "input_features": list(CHEM_INPUT_VARS),
                "task_count": task_count,
                "successful_tasks": len(results),
                "failed_tasks": len(failures),
                "mean_crps": aggregate_crps,
                "worst_domain_crps": worst_domain_crps,
                "worst_domain_weight": self.worst_domain_weight,
                "mean_nlpd": _mean_or_penalty(results, "nlpd", self.failure_penalty, task_count),
                "mean_rmse": _mean_or_penalty(results, "rmse", self.failure_penalty, task_count),
                "tasks": [result.__dict__ for result in results],
                "failures": failures,
                **novelty_metadata,
            },
        )

    def _evaluate_episode(
        self, candidate: CandidateBundle, episode: ChemEpisode
    ) -> PredictiveTaskResult:
        started = time.monotonic()
        n_train = episode.train_inputs.shape[0]
        inputs = np.vstack([episode.train_inputs, episode.test_inputs])
        gram = self.executor.gram(candidate, inputs).as_array()
        fit = _fit_gp_scale(
            gram[:n_train, :n_train],
            episode.train_targets,
            self.amplitude_grid,
            self.noise_grid,
        )
        mean, variance = _gp_predict(
            gram,
            n_train,
            episode.train_targets,
            amplitude=fit[1],
            noise=fit[2],
        )
        crps = _mean_crps(mean, variance, episode.test_targets)
        nlpd = float(
            np.mean(
                0.5 * np.log(2 * math.pi * variance)
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
            noise=fit[2],
            amplitude=fit[1],
            condition_number=fit[3],
            runtime_seconds=time.monotonic() - started,
        )


def chembench_baseline_candidates() -> list[CandidateBundle]:
    """Normalized-input reference kernels for the ChemBench adapter."""
    return [
        candidate.model_copy(
            update={
                "name": candidate.name.replace("meta_", "chem_", 1),
                "rationale": "Fixed normalized-input reference for ChemBench.",
            }
        )
        for candidate in meta_baseline_candidates()
    ]


def _load_oracle_factory(root: Path) -> OracleFactory:
    root = root.expanduser().resolve()
    if not (root / "autoscilab" / "oracle" / "chembench.py").is_file():
        raise FileNotFoundError(f"LLM-AutoSciLab ChemBench oracle not found under {root}")
    value = str(root)
    if value not in sys.path:
        sys.path.insert(0, value)
    from autoscilab.oracle.chembench import ChemBenchOracle  # type: ignore[import-not-found]

    def factory(domain_id: str, **kwargs: Any) -> Any:
        # ChemBenchOracle.__init__ prints a diagnostic line to stdout; the CLI
        # writes benchmark reports to stdout, so this would corrupt that JSON.
        with contextlib.redirect_stdout(io.StringIO()):
            return ChemBenchOracle(domain_id, **kwargs)

    return factory
