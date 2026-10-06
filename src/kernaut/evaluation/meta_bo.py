from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from kernaut.evaluation.novelty import FunctionalNoveltyEvaluator
from kernaut.execution import KernelExecutor
from kernaut.models import CandidateBundle, CandidateOrigin, ContractKind, EvaluationRecord

MetaSplit = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class TaskSpec:
    name: str
    dimension: int
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    maximum: float

    def evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        values = _to_domain(x, self.lower, self.upper)
        return _OBJECTIVES[self.name](values)


@dataclass(frozen=True)
class ProceduralTask:
    """Apply a deterministic, hidden coordinate transformation to a Bayesian optimization
    (BO) function.
    """

    spec: TaskSpec
    seed: int
    permutation: tuple[int, ...]
    reflected: tuple[bool, ...]
    powers: tuple[float, ...]
    observed_dimension: int
    active_dimensions: tuple[int, ...]
    couplings: tuple[float, ...]

    @property
    def name(self) -> str:
        return f"{self.spec.name}:seed-{self.seed}"

    @property
    def dimension(self) -> int:
        return self.observed_dimension

    @property
    def maximum(self) -> float:
        return self.spec.maximum

    def evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        unit = np.asarray(x, dtype=np.float64)
        transformed = unit[:, self.active_dimensions][:, self.permutation]
        reflected = np.asarray(self.reflected, dtype=bool)
        transformed = np.where(reflected[None, :], 1.0 - transformed, transformed)
        transformed = np.power(np.clip(transformed, 0.0, 1.0), np.asarray(self.powers))
        # An invertible triangular warp introduces non-separable geometry while preserving
        # the unit cube and therefore each objective's known optimum.
        transformed = transformed.copy()
        for index in range(1, self.spec.dimension):
            raw = transformed[:, index]
            interior = np.clip(raw, 1e-12, 1.0 - 1e-12)
            logits = np.log(interior / (1.0 - interior))
            warped = 1.0 / (
                1.0 + np.exp(-(logits + self.couplings[index] * (transformed[:, index - 1] - 0.5)))
            )
            transformed[:, index] = np.where(raw <= 0.0, 0.0, np.where(raw >= 1.0, 1.0, warped))
        return self.spec.evaluate(transformed)


@dataclass(frozen=True)
class PredictiveTaskResult:
    task: str
    crps: float
    nlpd: float
    rmse: float
    train_nll: float
    noise: float
    amplitude: float
    condition_number: float
    runtime_seconds: float


@dataclass(frozen=True)
class BORolloutResult:
    task: str
    regret_auc: float
    final_regret: float
    initial_best: float
    final_best: float
    runtime_seconds: float
    trace: tuple[float, ...]


TRAIN_TASKS = (
    TaskSpec("branin", 2, (-5.0, 0.0), (10.0, 15.0), -0.39788735772973816),
    TaskSpec("ackley", 2, (-32.768,) * 2, (32.768,) * 2, 0.0),
    TaskSpec("cosine", 8, (-1.0,) * 8, (1.0,) * 8, 8.8),
    TaskSpec("hartmann", 6, (0.0,) * 6, (1.0,) * 6, 3.322368011415515),
    TaskSpec("levy", 6, (-10.0,) * 6, (10.0,) * 6, 0.0),
)

TEST_TASKS = (
    TaskSpec("bukin", 2, (-15.0, -3.0), (-5.0, 3.0), 0.0),
    TaskSpec("dropwave", 2, (-5.12,) * 2, (5.12,) * 2, 1.0),
    TaskSpec("griewank", 5, (-600.0,) * 5, (600.0,) * 5, 0.0),
    TaskSpec("holdertable", 2, (-10.0,) * 2, (10.0,) * 2, 19.20850256788675),
    TaskSpec("rosenbrock", 6, (-5.0,) * 6, (10.0,) * 6, 0.0),
    TaskSpec("rastrigin", 6, (-5.12,) * 6, (5.12,) * 6, 0.0),
)


def task_specs(split: MetaSplit) -> tuple[TaskSpec, ...]:
    return TEST_TASKS if split == "test" else TRAIN_TASKS


def procedural_tasks(split: MetaSplit, episodes_per_family: int = 2) -> list[ProceduralTask]:
    if episodes_per_family < 1:
        raise ValueError("episodes_per_family must be positive")
    seed_offset = {"train": 0, "validation": 10_000, "test": 20_000}[split]
    tasks: list[ProceduralTask] = []
    for family_index, spec in enumerate(task_specs(split)):
        for episode in range(episodes_per_family):
            seed = seed_offset + family_index * 1_000 + episode
            rng = np.random.default_rng(seed)
            nuisance_dimensions = min(int(rng.integers(0, 3)), max(0, 8 - spec.dimension))
            observed_dimension = spec.dimension + nuisance_dimensions
            tasks.append(
                ProceduralTask(
                    spec=spec,
                    seed=seed,
                    permutation=tuple(int(value) for value in rng.permutation(spec.dimension)),
                    reflected=tuple(bool(value) for value in rng.integers(0, 2, spec.dimension)),
                    powers=tuple(float(value) for value in rng.uniform(0.75, 1.25, spec.dimension)),
                    observed_dimension=observed_dimension,
                    active_dimensions=tuple(
                        int(value)
                        for value in rng.choice(
                            observed_dimension, size=spec.dimension, replace=False
                        )
                    ),
                    couplings=(0.0,)
                    + tuple(float(value) for value in rng.uniform(-1.5, 1.5, spec.dimension - 1)),
                )
            )
    return tasks


class MetaKernelEvaluator:
    """Score predictive performance across tasks during kernel evolution.

    Fit only an outer covariance amplitude and diagonal noise. Candidate-specific
    parameters remain part of the generated program. Measure performance with the
    mean held-out continuous ranked probability score (CRPS) across generated task
    episodes. Return negative CRPS so higher scores indicate better predictions.
    """

    def __init__(
        self,
        executor: KernelExecutor,
        *,
        split: MetaSplit = "train",
        episodes_per_family: int = 2,
        train_points_base: int = 8,
        test_points: int = 32,
        amplitude_grid: tuple[float, ...] = (0.1, 0.3, 1.0, 3.0),
        noise_grid: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1),
        failure_penalty: float = 2.0,
        novelty_evaluator: FunctionalNoveltyEvaluator | None = None,
        novelty_failure_penalty: float = 0.1,
    ) -> None:
        self.executor = executor
        self.split = split
        self.tasks = procedural_tasks(split, episodes_per_family)
        self.train_points_base = train_points_base
        self.test_points = test_points
        self.amplitude_grid = amplitude_grid
        self.noise_grid = noise_grid
        self.failure_penalty = failure_penalty
        self.novelty_evaluator = novelty_evaluator
        self.novelty_failure_penalty = novelty_failure_penalty

    def evaluate(self, candidate: CandidateBundle, _dataset: object = None) -> EvaluationRecord:
        started = time.monotonic()
        results: list[PredictiveTaskResult] = []
        failures: list[dict[str, str]] = []
        for task in self.tasks:
            try:
                results.append(self._evaluate_task(candidate, task))
            except Exception as error:  # Candidate failures are part of benchmark fitness.
                failures.append({"task": task.name, "error": f"{type(error).__name__}: {error}"})

        task_count = len(self.tasks)
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

    def _evaluate_task(
        self, candidate: CandidateBundle, task: ProceduralTask
    ) -> PredictiveTaskResult:
        # Every candidate receives exactly the same observations for a given episode.
        rng = np.random.default_rng(_stable_seed(task.name, "predictive"))
        n_train = self.train_points_base + 2 * task.dimension
        x_train = rng.uniform(size=(n_train, task.dimension))
        x_test = rng.uniform(size=(self.test_points, task.dimension))
        y_train_raw = task.evaluate(x_train)
        y_test_raw = task.evaluate(x_test)
        y_mean = float(y_train_raw.mean())
        y_scale = float(y_train_raw.std())
        if y_scale < 1e-8:
            y_scale = 1.0
        y_train = (y_train_raw - y_mean) / y_scale
        y_test = (y_test_raw - y_mean) / y_scale

        started = time.monotonic()
        gram = self.executor.gram(candidate, np.vstack([x_train, x_test])).as_array()
        fit = _fit_gp_scale(gram[:n_train, :n_train], y_train, self.amplitude_grid, self.noise_grid)
        mean, variance = _gp_predict(
            gram,
            n_train,
            y_train,
            amplitude=fit[1],
            noise=fit[2],
        )
        crps = _mean_crps(mean, variance, y_test)
        nlpd = float(
            np.mean(0.5 * np.log(2 * math.pi * variance) + 0.5 * (y_test - mean) ** 2 / variance)
        )
        rmse = float(np.sqrt(np.mean((y_test - mean) ** 2)))
        return PredictiveTaskResult(
            task=task.name,
            crps=crps,
            nlpd=nlpd,
            rmse=rmse,
            train_nll=fit[0],
            amplitude=fit[1],
            noise=fit[2],
            condition_number=fit[3],
            runtime_seconds=time.monotonic() - started,
        )


class MetaBOBenchmark:
    """Test fixed kernels on short Bayesian optimization runs with a fixed protocol."""

    def __init__(
        self,
        executor: KernelExecutor,
        *,
        episodes_per_family: int = 1,
        bo_steps: int = 8,
        candidate_pool_size: int = 128,
        amplitude_grid: tuple[float, ...] = (0.1, 0.3, 1.0, 3.0),
        noise_grid: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1),
    ) -> None:
        if bo_steps < 1 or candidate_pool_size < 8:
            raise ValueError("bo_steps must be positive and candidate_pool_size must be at least 8")
        self.executor = executor
        self.episodes_per_family = episodes_per_family
        self.bo_steps = bo_steps
        self.candidate_pool_size = candidate_pool_size
        self.amplitude_grid = amplitude_grid
        self.noise_grid = noise_grid

    def evaluate(self, candidate: CandidateBundle, split: MetaSplit) -> dict[str, object]:
        started = time.monotonic()
        results: list[BORolloutResult] = []
        failures: list[dict[str, str]] = []
        for task in procedural_tasks(split, self.episodes_per_family):
            try:
                results.append(self._rollout(candidate, task))
            except Exception as error:
                failures.append({"task": task.name, "error": f"{type(error).__name__}: {error}"})
        total = len(results) + len(failures)
        penalty = float(self.bo_steps + 1)
        regret_auc = (
            sum(result.regret_auc for result in results) + penalty * len(failures)
        ) / total
        final_regret = (
            sum(result.final_regret for result in results) + penalty * len(failures)
        ) / total
        return {
            "candidate_id": candidate.candidate_id,
            "name": candidate.name,
            "origin": candidate.origin.value,
            "split": split,
            "regret_auc": regret_auc,
            "final_regret": final_regret,
            "successful_tasks": len(results),
            "failed_tasks": len(failures),
            "runtime_seconds": time.monotonic() - started,
            "tasks": [result.__dict__ for result in results],
            "failures": failures,
        }

    def _rollout(self, candidate: CandidateBundle, task: ProceduralTask) -> BORolloutResult:
        started = time.monotonic()
        rng = np.random.default_rng(_stable_seed(task.name, "bo"))
        n_initial = 4 + 2 * task.dimension
        observed_x = rng.uniform(size=(n_initial, task.dimension))
        observed_y = task.evaluate(observed_x)
        initial_best = float(observed_y.max())
        scale = max(task.maximum - initial_best, 1e-8)
        regret_trace = [max(task.maximum - initial_best, 0.0) / scale]

        for _ in range(self.bo_steps):
            pool = rng.uniform(size=(self.candidate_pool_size, task.dimension))
            n_train = len(observed_x)
            gram = self.executor.gram(candidate, np.vstack([observed_x, pool])).as_array()
            y_mean = float(observed_y.mean())
            y_scale = max(float(observed_y.std()), 1e-8)
            y_train = (observed_y - y_mean) / y_scale
            fit = _fit_gp_scale(
                gram[:n_train, :n_train], y_train, self.amplitude_grid, self.noise_grid
            )
            mean, variance = _gp_predict(gram, n_train, y_train, amplitude=fit[1], noise=fit[2])
            best_standardized = float(y_train.max())
            acquisition = _expected_improvement(mean, variance, best_standardized)
            query = pool[int(np.argmax(acquisition))]
            query_y = float(task.evaluate(query[None, :])[0])
            observed_x = np.vstack([observed_x, query])
            observed_y = np.append(observed_y, query_y)
            best = float(observed_y.max())
            regret_trace.append(max(task.maximum - best, 0.0) / scale)

        return BORolloutResult(
            task=task.name,
            regret_auc=float(np.mean(regret_trace[1:])),
            final_regret=float(regret_trace[-1]),
            initial_best=initial_best,
            final_best=float(observed_y.max()),
            runtime_seconds=time.monotonic() - started,
            trace=tuple(float(value) for value in regret_trace),
        )


class MetaSearchEvaluator:
    """Combine predictive performance with short Bayesian optimization runs on training tasks."""

    def __init__(self, predictive: MetaKernelEvaluator, bo_benchmark: MetaBOBenchmark) -> None:
        self.predictive = predictive
        self.bo_benchmark = bo_benchmark

    def evaluate(self, candidate: CandidateBundle, _dataset: object = None) -> EvaluationRecord:
        predictive = self.predictive.evaluate(candidate)
        bo = self.bo_benchmark.evaluate(candidate, "train")
        metadata = dict(predictive.metadata)
        metadata["bo"] = bo
        return predictive.model_copy(update={"metadata": metadata})


def _spectral_mixture_tree(rng: np.random.Generator) -> dict[str, Any]:
    components = 4
    return {
        "op": "base",
        "kind": "spectral_mixture",
        "variance": 1.0,
        "weights": rng.uniform(0.5, 2.0, size=components).tolist(),
        "means": rng.uniform(-2.5, 2.5, size=components).tolist(),
        "scales": (10.0 ** rng.uniform(-1.75, -0.25, size=components)).tolist(),
    }


def meta_baseline_candidates() -> list[CandidateBundle]:
    source = 'def closure_tree(parameters):\n    return parameters["tree"]\n'
    candidates: list[CandidateBundle] = []
    for kind in ("rbf", "matern32", "matern52"):
        for lengthscale in (0.1, 0.25, 0.5, 1.0):
            candidates.append(
                CandidateBundle(
                    name=f"meta_{kind}_ls{str(lengthscale).replace('.', '_')}",
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
                    rationale="Fixed normalized-input reference for the meta-BO benchmark.",
                )
            )
    candidates.append(
        CandidateBundle(
            name="meta_linear",
            contract=ContractKind.CLOSURE,
            origin=CandidateOrigin.BASELINE,
            source=source,
            parameters={"tree": {"op": "base", "kind": "linear", "variance": 1.0}},
            rationale="Fixed normalized-input reference for the meta-BO benchmark.",
        )
    )
    for lengthscale in (0.25, 0.5, 1.0):
        for period in (0.25, 0.5, 1.0, 2.0):
            candidates.append(
                CandidateBundle(
                    name=(
                        f"meta_periodic_ls{str(lengthscale).replace('.', '_')}"
                        f"_p{str(period).replace('.', '_')}"
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
                        "Coarse-grid periodic reference for the meta-BO benchmark "
                        "(Wilson 2013 baseline family)."
                    ),
                )
            )
    for lengthscale in (0.25, 0.5, 1.0):
        for alpha in (0.5, 1.0, 4.0):
            candidates.append(
                CandidateBundle(
                    name=(
                        f"meta_rq_ls{str(lengthscale).replace('.', '_')}"
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
                    rationale="Coarse-grid rational-quadratic reference.",
                )
            )
    for kind in ("ard_rbf", "ard_matern32", "ard_matern52"):
        suffix = kind.removeprefix("ard_")
        patterns: tuple[tuple[str, dict[str, Any]], ...] = (
            ("uniform0_25", {"lengthscales": 0.25}),
            ("uniform1", {"lengthscales": 1.0}),
            (
                "ramp_up",
                {"lengthscales": {"mode": "geometric_ramp", "min": 0.25, "max": 4.0}},
            ),
            (
                "ramp_down",
                {"lengthscales": {"mode": "geometric_ramp", "min": 4.0, "max": 0.25}},
            ),
        )
        for label, pattern in patterns:
            candidates.append(
                CandidateBundle(
                    name=f"meta_{kind}_{label}",
                    contract=ContractKind.CLOSURE,
                    origin=CandidateOrigin.BASELINE,
                    source=source,
                    parameters={"tree": {"op": "base", "kind": kind, "variance": 1.0, **pattern}},
                    rationale=(
                        f"Per-coordinate {suffix} reference; structured patterns cover "
                        "irrelevant or warped coordinates without per-task tuning."
                    ),
                )
            )
    rng = np.random.default_rng(2026_08_21)
    for start in range(8):
        candidates.append(
            CandidateBundle(
                name=f"meta_sm_start{start:02d}",
                contract=ContractKind.CLOSURE,
                origin=CandidateOrigin.BASELINE,
                source=source,
                parameters={"tree": _spectral_mixture_tree(rng)},
                rationale=(
                    "Seeded spectral-mixture multi-start reference (Wilson 2013); "
                    "the best start is selected on meta-training fitness only."
                ),
            )
        )
    return candidates


def _fit_gp_scale(
    kernel: NDArray[np.float64],
    y: NDArray[np.float64],
    amplitude_grid: tuple[float, ...],
    noise_grid: tuple[float, ...],
) -> tuple[float, float, float, float]:
    symmetric = (kernel + kernel.T) / 2
    best: tuple[float, float, float, float] | None = None
    for amplitude in amplitude_grid:
        for noise in noise_grid:
            matrix = amplitude * symmetric + noise * np.eye(len(y))
            try:
                factor = np.linalg.cholesky(matrix)
                alpha = np.linalg.solve(factor.T, np.linalg.solve(factor, y))
            except np.linalg.LinAlgError:
                continue
            nll = float(
                0.5 * y @ alpha
                + np.log(np.diag(factor)).sum()
                + 0.5 * len(y) * math.log(2 * math.pi)
            )
            candidate = (nll, amplitude, noise, float(np.linalg.cond(matrix)))
            if best is None or candidate[0] < best[0]:
                best = candidate
    if best is None:
        raise ValueError("GP fitting failed for every amplitude/noise pair")
    return best


def _gp_predict(
    gram: NDArray[np.float64],
    n_train: int,
    y_train: NDArray[np.float64],
    *,
    amplitude: float,
    noise: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    train = amplitude * (gram[:n_train, :n_train] + gram[:n_train, :n_train].T) / 2
    cross = amplitude * gram[:n_train, n_train:]
    test_diag = amplitude * np.diag(gram[n_train:, n_train:])
    matrix = train + noise * np.eye(n_train)
    factor = np.linalg.cholesky(matrix)
    alpha = np.linalg.solve(factor.T, np.linalg.solve(factor, y_train))
    # Some BLAS builds leak floating-point status flags into an otherwise finite matmul.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        mean = cross.T @ alpha
        projected = np.linalg.solve(factor, cross)
        variance = np.maximum(test_diag + noise - np.sum(projected * projected, axis=0), 1e-10)
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(variance)):
        raise ValueError("GP prediction produced non-finite values")
    return mean, variance


def _expected_improvement(
    mean: NDArray[np.float64], variance: NDArray[np.float64], best: float
) -> NDArray[np.float64]:
    sigma = np.sqrt(variance)
    improvement = mean - best
    z = improvement / sigma
    cdf = np.asarray([0.5 * (1.0 + math.erf(value / math.sqrt(2))) for value in z])
    pdf = np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    return np.maximum(improvement * cdf + sigma * pdf, 0.0)


def _mean_crps(
    mean: NDArray[np.float64], variance: NDArray[np.float64], target: NDArray[np.float64]
) -> float:
    sigma = np.sqrt(variance)
    z = (target - mean) / sigma
    cdf = np.asarray([0.5 * (1.0 + math.erf(value / math.sqrt(2))) for value in z])
    pdf = np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    values = sigma * (z * (2 * cdf - 1) + 2 * pdf - 1 / math.sqrt(math.pi))
    return float(np.mean(values))


def _mean_or_penalty(
    results: list[PredictiveTaskResult],
    field: Literal["nlpd", "rmse"],
    penalty: float,
    total: int,
) -> float:
    return (
        sum(float(getattr(result, field)) for result in results) + penalty * (total - len(results))
    ) / total


def _stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _to_domain(
    unit: NDArray[np.float64], lower: tuple[float, ...], upper: tuple[float, ...]
) -> NDArray[np.float64]:
    low = np.asarray(lower, dtype=np.float64)
    return low + np.asarray(unit, dtype=np.float64) * (np.asarray(upper) - low)


def _branin(x: NDArray[np.float64]) -> NDArray[np.float64]:
    first = x[:, 1] - 5.1 * x[:, 0] ** 2 / (4 * math.pi**2) + 5 * x[:, 0] / math.pi - 6
    return -(first**2 + 10 * (1 - 1 / (8 * math.pi)) * np.cos(x[:, 0]) + 10)


def _ackley(x: NDArray[np.float64]) -> NDArray[np.float64]:
    d = x.shape[1]
    value = -20 * np.exp(-0.2 * np.sqrt(np.sum(x * x, axis=1) / d))
    value -= np.exp(np.mean(np.cos(2 * math.pi * x), axis=1))
    return -(value + 20 + math.e)


def _cosine(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return -np.sum(0.1 * np.cos(5 * math.pi * x) - x * x, axis=1)


def _hartmann(x: NDArray[np.float64]) -> NDArray[np.float64]:
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    a = np.array(
        [
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14],
        ]
    )
    p = 1e-4 * np.array(
        [
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ]
    )
    return np.sum(
        alpha[None, :] * np.exp(-np.sum(a[None, :, :] * (x[:, None, :] - p) ** 2, axis=2)), axis=1
    )


def _levy(x: NDArray[np.float64]) -> NDArray[np.float64]:
    w = 1 + (x - 1) / 4
    first = np.sin(math.pi * w[:, 0]) ** 2
    middle = np.sum((w[:, :-1] - 1) ** 2 * (1 + 10 * np.sin(math.pi * w[:, :-1] + 1) ** 2), axis=1)
    last = (w[:, -1] - 1) ** 2 * (1 + np.sin(2 * math.pi * w[:, -1]) ** 2)
    return -(first + middle + last)


def _bukin(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return -(100 * np.sqrt(np.abs(x[:, 1] - 0.01 * x[:, 0] ** 2)) + 0.01 * np.abs(x[:, 0] + 10))


def _dropwave(x: NDArray[np.float64]) -> NDArray[np.float64]:
    radius = np.linalg.norm(x, axis=1)
    return (1 + np.cos(12 * radius)) / (0.5 * radius**2 + 2)


def _griewank(x: NDArray[np.float64]) -> NDArray[np.float64]:
    indices = np.sqrt(np.arange(1, x.shape[1] + 1))
    return -(np.sum(x * x, axis=1) / 4000 - np.prod(np.cos(x / indices), axis=1) + 1)


def _holdertable(x: NDArray[np.float64]) -> NDArray[np.float64]:
    term = np.abs(1 - np.linalg.norm(x, axis=1) / math.pi)
    return np.abs(np.sin(x[:, 0]) * np.cos(x[:, 1]) * np.exp(term))


def _rosenbrock(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return -np.sum(100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1 - x[:, :-1]) ** 2, axis=1)


def _rastrigin(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return -(10 * x.shape[1] + np.sum(x * x - 10 * np.cos(2 * math.pi * x), axis=1))


_OBJECTIVES = {
    "branin": _branin,
    "ackley": _ackley,
    "cosine": _cosine,
    "hartmann": _hartmann,
    "levy": _levy,
    "bukin": _bukin,
    "dropwave": _dropwave,
    "griewank": _griewank,
    "holdertable": _holdertable,
    "rosenbrock": _rosenbrock,
    "rastrigin": _rastrigin,
}
