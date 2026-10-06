from __future__ import annotations

import math
import time

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, field_validator

from kernaut.execution import KernelExecutor
from kernaut.models import CandidateBundle, EvaluationRecord


class Dataset(BaseModel):
    x: list[list[float]]
    y: list[float]

    @field_validator("x")
    @classmethod
    def nonempty_matrix(cls, value: list[list[float]]) -> list[list[float]]:
        if not value or not value[0] or any(len(row) != len(value[0]) for row in value):
            raise ValueError("x must be a nonempty rectangular matrix")
        return value

    def arrays(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        x = np.asarray(self.x, dtype=np.float64)
        y = np.asarray(self.y, dtype=np.float64)
        if len(x) != len(y) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("dataset shapes or values are invalid")
        return x, y


class GaussianProcessEvaluator:
    """Deterministic exact-GP marginal-likelihood evaluator.

    Noise is selected from a fixed grid, avoiding optimizer nondeterminism in
    the harness. More elaborate fitters can be added behind this interface.
    """

    def __init__(
        self,
        executor: KernelExecutor,
        noise_grid: tuple[float, ...] = (1e-6, 1e-4, 1e-3, 1e-2, 1e-1),
    ) -> None:
        self.executor = executor
        self.noise_grid = noise_grid

    def evaluate(self, candidate: CandidateBundle, dataset: Dataset) -> EvaluationRecord:
        x, y = dataset.arrays()
        y = y - y.mean()
        started = time.monotonic()
        execution = self.executor.gram(candidate, x)
        kernel = (execution.as_array() + execution.as_array().T) / 2
        best: tuple[float, float, float, float] | None = None
        for noise in self.noise_grid:
            jitter = noise
            try:
                matrix = kernel + jitter * np.eye(len(x))
                factor = np.linalg.cholesky(matrix)
                alpha = np.linalg.solve(factor.T, np.linalg.solve(factor, y))
                nll = float(
                    0.5 * y @ alpha
                    + np.log(np.diag(factor)).sum()
                    + 0.5 * len(x) * math.log(2 * math.pi)
                )
                condition = float(np.linalg.cond(matrix))
            except np.linalg.LinAlgError:
                continue
            if best is None or nll < best[0]:
                best = (nll, noise, jitter, condition)
        if best is None:
            raise ValueError("GP fitting failed for every noise value")
        nll, noise, jitter, condition = best
        return EvaluationRecord(
            candidate_id=candidate.candidate_id,
            score=-nll,
            negative_log_likelihood=nll,
            runtime_seconds=time.monotonic() - started,
            jitter=jitter,
            condition_number=condition,
            metadata={"noise": noise, "n": len(x), "dimension": x.shape[1]},
        )
