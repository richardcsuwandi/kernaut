from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from kernaut.models import CandidateBundle


class ExecutionFailure(RuntimeError):
    pass


class ExecutionResult(BaseModel):
    gram: list[list[float]]
    runtime_seconds: float
    stdout: str = ""
    stderr: str = ""
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)

    def as_array(self) -> NDArray[np.float64]:
        return np.asarray(self.gram, dtype=np.float64)


class KernelExecutor(ABC):
    """Execution seam; a future container backend implements this same interface."""

    @abstractmethod
    def gram(self, candidate: CandidateBundle, x: NDArray[np.float64]) -> ExecutionResult:
        raise NotImplementedError
