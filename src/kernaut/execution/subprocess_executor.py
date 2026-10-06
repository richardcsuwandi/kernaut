from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from kernaut.config import ExecutionConfig
from kernaut.models import CandidateBundle

from .base import ExecutionFailure, ExecutionResult, KernelExecutor


class SubprocessExecutor(KernelExecutor):
    """Run each candidate in a new Python process with resource limits.

    This process limits the effect of failures in research code. It does not provide
    secure isolation. Use a container backend for untrusted third-party code.
    """

    def __init__(self, config: ExecutionConfig | None = None) -> None:
        self.config = config or ExecutionConfig()

    def gram(self, candidate: CandidateBundle, x: NDArray[np.float64]) -> ExecutionResult:
        payload = {
            "candidate": candidate.model_dump(mode="json"),
            "x": np.asarray(x, dtype=np.float64).tolist(),
            "max_output_bytes": self.config.max_output_bytes,
            "limits": {
                "memory_bytes": self.config.memory_mb * 1024 * 1024,
                "cpu_seconds": self.config.cpu_seconds,
                "file_bytes": self.config.max_output_bytes,
                "open_files": 64,
            },
        }
        src_root = str(Path(__file__).resolve().parents[2])
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": src_root,
            "PYTHONNOUSERSITE": "1",
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
        started = time.monotonic()
        try:
            process = subprocess.run(
                [sys.executable, "-s", "-m", "kernaut.execution.worker"],
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                env=env,
            )
        except subprocess.TimeoutExpired as error:
            raise ExecutionFailure(
                f"candidate exceeded {self.config.timeout_seconds:.1f}s wall-clock limit"
            ) from error
        elapsed = time.monotonic() - started
        if len(process.stdout.encode()) > self.config.max_output_bytes:
            raise ExecutionFailure("worker output exceeded configured limit")
        if process.returncode != 0:
            detail = process.stderr.strip()[-2000:] or f"exit code {process.returncode}"
            raise ExecutionFailure(f"worker failed: {detail}")
        try:
            response = json.loads(process.stdout)
        except json.JSONDecodeError as error:
            raise ExecutionFailure("worker returned malformed output") from error
        if not response.get("ok"):
            raise ExecutionFailure(response.get("error", "unknown worker failure"))
        return ExecutionResult(
            gram=response["gram"],
            runtime_seconds=elapsed,
            stdout=response.get("candidate_stdout", ""),
            stderr=response.get("candidate_stderr", ""),
            metadata={
                "backend": "subprocess",
                "limit_warnings": json.dumps(response.get("limit_warnings", {})),
            },
        )
