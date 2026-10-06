from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from kernaut.contracts import required_entrypoint, validate_source_shape
from kernaut.execution import ExecutionFailure, KernelExecutor
from kernaut.models import (
    CandidateBundle,
    CheckResult,
    ContractKind,
    EvidenceRecord,
    EvidenceTier,
)


@dataclass(frozen=True)
class VerificationPolicy:
    trials: int = 8
    contract_trials: int = 2
    points_per_trial: int = 12
    input_dimension: int = 2
    input_dimensions: tuple[int, ...] | None = None
    eigenvalue_tolerance: float = 1e-8
    contract_absolute_tolerance: float = 1e-8
    contract_relative_tolerance: float = 1e-7
    seed: int = 0


class Verifier:
    def __init__(
        self,
        executor: KernelExecutor,
        policy: VerificationPolicy | None = None,
    ) -> None:
        self.executor = executor
        self.policy = policy or VerificationPolicy()

    def verify(self, candidate: CandidateBundle) -> EvidenceRecord:
        checks: list[CheckResult] = []
        source_errors = validate_source_shape(candidate.source, candidate.contract)
        checks.append(
            CheckResult(
                name="source_shape",
                passed=not source_errors,
                detail="; ".join(source_errors)
                if source_errors
                else f"found {required_entrypoint(candidate.contract)}",
            )
        )
        if source_errors:
            return self._record(candidate, None, False, checks)

        rng = np.random.default_rng(self.policy.seed)
        smallest_eigenvalue = float("inf")
        largest_asymmetry = 0.0
        total_runtime = 0.0
        dimensions = self.policy.input_dimensions or (self.policy.input_dimension,)
        for dimension in dimensions:
            for trial in range(self.policy.trials):
                x = rng.normal(size=(self.policy.points_per_trial, dimension))
                try:
                    result = self.executor.gram(candidate, x)
                except ExecutionFailure as error:
                    checks.append(
                        CheckResult(
                            name="execution",
                            passed=False,
                            detail=f"dimension {dimension}, trial {trial}: {error}",
                        )
                    )
                    return self._record(candidate, None, False, checks)
                gram = result.as_array()
                total_runtime += result.runtime_seconds
                asymmetry = float(np.max(np.abs(gram - gram.T)))
                eigenvalue = float(np.linalg.eigvalsh((gram + gram.T) / 2).min())
                largest_asymmetry = max(largest_asymmetry, asymmetry)
                smallest_eigenvalue = min(smallest_eigenvalue, eigenvalue)
        checks.append(
            CheckResult(
                name="execution",
                passed=True,
                detail=(
                    f"{self.policy.trials * len(dimensions)} isolated executions succeeded "
                    f"across dimensions {list(dimensions)}"
                ),
                metrics={"runtime_seconds": total_runtime},
            )
        )
        empirical = (
            largest_asymmetry <= self.policy.eigenvalue_tolerance
            and smallest_eigenvalue >= -self.policy.eigenvalue_tolerance
        )
        checks.append(
            CheckResult(
                name="randomized_psd",
                passed=empirical,
                detail="randomized Gram matrices checked",
                metrics={
                    "minimum_eigenvalue": smallest_eigenvalue,
                    "maximum_asymmetry": largest_asymmetry,
                },
            )
        )
        if not empirical:
            return self._record(candidate, EvidenceTier.EXECUTABLE, False, checks)

        if candidate.contract == ContractKind.UNVERIFIED:
            checks.append(
                CheckResult(
                    name="contract_wrapper",
                    passed=False,
                    detail=(
                        "no verified construction interface; retained as empirical evidence only"
                    ),
                )
            )
            return self._record(candidate, EvidenceTier.EMPIRICAL, False, checks)

        conformance = self._check_contract_conformance(candidate, rng)
        checks.append(conformance)
        if not conformance.passed:
            return self._record(candidate, EvidenceTier.EMPIRICAL, False, checks)

        checks.append(
            CheckResult(
                name="contract_interpreter",
                passed=True,
                detail=(
                    f"trusted {candidate.contract.value} interpreter owns the "
                    "PSD-preserving construction"
                ),
            )
        )
        return self._record(candidate, EvidenceTier.CONTRACT_CERTIFIED, True, checks)

    def _check_contract_conformance(
        self, candidate: CandidateBundle, rng: np.random.Generator
    ) -> CheckResult:
        """Check that the kernel does not depend on how inputs are grouped into a batch.

        A PSD-preserving wrapper alone cannot establish this independence. A candidate
        could make one feature row depend on unrelated points in the batch. These checks
        change the batch representation and compare the resulting kernels. Candidate
        values remain inside the subprocess.
        """
        relation_errors: dict[str, float] = {
            "determinism": 0.0,
            "permutation": 0.0,
            "subset": 0.0,
            "extension": 0.0,
            "duplicate": 0.0,
        }
        total_runtime = 0.0
        dimensions = self.policy.input_dimensions or (self.policy.input_dimension,)
        try:
            for dimension in dimensions:
                for _ in range(self.policy.contract_trials):
                    runtime, errors = self._check_contract_trial(
                        candidate, rng, dimension=dimension
                    )
                    total_runtime += runtime
                    for name, error in errors.items():
                        relation_errors[name] = max(relation_errors[name], error)
        except ExecutionFailure as error:
            return CheckResult(
                name="contract_conformance",
                passed=False,
                detail=f"contract check execution failed: {error}",
                metrics={"runtime_seconds": total_runtime},
            )

        tolerance = (
            self.policy.contract_absolute_tolerance + self.policy.contract_relative_tolerance
        )
        failed = [name for name, error in relation_errors.items() if error > tolerance]
        metrics: dict[str, float | int | str | bool | None] = {
            f"{name}_relative_error": error for name, error in relation_errors.items()
        }
        metrics["runtime_seconds"] = total_runtime
        metrics["trials"] = self.policy.contract_trials * len(dimensions)
        metrics["dimensions"] = ",".join(str(dimension) for dimension in dimensions)
        return CheckResult(
            name="contract_conformance",
            passed=not failed,
            detail=(
                "determinism, permutation, subset, extension, and duplicate checks passed"
                if not failed
                else f"failed observable kernel relations: {', '.join(failed)}"
            ),
            metrics=metrics,
        )

    def _check_contract_trial(
        self,
        candidate: CandidateBundle,
        rng: np.random.Generator,
        *,
        dimension: int,
    ) -> tuple[float, dict[str, float]]:
        count = max(self.policy.points_per_trial, 4)
        x = rng.normal(size=(count, dimension))
        runtime = 0.0
        errors: dict[str, float] = {}

        base_result = self.executor.gram(candidate, x)
        replay_result = self.executor.gram(candidate, x)
        base = base_result.as_array()
        replay = replay_result.as_array()
        runtime += base_result.runtime_seconds + replay_result.runtime_seconds
        errors["determinism"] = self._relative_error(replay, base)

        permutation = rng.permutation(count)
        permuted_result = self.executor.gram(candidate, x[permutation])
        runtime += permuted_result.runtime_seconds
        errors["permutation"] = self._relative_error(
            permuted_result.as_array(), base[np.ix_(permutation, permutation)]
        )

        subset = permutation[: max(2, count // 2)]
        subset_result = self.executor.gram(candidate, x[subset])
        runtime += subset_result.runtime_seconds
        errors["subset"] = self._relative_error(
            subset_result.as_array(), base[np.ix_(subset, subset)]
        )

        extra = rng.normal(size=(3, dimension))
        extended_result = self.executor.gram(candidate, np.concatenate([x, extra]))
        runtime += extended_result.runtime_seconds
        errors["extension"] = self._relative_error(extended_result.as_array()[:count, :count], base)

        duplicated_result = self.executor.gram(candidate, np.concatenate([x, x[[0]]]))
        runtime += duplicated_result.runtime_seconds
        duplicated = duplicated_result.as_array()
        duplicate_expected = np.concatenate([base[0], base[[0], 0]])
        duplicate_observed = np.concatenate([duplicated[-1, :-1], duplicated[[-1], -1]])
        errors["duplicate"] = self._relative_error(duplicate_observed, duplicate_expected)
        return runtime, errors

    def _relative_error(self, observed: np.ndarray, expected: np.ndarray) -> float:
        absolute = float(np.max(np.abs(observed - expected), initial=0.0))
        scale = max(float(np.max(np.abs(expected), initial=0.0)), 1.0)
        return absolute / scale

    @staticmethod
    def _record(
        candidate: CandidateBundle,
        tier: EvidenceTier | None,
        accepted: bool,
        checks: list[CheckResult],
    ) -> EvidenceRecord:
        return EvidenceRecord(
            candidate_id=candidate.candidate_id,
            tier=tier,
            accepted=accepted,
            checks=checks,
            contract=candidate.contract,
            implementation_digest=hashlib.sha256(candidate.source.encode()).hexdigest(),
        )
