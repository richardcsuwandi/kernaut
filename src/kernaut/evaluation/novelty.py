from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from kernaut.archive import CandidateStore
from kernaut.contracts import validate_source_shape
from kernaut.execution import ExecutionFailure, KernelExecutor
from kernaut.models import CandidateBundle, CandidateOrigin, ContractKind


@dataclass(frozen=True)
class FunctionalNoveltyResult:
    distance: float
    nearest_candidate_id: str
    nearest_name: str
    passed: bool
    skipped_candidate_ids: tuple[str, ...] = ()


class NoveltyPolicy:
    """Check structural novelty requirements when a kernel is submitted."""

    required_fields = (
        "mathematical_form",
        "psd_argument",
        "novelty_claim",
        "closest_known_kernel",
        "expected_bo_behavior",
        "construction_niche",
        "equivalence_analysis",
        "optimized_parameters",
        "domain_scale_analysis",
        "irrelevant_coordinate_mechanism",
        "cross_coordinate_mechanism",
        "falsification_test",
        "feature_growth",
        "expected_conditioning",
    )
    construction_niches = {
        "input_geometry",
        "spectral_periodic",
        "feature_projection",
        "additive_interaction",
        "multiresolution_local",
        "residual_composite",
        "open_ended",
    }

    def __init__(self, *, allow_closure: bool = False) -> None:
        self.allow_closure = allow_closure

    def validate_submission(self, candidate: CandidateBundle, store: CandidateStore) -> None:
        if candidate.origin != CandidateOrigin.DISCOVERED:
            return
        if candidate.contract == ContractKind.CLOSURE and not self.allow_closure:
            raise ValueError(
                "novelty mode rejects closure-tree candidates: propose a new feature map, "
                "input transform, or spectral construction instead"
            )
        missing = [field for field in self.required_fields if not getattr(candidate, field).strip()]
        if missing:
            raise ValueError("novelty mode requires: " + ", ".join(missing))
        if candidate.construction_niche not in self.construction_niches:
            raise ValueError(
                "construction_niche must be one of: " + ", ".join(sorted(self.construction_niches))
            )

        source_errors = validate_source_shape(candidate.source, candidate.contract)
        if source_errors:
            raise ValueError("invalid candidate source: " + "; ".join(source_errors))

        # An empty parent list identifies an independent idea. A parent ID claims a revision
        # and triggers the relationship checks below. Later checks compare functional novelty
        # against accepted discoveries. Assigning unrelated parents adds no protection against
        # duplicates and records a false relationship.
        for parent_id in candidate.parents:
            parent = store.get_candidate(parent_id)
            if parent is None:
                raise ValueError(f"parent candidate not found: {parent_id}")
            if parent.contract == candidate.contract and _semantic_source(
                parent.source
            ) == _semantic_source(candidate.source):
                raise ValueError(
                    f"candidate is a parameter-only or metadata-only revision of parent {parent_id}"
                )


class FunctionalNoveltyEvaluator:
    """Measure distance from a fixed set of reference Gram matrices.

    The distance is invariant to kernel amplitude.
    """

    def __init__(
        self,
        executor: KernelExecutor,
        *,
        archive_candidates: Callable[[], list[CandidateBundle]] | None = None,
        minimum_distance: float = 0.08,
        probe_dimensions: tuple[int, ...] = (2, 5, 8),
        probe_points: int = 16,
        seed: int = 31_415,
    ) -> None:
        self.executor = executor
        self.archive_candidates = archive_candidates or (lambda: [])
        self.minimum_distance = minimum_distance
        rng = np.random.default_rng(seed)
        self.probes = tuple(
            rng.uniform(size=(probe_points, dimension)) for dimension in probe_dimensions
        )
        self.references = _reference_candidates()
        self._cache: dict[str, tuple[NDArray[np.float64], ...]] = {}

    def evaluate(self, candidate: CandidateBundle) -> FunctionalNoveltyResult:
        candidate_fp = self._fingerprint(candidate)
        comparison = list(self.references)
        comparison.extend(
            item
            for item in self.archive_candidates()
            if item.origin == CandidateOrigin.DISCOVERED
            and item.candidate_id != candidate.candidate_id
        )
        nearest: CandidateBundle | None = None
        distance = float("inf")
        skipped: list[str] = []
        for reference in comparison:
            try:
                reference_fp = self._fingerprint(reference)
            except ExecutionFailure:
                # Historical rejected candidates must not prevent a valid candidate from
                # being scored. Callers should still supply accepted candidates only.
                skipped.append(reference.candidate_id)
                continue
            distances = [
                np.linalg.norm(a - b, ord="fro")
                for a, b in zip(candidate_fp, reference_fp, strict=True)
            ]
            current = float(np.mean(distances))
            if current < distance:
                distance, nearest = current, reference
        if nearest is None:
            raise RuntimeError("functional novelty reference bank is empty")
        return FunctionalNoveltyResult(
            distance=distance,
            nearest_candidate_id=nearest.candidate_id,
            nearest_name=nearest.name,
            passed=distance >= self.minimum_distance,
            skipped_candidate_ids=tuple(skipped),
        )

    def _fingerprint(self, candidate: CandidateBundle) -> tuple[NDArray[np.float64], ...]:
        cached = self._cache.get(candidate.candidate_id)
        if cached is not None:
            return cached
        fingerprint = tuple(
            _normalized_centered_gram(self.executor.gram(candidate, probe).as_array())
            for probe in self.probes
        )
        self._cache[candidate.candidate_id] = fingerprint
        return fingerprint


def formulation_id(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _semantic_source(source: str) -> str:
    """Canonicalize Python source while ignoring comments and docstrings."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            del body[0]
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


def _normalized_centered_gram(gram: NDArray[np.float64]) -> NDArray[np.float64]:
    matrix = np.asarray(gram, dtype=np.float64)
    centered = matrix - matrix.mean(axis=0)[None, :] - matrix.mean(axis=1)[:, None]
    centered += float(matrix.mean())
    norm = float(np.linalg.norm(centered, ord="fro"))
    if not np.isfinite(norm) or norm < 1e-12:
        return np.zeros_like(centered)
    return centered / norm


def _reference_candidates() -> list[CandidateBundle]:
    source = 'def closure_tree(parameters):\n    return parameters["tree"]\n'
    nodes: list[tuple[str, dict[str, Any]]] = []
    for kind in ("rbf", "matern32", "matern52"):
        for lengthscale in (0.1, 0.25, 0.5, 1.0):
            nodes.append(
                (
                    f"reference_{kind}_{str(lengthscale).replace('.', '_')}",
                    {"op": "base", "kind": kind, "lengthscale": lengthscale},
                )
            )
    nodes.extend(
        [
            (
                "reference_short_matern_long_rbf",
                {
                    "op": "sum",
                    "children": [
                        {"op": "base", "kind": "matern32", "lengthscale": 0.25},
                        {"op": "base", "kind": "rbf", "lengthscale": 1.0},
                    ],
                },
            ),
            (
                "reference_product_matern_multiscale",
                {
                    "op": "sum",
                    "children": [
                        {
                            "op": "product",
                            "children": [
                                {"op": "base", "kind": "matern52", "lengthscale": 0.5},
                                {"op": "base", "kind": "matern32", "lengthscale": 0.25},
                            ],
                        },
                        {"op": "base", "kind": "matern52", "lengthscale": 1.0},
                    ],
                },
            ),
        ]
    )
    return [
        CandidateBundle(
            name=name,
            contract=ContractKind.CLOSURE,
            origin=CandidateOrigin.BASELINE,
            source=source,
            parameters={"tree": tree},
        )
        for name, tree in nodes
    ]
