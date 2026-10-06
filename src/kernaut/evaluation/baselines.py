from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from kernaut.archive import CandidateStore
from kernaut.models import CandidateBundle, CandidateOrigin, ContractKind, EvaluationRecord
from kernaut.verification import Verifier

from .gp import Dataset, GaussianProcessEvaluator

BASELINE_SOURCE = """def closure_tree(parameters):
    return parameters["tree"]
"""


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    label: str
    kind: str
    uses_lengthscale: bool = True


STANDARD_BASELINES = (
    BaselineSpec("baseline_linear", "Linear", "linear", uses_lengthscale=False),
    BaselineSpec("baseline_rbf", "RBF", "rbf"),
    BaselineSpec("baseline_matern32", "Matérn-3/2", "matern32"),
    BaselineSpec("baseline_matern52", "Matérn-5/2", "matern52"),
    BaselineSpec("baseline_periodic", "Periodic", "periodic"),
    BaselineSpec("baseline_rq", "RationalQuadratic", "rq"),
    BaselineSpec("baseline_spectral_mixture", "SpectralMixture", "spectral_mixture"),
    BaselineSpec("baseline_rff", "RandomFourierFeatures", "rff"),
)


def _spectral_mixture_trees(
    x: NDArray[np.float64], rng: np.random.Generator, *, components: int = 4, starts: int = 8
) -> list[dict[str, Any]]:
    distances = np.sqrt(
        np.maximum(
            np.sum(x * x, axis=1)[:, None] + np.sum(x * x, axis=1)[None, :] - 2.0 * x @ x.T,
            0.0,
        )
    )
    positive = distances[distances > 1e-12]
    reference_scale = float(np.median(positive)) if positive.size else 1.0
    trees: list[dict[str, Any]] = []
    for _ in range(starts):
        scales = np.exp(rng.normal(np.log(reference_scale), 0.75, size=components))
        trees.append(
            {
                "op": "base",
                "kind": "spectral_mixture",
                "variance": 1.0,
                "weights": rng.uniform(0.5, 2.0, size=components).tolist(),
                "means": rng.uniform(-2.5, 2.5, size=components).tolist(),
                "scales": scales.tolist(),
            }
        )
    return trees


def _random_feature_trees(x: NDArray[np.float64], rng: np.random.Generator) -> list[dict]:
    distances = np.sqrt(
        np.maximum(
            np.sum(x * x, axis=1)[:, None] + np.sum(x * x, axis=1)[None, :] - 2.0 * x @ x.T,
            0.0,
        )
    )
    positive = distances[distances > 1e-12]
    reference_scale = float(np.median(positive)) if positive.size else 1.0
    return [
        {
            "op": "base",
            "kind": "rff",
            "variance": 1.0,
            "features": features,
            "lengthscale": reference_scale,
            "seed": int(rng.integers(0, 1_000_000)),
        }
        for features in (16, 256)
    ]


class StandardBaselineRunner:
    """Tune reference kernels on a coarse parameter grid, then verify, evaluate, and archive
    them.
    """

    def __init__(
        self,
        store: CandidateStore,
        verifier: Verifier,
        evaluator: GaussianProcessEvaluator,
        *,
        lengthscales: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0),
        variances: tuple[float, ...] = (0.1, 0.3, 1.0, 3.0),
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self.store = store
        self.verifier = verifier
        self.evaluator = evaluator
        self.lengthscales = lengthscales
        self.variances = variances
        self.progress = progress or (lambda _message: None)

    def run(self, dataset: Dataset) -> list[tuple[CandidateBundle, EvaluationRecord]]:
        winners: list[tuple[CandidateBundle, EvaluationRecord]] = []
        self.progress(f"[baselines] tuning {len(STANDARD_BASELINES)} standard kernels...")
        x, _ = dataset.arrays()
        rng = np.random.default_rng(2026_08_21)
        for spec in STANDARD_BASELINES:
            best: tuple[CandidateBundle, EvaluationRecord] | None = None
            trees = self._spec_trees(spec, x, rng)
            for tree in trees:
                candidate = CandidateBundle(
                    name=spec.name,
                    contract=ContractKind.CLOSURE,
                    origin=CandidateOrigin.BASELINE,
                    source=BASELINE_SOURCE,
                    parameters={"tree": tree},
                    rationale=(
                        f"Coarse-tuned {spec.label} reference kernel. This deterministic "
                        "baseline is evaluated with the same GP backend as discovered programs."
                    ),
                )
                evaluation = self.evaluator.evaluate(candidate, dataset)
                if best is None or evaluation.score > best[1].score:
                    best = (candidate, evaluation)
            if best is None:  # pragma: no cover - grids are nonempty by construction
                continue
            candidate, evaluation = best
            evidence = self.verifier.verify(candidate)
            if not evidence.accepted:
                raise RuntimeError(f"standard baseline {spec.label} failed verification")
            self.store.add_candidate(candidate)
            self.store.add_evidence(evidence)
            self.store.add_evaluation(evaluation)
            winners.append(best)
            self.progress(
                f"[baselines] {spec.label}: score={evaluation.score:.6f} "
                f"jitter={evaluation.jitter:g}"
            )
        self.progress("[baselines] complete")
        return winners

    def _spec_trees(
        self, spec: BaselineSpec, x: NDArray[np.float64], rng: np.random.Generator
    ) -> list[dict[str, Any]]:
        if spec.kind == "spectral_mixture":
            return _spectral_mixture_trees(x, rng)
        if spec.kind == "rff":
            return _random_feature_trees(x, rng)
        lengthscales: tuple[float | None, ...] = (
            self.lengthscales if spec.uses_lengthscale else (None,)
        )
        trees: list[dict[str, Any]] = []
        for lengthscale in lengthscales:
            for variance in self.variances:
                tree: dict[str, Any] = {
                    "op": "base",
                    "kind": spec.kind,
                    "variance": variance,
                }
                if lengthscale is not None:
                    tree["lengthscale"] = lengthscale
                trees.append(tree)
        if spec.kind == "periodic":
            for period in (0.25, 0.5, 1.0, 2.0):
                for lengthscale in self.lengthscales[:4]:
                    trees.append(
                        {
                            "op": "base",
                            "kind": "periodic",
                            "variance": 1.0,
                            "lengthscale": lengthscale,
                            "period": period,
                        }
                    )
        elif spec.kind == "rq":
            for alpha in (0.5, 2.0):
                for lengthscale in self.lengthscales[:3]:
                    trees.append(
                        {
                            "op": "base",
                            "kind": "rq",
                            "variance": 1.0,
                            "lengthscale": lengthscale,
                            "alpha": alpha,
                        }
                    )
        return trees
