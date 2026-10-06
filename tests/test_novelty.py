from __future__ import annotations

import json

from kernaut.agent import HarnessTools
from kernaut.archive import CandidateStore
from kernaut.evaluation import Dataset, GaussianProcessEvaluator
from kernaut.evaluation.novelty import FunctionalNoveltyEvaluator, NoveltyPolicy
from kernaut.execution import SubprocessExecutor
from kernaut.models import CandidateBundle, CandidateOrigin, EvaluationRecord
from kernaut.verification import Verifier

FORMULATION = {
    "mathematical_form": "k(x,x') = Matern52(T(x), T(x'))",
    "psd_argument": "A positive-definite kernel remains positive definite under pullback by T.",
    "novelty_claim": "T couples coordinates through bounded nonlinear harmonics.",
    "closest_known_kernel": "Deep-kernel input warping, but deterministic and dimension shared.",
    "expected_bo_behavior": "Resolve repeated basins while retaining local Matern regularity.",
    "construction_niche": "input_geometry",
    "equivalence_analysis": "A deterministic warped-input kernel, not a polynomial kernel.",
    "optimized_parameters": "No candidate parameters are optimized in this unit test.",
    "domain_scale_analysis": "The harmonic spans one cycle on normalized inputs.",
    "irrelevant_coordinate_mechanism": "None; irrelevant dimensions remain a limitation.",
    "cross_coordinate_mechanism": "The test formulation describes bounded coupling.",
    "falsification_test": "Reject if it matches the Matern fingerprint or degrades CRPS.",
    "feature_growth": "Linear in input dimension.",
    "expected_conditioning": "Comparable to the trusted Matern base.",
}


def _tools(tmp_path) -> HarnessTools:
    executor = SubprocessExecutor()
    return HarnessTools(
        CandidateStore(tmp_path / "archive.sqlite"),
        Verifier(executor),
        GaussianProcessEvaluator(executor),
        Dataset(x=[[0.0], [1.0]], y=[0.0, 1.0]),
        novelty_policy=NoveltyPolicy(),
    )


def test_novelty_mode_requires_math_first_formulation(tmp_path) -> None:
    tools = _tools(tmp_path)
    rejected = json.loads(
        tools.submit_candidate(
            name="warp",
            contract="input_transform",
            source="def transform_point(x, parameters):\n    return x\n",
        )
    )
    assert not rejected["ok"]
    assert "propose_formulation" in rejected["error"]

    proposal = json.loads(tools.propose_formulation(**FORMULATION))
    accepted = json.loads(
        tools.submit_candidate(
            name="warp",
            contract="input_transform",
            source="def transform_point(x, parameters):\n    return x\n",
            formulation_id=proposal["formulation_id"],
        )
    )
    assert accepted["ok"]


def test_novelty_submission_rejects_wrong_entrypoint_before_archiving(tmp_path) -> None:
    tools = _tools(tmp_path)
    proposal = json.loads(tools.propose_formulation(**FORMULATION))
    rejected = json.loads(
        tools.submit_candidate(
            name="wrong_entrypoint",
            contract="input_transform",
            source="def input_transform(x, parameters):\n    return x\n",
            formulation_id=proposal["formulation_id"],
        )
    )
    assert not rejected["ok"]
    assert "transform_point" in rejected["error"]
    assert tools.store.recent() == []


def test_novelty_policy_rejects_discovered_closure(tmp_path) -> None:
    store = CandidateStore(tmp_path / "archive.sqlite")
    candidate = CandidateBundle(
        name="ordinary_sum",
        contract="closure",
        source='def closure_tree(parameters):\n    return parameters["tree"]\n',
        parameters={"tree": {"op": "base", "kind": "rbf"}},
        **FORMULATION,
    )
    try:
        NoveltyPolicy().validate_submission(candidate, store)
    except ValueError as error:
        assert "rejects closure-tree" in str(error)
    else:
        raise AssertionError("discovered closure should be rejected")


def test_functional_novelty_rejects_disguised_matern() -> None:
    evaluator = FunctionalNoveltyEvaluator(
        SubprocessExecutor(), probe_dimensions=(2,), probe_points=10
    )
    identity = CandidateBundle(
        name="identity_matern",
        contract="input_transform",
        source="def transform_point(x, parameters):\n    return x\n",
        parameters={"base_kernel": {"kind": "matern52", "lengthscale": 0.5}},
    )
    nonlinear = CandidateBundle(
        name="harmonic_warp",
        contract="input_transform",
        source=(
            "import numpy as np\n"
            "def transform_point(x, parameters):\n"
            "    return np.concatenate([x, np.sin(6.283185307*x), x*x])\n"
        ),
        parameters={"base_kernel": {"kind": "matern52", "lengthscale": 0.5}},
    )
    disguised = evaluator.evaluate(identity)
    transformed = evaluator.evaluate(nonlinear)
    assert disguised.distance < 1e-12
    assert not disguised.passed
    assert transformed.distance > evaluator.minimum_distance
    assert transformed.passed


def test_functional_novelty_skips_broken_archived_candidate() -> None:
    broken = CandidateBundle(
        name="broken_history",
        contract="input_transform",
        source="def input_transform(x, parameters):\n    return x\n",
    )
    nonlinear = CandidateBundle(
        name="harmonic_warp",
        contract="input_transform",
        source=(
            "import numpy as np\n"
            "def transform_point(x, parameters):\n"
            "    return np.concatenate([x, np.sin(6.283185307*x), x*x])\n"
        ),
        parameters={"base_kernel": {"kind": "matern52", "lengthscale": 0.5}},
    )
    evaluator = FunctionalNoveltyEvaluator(
        SubprocessExecutor(),
        archive_candidates=lambda: [broken],
        probe_dimensions=(2,),
        probe_points=10,
    )
    result = evaluator.evaluate(nonlinear)
    assert result.passed
    assert result.skipped_candidate_ids == (broken.candidate_id,)


def test_novelty_policy_rejects_comment_only_revision(tmp_path) -> None:
    store = CandidateStore(tmp_path / "archive.sqlite")
    parent = CandidateBundle(
        name="parent",
        contract="input_transform",
        source="def transform_point(x, parameters):\n    return x\n",
        **FORMULATION,
    )
    store.add_candidate(parent)
    store.add_evidence(Verifier(SubprocessExecutor()).verify(parent))
    revision = CandidateBundle(
        name="comment_only_revision",
        contract="input_transform",
        source=(
            "def transform_point(x, parameters):\n"
            '    """Only the docstring and comment changed."""\n'
            "    # This is not a new kernel.\n"
            "    return x\n"
        ),
        parents=(parent.candidate_id,),
        **FORMULATION,
    )
    try:
        NoveltyPolicy().validate_submission(revision, store)
    except ValueError as error:
        assert "parameter-only or metadata-only" in str(error)
    else:
        raise AssertionError("comment-only revision should be rejected")


def test_novelty_policy_allows_independent_root_after_accepted_discovery(tmp_path) -> None:
    store = CandidateStore(tmp_path / "archive.sqlite")
    first_root = CandidateBundle(
        name="first_root",
        contract="input_transform",
        source="def transform_point(x, parameters):\n    return x\n",
        **FORMULATION,
    )
    store.add_candidate(first_root)
    store.add_evidence(Verifier(SubprocessExecutor()).verify(first_root))

    independent_root = CandidateBundle(
        name="independent_root",
        contract="feature_map",
        source="def feature_point(x, parameters):\n    return x * x\n",
        parents=(),
        **FORMULATION,
    )

    NoveltyPolicy().validate_submission(independent_root, store)


def test_novelty_policy_validates_supplied_revision_parent(tmp_path) -> None:
    store = CandidateStore(tmp_path / "archive.sqlite")
    revision = CandidateBundle(
        name="orphan_revision",
        contract="feature_map",
        source="def feature_point(x, parameters):\n    return x * x\n",
        parents=("missing-parent",),
        **FORMULATION,
    )

    try:
        NoveltyPolicy().validate_submission(revision, store)
    except ValueError as error:
        assert "parent candidate not found" in str(error)
    else:
        raise AssertionError("a supplied revision parent must exist")


class _ParameterEvaluator:
    def evaluate(self, candidate, _dataset):
        scale = float(candidate.parameters["scale"])
        return EvaluationRecord(
            candidate_id=candidate.candidate_id,
            score=-((scale - 2.0) ** 2),
            negative_log_likelihood=0.0,
            runtime_seconds=0.0,
            jitter=1e-4,
            condition_number=1.0,
        )


def test_staged_candidate_parameter_optimization_and_submission(tmp_path) -> None:
    store = CandidateStore(tmp_path / "archive.sqlite")
    executor = SubprocessExecutor()
    tools = HarnessTools(
        store,
        Verifier(executor),
        _ParameterEvaluator(),
        Dataset(x=[[0.0], [1.0]], y=[0.0, 1.0]),
        novelty_policy=NoveltyPolicy(),
    )
    proposal = json.loads(tools.propose_formulation(**FORMULATION))
    staged = json.loads(
        tools.stage_candidate(
            name="tunable_features",
            contract="feature_map",
            source="def feature_point(x, parameters):\n    return parameters['scale'] * x\n",
            initial_parameters={"scale": 0.2},
            parameter_space={
                "scale": {"type": "float", "scale": "linear", "lower": 0.1, "upper": 3.0}
            },
            formulation_id=proposal["formulation_id"],
        )
    )
    assert staged["ok"]

    # Stored formulations and drafts remain available after recreating HarnessTools.
    tools = HarnessTools(
        CandidateStore(tmp_path / "archive.sqlite"),
        Verifier(executor),
        _ParameterEvaluator(),
        Dataset(x=[[0.0], [1.0]], y=[0.0, 1.0]),
        novelty_policy=NoveltyPolicy(),
    )
    optimized = json.loads(tools.optimize_parameters(staged["draft_id"], budget=6))
    assert optimized["ok"]
    assert abs(optimized["best_parameters"]["scale"] - 2.0) < 0.4

    submitted = json.loads(
        tools.submit_staged_candidate(staged["draft_id"], optimized["best_trial_id"])
    )
    assert submitted["ok"]
    archived = tools.store.get_candidate(submitted["candidate_id"])
    assert archived is not None
    assert archived.parameters == optimized["best_parameters"]


def test_parameter_optimization_preserves_array_shapes(tmp_path) -> None:
    store = CandidateStore(tmp_path / "archive.sqlite")
    executor = SubprocessExecutor()

    class ArrayEvaluator:
        def evaluate(self, candidate, _dataset):
            weights = candidate.parameters["weights"]
            assert isinstance(weights, list)
            assert len(weights) == 3
            assert all(isinstance(value, float) for value in weights)
            score = -sum(
                (value - target) ** 2
                for value, target in zip(weights, (0.2, 0.4, 0.6), strict=True)
            )
            return EvaluationRecord(
                candidate_id=candidate.candidate_id,
                score=score,
                negative_log_likelihood=0.0,
                runtime_seconds=0.0,
                jitter=1e-4,
                condition_number=1.0,
            )

    tools = HarnessTools(
        store,
        Verifier(executor),
        ArrayEvaluator(),
        Dataset(x=[[0.0], [1.0]], y=[0.0, 1.0]),
        novelty_policy=NoveltyPolicy(),
    )
    proposal = json.loads(tools.propose_formulation(**FORMULATION))
    staged = json.loads(
        tools.stage_candidate(
            name="array_tunable_features",
            contract="feature_map",
            source="def feature_point(x, parameters):\n    return x\n",
            initial_parameters={"weights": [0.1, 0.1, 0.1]},
            parameter_space={"weights": {"type": "float", "lower": 0.0, "upper": 1.0}},
            formulation_id=proposal["formulation_id"],
        )
    )

    optimized = json.loads(tools.optimize_parameters(staged["draft_id"], budget=5))

    assert optimized["ok"]
    assert len(optimized["trials"]) == 5
    assert all(isinstance(trial["parameters"]["weights"], list) for trial in optimized["trials"])
    assert all(len(trial["parameters"]["weights"]) == 3 for trial in optimized["trials"])


def test_zero_regret_reference_does_not_divide_by_zero(tmp_path):
    import math

    tools = _tools(tmp_path)
    candidate = CandidateBundle(
        name="candidate",
        contract="feature_map",
        source="def feature_point(x, parameters): return x",
    )
    baseline = candidate.model_copy(update={"name": "baseline", "origin": CandidateOrigin.BASELINE})
    tools.store.add_candidate(baseline)
    reference = EvaluationRecord(
        candidate_id=baseline.candidate_id,
        score=0.0,
        negative_log_likelihood=0.0,
        runtime_seconds=0.1,
        jitter=0.01,
        condition_number=1.0,
        metadata={"bo": {"regret_auc": 0.0, "final_regret": 0.0}},
    )
    tools.store.add_evaluation(reference)
    result = reference.model_copy(
        update={
            "candidate_id": candidate.candidate_id,
            "metadata": {"tasks": [], "bo": {"regret_auc": 1.0, "final_regret": 1.0}},
        }
    )
    enriched = tools._enrich_evaluation(candidate, result)
    assert math.isfinite(enriched.metadata["selection_score"])
