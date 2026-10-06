from __future__ import annotations

import numpy as np
import pytest

from kernaut.config import ExecutionConfig
from kernaut.contracts import validate_source_shape
from kernaut.execution import ExecutionFailure, SubprocessExecutor
from kernaut.models import CandidateBundle, ContractKind, EvidenceTier
from kernaut.verification import VerificationPolicy, Verifier

SOURCES = {
    ContractKind.UNVERIFIED: """import numpy as np
def kernel_matrix(x, parameters):
    return np.exp(-(x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2)
""",
    ContractKind.FEATURE_MAP: """import numpy as np
def feature_point(x, parameters):
    return np.concatenate([x, np.sin(x)])
""",
    ContractKind.RESIDUAL_FEATURE_MAP: """import numpy as np
def feature_point(x, parameters):
    return np.concatenate([x, np.sin(x)])
""",
    ContractKind.ADDITIVE_FEATURE_MAP: """import numpy as np
def coordinate_feature(value, parameters):
    return np.array([value, np.sin(value)])
""",
    ContractKind.SPECTRAL: """import numpy as np
def spectral_components(input_dimension, parameters):
    frequencies = np.zeros((2, input_dimension))
    frequencies[0, :] = 1.0 / np.sqrt(input_dimension)
    frequencies[1, :] = 2.0 / np.sqrt(input_dimension)
    return frequencies, np.array([1.0, -2.0])
""",
    ContractKind.INPUT_TRANSFORM: """import numpy as np
def transform_point(x, parameters):
    return np.concatenate([x, np.sin(2.0 * np.pi * x), x * x])
""",
    ContractKind.RESIDUAL_INPUT_TRANSFORM: """import numpy as np
def transform_point(x, parameters):
    return np.concatenate([x, np.sin(2.0 * np.pi * x), x * x])
""",
    ContractKind.CLOSURE: """def closure_tree(parameters):
    return {'op': 'sum', 'children': [
        {'op': 'base', 'kind': 'rbf', 'lengthscale': 1.2},
        {'op': 'scale', 'weight': 0.2, 'child': {'op': 'base', 'kind': 'linear'}}
    ]}
""",
}


@pytest.fixture
def executor() -> SubprocessExecutor:
    return SubprocessExecutor(ExecutionConfig(timeout_seconds=10, memory_mb=1024, cpu_seconds=8))


@pytest.mark.parametrize("contract", list(ContractKind))
def test_every_contract_constructs_psd_gram(
    executor: SubprocessExecutor, contract: ContractKind
) -> None:
    candidate = CandidateBundle(
        name=f"candidate_{contract.value}", contract=contract, source=SOURCES[contract]
    )
    gram = executor.gram(candidate, np.array([[0.0, 1.0], [1.0, 0.0], [0.5, -0.2]])).as_array()
    assert gram.shape == (3, 3)
    assert np.allclose(gram, gram.T)
    assert np.linalg.eigvalsh(gram).min() >= -1e-9


def test_unverified_candidate_stays_empirical(executor: SubprocessExecutor) -> None:
    candidate = CandidateBundle(
        name="empirical_only",
        contract="unverified",
        source=SOURCES[ContractKind.UNVERIFIED],
    )
    policy = VerificationPolicy(trials=2, points_per_trial=5, input_dimension=2)
    evidence = Verifier(executor, policy).verify(candidate)
    assert evidence.tier == EvidenceTier.EMPIRICAL
    assert not evidence.accepted


def test_input_transform_uses_trusted_base_kernel(executor: SubprocessExecutor) -> None:
    source = """import numpy as np
def transform_point(x, parameters):
    return np.concatenate([x, np.sin(x)])
"""
    candidate = CandidateBundle(
        name="warped_rbf",
        contract="input_transform",
        source=source,
        parameters={"base_kernel": {"kind": "rbf", "lengthscale": 0.7}},
    )
    x = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, -0.2]])
    transformed = np.concatenate([x, np.sin(x)], axis=1)
    squared_distance = np.sum((transformed[:, None, :] - transformed[None, :, :]) ** 2, axis=2)
    expected = np.exp(-0.5 * squared_distance / 0.7**2)
    assert np.allclose(executor.gram(candidate, x).as_array(), expected)


def test_dimension_aware_spectral_contract_supports_multiple_dimensions(
    executor: SubprocessExecutor,
) -> None:
    candidate = CandidateBundle(
        name="dimension_aware_spectral",
        contract="spectral",
        source=SOURCES[ContractKind.SPECTRAL],
    )
    policy = VerificationPolicy(
        trials=1, contract_trials=1, points_per_trial=5, input_dimensions=(2, 5, 6, 8)
    )
    evidence = Verifier(executor, policy).verify(candidate)
    assert evidence.accepted
    assert evidence.tier == EvidenceTier.CONTRACT_CERTIFIED


def test_residual_feature_map_retains_trusted_base(executor: SubprocessExecutor) -> None:
    source = "def feature_point(x, parameters):\n    return x\n"
    x = np.array([[0.0], [1.0]])
    candidate = CandidateBundle(
        name="residual_features",
        contract="residual_feature_map",
        source=source,
        parameters={
            "base_kernel": {"kind": "rbf", "lengthscale": 1.0},
            "residual_weight": -2.0,
        },
    )
    expected_base = np.exp(-0.5 * np.array([[0.0, 1.0], [1.0, 0.0]]))
    expected = expected_base + 4.0 * (x @ x.T)
    assert np.allclose(executor.gram(candidate, x).as_array(), expected)


def test_verifier_awards_contract_certified_tier(executor: SubprocessExecutor) -> None:
    candidate = CandidateBundle(
        name="features", contract="feature_map", source=SOURCES[ContractKind.FEATURE_MAP]
    )
    policy = VerificationPolicy(trials=2, points_per_trial=5, input_dimension=2, seed=4)
    evidence = Verifier(executor, policy).verify(candidate)
    assert evidence.accepted
    assert evidence.tier == EvidenceTier.CONTRACT_CERTIFIED
    assert evidence.checks[-1].name == "contract_interpreter"
    assert evidence.checks[-1].passed
    assert next(check for check in evidence.checks if check.name == "contract_conformance").passed


def test_pointwise_contract_has_no_batch_interface(executor: SubprocessExecutor) -> None:
    candidate = CandidateBundle(
        name="batch_dependent",
        contract="feature_map",
        source="""import numpy as np
def feature_map(x, parameters):
    return x + np.mean(x, axis=0, keepdims=True)
""",
    )
    policy = VerificationPolicy(
        trials=1, contract_trials=1, points_per_trial=6, input_dimension=2, seed=3
    )
    evidence = Verifier(executor, policy).verify(candidate)
    assert not evidence.accepted
    assert evidence.tier is None
    assert "feature_point" in evidence.checks[0].detail


def test_certified_source_rejects_state_randomness_and_mutable_defaults() -> None:
    source = """import numpy as np
state = []
def feature_point(x, parameters, cache=[]):
    return x + np.random.normal(size=x.shape)
"""
    errors = validate_source_shape(source, ContractKind.FEATURE_MAP)
    assert "certified programs cannot define module-level state" in errors
    assert "stateful or nondeterministic attribute 'random' is not allowed" in errors
    assert "'feature_point' must have the exact signature (x, parameters)" in errors


def test_certified_source_rejects_dunder_reflection() -> None:
    source = """import numpy as np
def feature_point(x, parameters):
    random_module = np.__dict__["random"]
    return x + random_module.normal(size=x.shape)
"""
    errors = validate_source_shape(source, ContractKind.FEATURE_MAP)
    assert "dunder reflection is not allowed in certified programs" in errors


def test_static_gate_rejects_forbidden_import_and_missing_entrypoint() -> None:
    errors = validate_source_shape("import os\ndef kernel(x): return x", ContractKind.FEATURE_MAP)
    assert "required function 'feature_point' is missing" in errors
    assert "import 'os' is not allowed" in errors


def test_worker_rejects_runtime_forbidden_import(executor: SubprocessExecutor) -> None:
    source = "def feature_point(x, parameters):\n    import pathlib\n    return x\n"
    candidate = CandidateBundle(name="bad_import", contract="feature_map", source=source)
    with pytest.raises(ExecutionFailure, match="not allowed"):
        executor.gram(candidate, np.ones((2, 1)))
