from __future__ import annotations

import ast
import copy
from collections.abc import Callable
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from kernaut.models import ContractKind

ENTRYPOINTS = {
    ContractKind.UNVERIFIED: "kernel_matrix",
    ContractKind.FEATURE_MAP: "feature_point",
    ContractKind.RESIDUAL_FEATURE_MAP: "feature_point",
    ContractKind.ADDITIVE_FEATURE_MAP: "coordinate_feature",
    ContractKind.SPECTRAL: "spectral_components",
    ContractKind.INPUT_TRANSFORM: "transform_point",
    ContractKind.RESIDUAL_INPUT_TRANSFORM: "transform_point",
    ContractKind.CLOSURE: "closure_tree",
}
ENTRYPOINT_ARGUMENTS = {
    ContractKind.UNVERIFIED: ["x", "parameters"],
    ContractKind.FEATURE_MAP: ["x", "parameters"],
    ContractKind.RESIDUAL_FEATURE_MAP: ["x", "parameters"],
    ContractKind.ADDITIVE_FEATURE_MAP: ["value", "parameters"],
    ContractKind.SPECTRAL: ["input_dimension", "parameters"],
    ContractKind.INPUT_TRANSFORM: ["x", "parameters"],
    ContractKind.RESIDUAL_INPUT_TRANSFORM: ["x", "parameters"],
    ContractKind.CLOSURE: ["parameters"],
}

ALLOWED_IMPORT_ROOTS = {"math", "numpy"}
DISALLOWED_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "dir",
    "delattr",
    "eval",
    "exec",
    "getattr",
    "globals",
    "id",
    "input",
    "locals",
    "open",
    "setattr",
    "type",
    "vars",
}
STATEFUL_NUMPY_ATTRIBUTES = {
    "random",
    "set_string_function",
    "setbufsize",
    "seterr",
    "seterrcall",
    "set_printoptions",
}


def required_entrypoint(contract: ContractKind) -> str:
    return ENTRYPOINTS[contract]


def validate_source_shape(source: str, contract: ContractKind) -> list[str]:
    """Check the restricted certificate interface before isolated execution.

    Certified point functions have no module-level state, decorators, closures, mutable
    defaults, reflection, or randomness. The trusted interpreter supplies one copied point
    and one copied parameter object per call and owns every PSD-preserving construction.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return [f"syntax error: {error.msg} at line {error.lineno}"]
    errors: list[str] = []
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    expected = required_entrypoint(contract)
    if expected not in functions:
        errors.append(f"required function '{expected}' is missing")
    for statement in tree.body:
        if contract != ContractKind.UNVERIFIED and not isinstance(
            statement, (ast.Import, ast.ImportFrom, ast.FunctionDef)
        ):
            errors.append("certified programs cannot define module-level state")
    entrypoint = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == expected),
        None,
    )
    if entrypoint is not None:
        positional = [*entrypoint.args.posonlyargs, *entrypoint.args.args]
        required_arguments = ENTRYPOINT_ARGUMENTS[contract]
        if (
            [argument.arg for argument in positional] != required_arguments
            or entrypoint.args.vararg
            or entrypoint.args.kwarg
            or entrypoint.args.kwonlyargs
            or entrypoint.args.defaults
        ):
            signature = ", ".join(required_arguments)
            errors.append(f"'{expected}' must have the exact signature ({signature})")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            for name in names:
                if name.split(".")[0] not in ALLOWED_IMPORT_ROOTS:
                    errors.append(f"import '{name}' is not allowed")
                if contract != ContractKind.UNVERIFIED and "random" in name.split("."):
                    errors.append("randomness is not allowed in certified programs")
            if contract != ContractKind.UNVERIFIED:
                for alias in node.names:
                    if alias.name == "random":
                        errors.append("randomness is not allowed in certified programs")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in DISALLOWED_CALLS
        ):
            errors.append(f"call to '{node.func.id}' is not allowed")
        if contract != ContractKind.UNVERIFIED:
            if isinstance(node, (ast.Global, ast.Nonlocal)):
                errors.append("global and nonlocal state are not allowed in certified programs")
            if isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef)):
                errors.append("classes and async functions are not allowed in certified programs")
            if isinstance(node, ast.FunctionDef) and (
                node.decorator_list or node.args.defaults or node.args.kw_defaults
            ):
                errors.append(
                    "decorators and default arguments are not allowed in certified programs"
                )
            if isinstance(node, ast.Attribute) and node.attr in STATEFUL_NUMPY_ATTRIBUTES:
                errors.append(
                    f"stateful or nondeterministic attribute '{node.attr}' is not allowed"
                )
            if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                errors.append("dunder reflection is not allowed in certified programs")
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(isinstance(target, ast.Attribute) for target in targets):
                    errors.append("attribute mutation is not allowed in certified programs")
    return sorted(set(errors))


def build_gram_matrix(
    module: ModuleType,
    contract: ContractKind,
    x: NDArray[np.float64],
    parameters: dict[str, Any],
) -> NDArray[np.float64]:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("inputs must have shape [n, d]")
    if contract == ContractKind.UNVERIFIED:
        return _matrix(module.kernel_matrix(x, parameters), rows=len(x), label="kernel matrix")
    if contract in {ContractKind.FEATURE_MAP, ContractKind.RESIDUAL_FEATURE_MAP}:
        phi = _stack_pointwise(module.feature_point, x, parameters, label="feature point")
        feature_gram = phi @ phi.T
        if contract == ContractKind.FEATURE_MAP:
            return feature_gram
        return _base_gram(x, parameters) + _residual_weight(parameters) * feature_gram
    if contract == ContractKind.ADDITIVE_FEATURE_MAP:
        result = np.zeros((len(x), len(x)), dtype=np.float64)
        for dimension in range(x.shape[1]):
            values = x[:, dimension][:, None]
            phi = _stack_pointwise(
                module.coordinate_feature,
                values,
                parameters,
                label="coordinate feature",
                scalar_input=True,
            )
            result += phi @ phi.T
        return result / max(x.shape[1], 1)
    if contract in {ContractKind.INPUT_TRANSFORM, ContractKind.RESIDUAL_INPUT_TRANSFORM}:
        transformed = _stack_pointwise(
            module.transform_point, x, parameters, label="transformed point"
        )
        transformed_gram = _base_gram(transformed, parameters)
        if contract == ContractKind.INPUT_TRANSFORM:
            return transformed_gram
        return _base_gram(x, parameters, key="residual_base_kernel") + (
            _residual_weight(parameters) * transformed_gram
        )
    if contract == ContractKind.SPECTRAL:
        frequencies, raw_weights = module.spectral_components(x.shape[1], copy.deepcopy(parameters))
        frequencies = np.asarray(frequencies, dtype=np.float64)
        raw_weights = np.asarray(raw_weights, dtype=np.float64).reshape(-1)
        if frequencies.ndim != 2 or frequencies.shape[1] != x.shape[1]:
            raise ValueError("frequencies must have shape [m, input_dimension]")
        if len(raw_weights) != len(frequencies):
            raise ValueError("one spectral weight is required per frequency")
        weights = np.square(raw_weights) / max(len(raw_weights), 1)
        phase = x @ frequencies.T
        features = np.concatenate([np.cos(phase), np.sin(phase)], axis=1)
        scaled = features * np.sqrt(np.concatenate([weights, weights]))
        return scaled @ scaled.T
    if contract == ContractKind.CLOSURE:
        tree = module.closure_tree(parameters)
        return _evaluate_tree(tree, x)
    raise ValueError(f"unsupported contract: {contract}")


def _stack_pointwise(
    function: Callable[[Any, dict[str, Any]], Any],
    x: NDArray[np.float64],
    parameters: dict[str, Any],
    *,
    label: str,
    scalar_input: bool = False,
) -> NDArray[np.float64]:
    vectors: list[NDArray[np.float64]] = []
    width: int | None = None
    for row in x:
        if scalar_input:
            point: Any = float(row[0])
        else:
            point = np.array(row, dtype=np.float64, copy=True)
            point.setflags(write=False)
        value = np.asarray(function(point, copy.deepcopy(parameters)), dtype=np.float64)
        if value.ndim != 1 or value.size == 0:
            raise ValueError(f"{label} must return a nonempty vector")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{label} contains non-finite values")
        if width is None:
            width = value.size
        elif value.size != width:
            raise ValueError(f"{label} dimension must be identical for every input")
        vectors.append(value)
    if not vectors:
        raise ValueError("at least one input point is required")
    return np.stack(vectors)


def _base_gram(
    x: NDArray[np.float64], parameters: dict[str, Any], *, key: str = "base_kernel"
) -> NDArray[np.float64]:
    default = {"kind": "matern52", "lengthscale": 0.5}
    base = parameters.get(key, default)
    if not isinstance(base, dict):
        raise ValueError(f"{key} must be a closure base-node dictionary")
    base_node = dict(base)
    base_node["op"] = "base"
    return _evaluate_tree(base_node, x)


def _residual_weight(parameters: dict[str, Any]) -> float:
    raw_weight = float(parameters.get("residual_weight", 1.0))
    if not np.isfinite(raw_weight):
        raise ValueError("residual_weight must be finite")
    return raw_weight**2


def _matrix(value: Any, *, rows: int, label: str) -> NDArray[np.float64]:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != rows or matrix.shape[1] == 0:
        raise ValueError(f"{label} must have shape [n, m] with m > 0")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} contains non-finite values")
    return matrix


def _vector(value: Any, *, label: str) -> NDArray[np.float64]:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size == 0:
        raise ValueError(f"{label} must be a nonempty vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} contains non-finite values")
    return vector


def _evaluate_tree(node: Any, x: NDArray[np.float64]) -> NDArray[np.float64]:
    if not isinstance(node, dict) or "op" not in node:
        raise ValueError("closure nodes must be dictionaries containing 'op'")
    op = node["op"]
    if op in {"sum", "product"}:
        children = node.get("children")
        if not isinstance(children, list) or len(children) < 2:
            raise ValueError(f"'{op}' requires at least two children")
        matrices = [_evaluate_tree(child, x) for child in children]
        reducer: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        reducer = np.add if op == "sum" else np.multiply
        result = matrices[0]
        for matrix in matrices[1:]:
            result = reducer(result, matrix)
        return result
    if op == "scale":
        weight = float(node.get("weight", -1))
        if weight < 0 or not np.isfinite(weight):
            raise ValueError("closure scale must be finite and nonnegative")
        return weight * _evaluate_tree(node.get("child"), x)
    if op == "pullback":
        projection = np.asarray(node.get("matrix"), dtype=np.float64)
        if projection.ndim != 2 or projection.shape[0] != x.shape[1]:
            raise ValueError("pullback matrix must have shape [input_dimension, output_dimension]")
        return _evaluate_tree(node.get("child"), x @ projection)
    if op != "base":
        raise ValueError(f"unsupported closure operation: {op!r}")
    kind = node.get("kind")
    variance = float(node.get("variance", 1.0))
    if variance < 0 or not np.isfinite(variance):
        raise ValueError("base variance must be finite and nonnegative")
    if kind == "linear":
        return variance * (x @ x.T)
    lengthscale = float(node.get("lengthscale", 1.0))
    if lengthscale <= 0 or not np.isfinite(lengthscale):
        raise ValueError("lengthscale must be finite and positive")
    sqdist = np.maximum(
        np.sum(x * x, axis=1)[:, None] + np.sum(x * x, axis=1)[None, :] - 2 * x @ x.T,
        0,
    )
    if kind == "rbf":
        return variance * np.exp(-0.5 * sqdist / lengthscale**2)
    distance = np.sqrt(sqdist + 1e-30)
    if kind == "matern32":
        z = np.sqrt(3.0) * distance / lengthscale
        return variance * (1 + z) * np.exp(-z)
    if kind == "matern52":
        z = np.sqrt(5.0) * distance / lengthscale
        return variance * (1 + z + z**2 / 3) * np.exp(-z)
    if kind == "rq":
        alpha = float(node.get("alpha", 1.0))
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("rq alpha must be finite and positive")
        return variance * (1.0 + sqdist / (2.0 * alpha * lengthscale**2)) ** (-alpha)
    if kind == "periodic":
        period = float(node.get("period", 1.0))
        if not np.isfinite(period) or period <= 0:
            raise ValueError("period must be finite and positive")
        # Coordinate-wise MacKay periodic kernel: a product of one-dimensional periodic
        # kernels, which preserves positive semidefiniteness in any input dimension. The
        # naive Euclidean-distance variant violates PSD for d >= 2.
        offsets = np.abs(x[:, None, :] - x[None, :, :])
        phase = np.sin(np.pi * offsets / period) ** 2
        return variance * np.exp(-2.0 * np.sum(phase, axis=-1) / lengthscale**2)
    if kind == "spectral_mixture":
        weights = _vector(node.get("weights"), label="spectral mixture weights")
        means_raw = np.asarray(node.get("means"), dtype=np.float64)
        scales_raw = np.asarray(node.get("scales"), dtype=np.float64)
        if not np.all(np.isfinite(means_raw)) or not np.all(np.isfinite(scales_raw)):
            raise ValueError("spectral mixture parameters contain non-finite values")
        means = means_raw.reshape(weights.size, -1)
        scales = scales_raw.reshape(weights.size, -1)
        if means.shape[0] != weights.size or scales.shape[0] != weights.size:
            raise ValueError("one spectral mixture mean and scale are required per component")
        if means.shape[1] not in {1, x.shape[1]} or scales.shape[1] not in {1, x.shape[1]}:
            raise ValueError("spectral mixture means/scales must broadcast to input dimension")
        means = np.broadcast_to(means, (weights.size, x.shape[1]))
        scales = np.broadcast_to(scales, (weights.size, x.shape[1]))
        if np.any(scales < 0):
            raise ValueError("spectral mixture scales must be nonnegative")
        if float(weights.sum()) <= 0:
            raise ValueError("spectral mixture weights must have positive mass")
        tau = x[:, None, :] - x[None, :, :]
        gram = np.zeros_like(sqdist)
        for weight, mean, scale in zip(weights, means, scales, strict=True):
            spectral_distance = np.sum(tau**2 * scale**2, axis=-1)
            gram += weight * (
                np.exp(-2.0 * np.pi**2 * spectral_distance) * np.cos(2.0 * np.pi * tau @ mean)
            )
        return variance * gram
    if kind == "rff":
        # Rahimi-Recht random Fourier features. Weights are drawn deterministically from a
        # stored seed so the same candidate reproduces the same Gram at any input
        # dimension; explicit omega/bias matrices are also accepted.
        seed = int(node.get("seed", 0))
        features = int(node.get("features", 64))
        if features < 1:
            raise ValueError("rff requires at least one feature")
        explicit_omega = node.get("omega")
        if explicit_omega is not None:
            omega = np.asarray(explicit_omega, dtype=np.float64)
            bias = np.asarray(node.get("bias"), dtype=np.float64).reshape(-1)
            if omega.ndim != 2 or omega.shape[1] != x.shape[1]:
                raise ValueError("rff omega must have shape [features, input_dimension]")
            if bias.shape[0] != omega.shape[0]:
                raise ValueError("one rff bias is required per feature")
        else:
            lengthscale_rff = float(node.get("lengthscale", 1.0))
            if not np.isfinite(lengthscale_rff) or lengthscale_rff <= 0:
                raise ValueError("rff lengthscale must be finite and positive")
            rng_rff = np.random.default_rng(seed)
            omega = rng_rff.normal(size=(features, x.shape[1])) / lengthscale_rff
            bias = rng_rff.uniform(0.0, 2.0 * np.pi, size=features)
        if not np.all(np.isfinite(omega)) or not np.all(np.isfinite(bias)):
            raise ValueError("rff parameters contain non-finite values")
        phi = np.sqrt(2.0 / omega.shape[0]) * np.cos(x @ omega.T + bias)
        return variance * (phi @ phi.T)
    ard_kinds = {"ard_rbf": None, "ard_matern32": 3.0, "ard_matern52": 5.0}
    if kind in ard_kinds:
        raw = node.get("lengthscales")
        if isinstance(raw, dict):
            mode = raw.get("mode")
            minimum = float(raw.get("min", -1))
            maximum = float(raw.get("max", -1))
            if mode != "geometric_ramp" or not (
                np.isfinite(minimum) and np.isfinite(maximum) and minimum > 0 and maximum > 0
            ):
                raise ValueError(
                    "ard lengthscales pattern requires mode 'geometric_ramp' with positive "
                    "'min' and 'max'"
                )
            lengthscales = np.geomspace(minimum, maximum, x.shape[1])
        else:
            lengthscales = np.asarray(raw, dtype=np.float64).reshape(-1)
            if lengthscales.size == 1:
                lengthscales = np.full(x.shape[1], float(lengthscales[0]))
            elif lengthscales.shape[0] != x.shape[1]:
                raise ValueError("one ard lengthscale is required per input dimension")
        if not np.all(np.isfinite(lengthscales)) or np.any(lengthscales <= 0):
            raise ValueError("ard lengthscales must be finite and positive")
        scaled = x / lengthscales
        sqdist_ard = np.maximum(
            np.sum(scaled * scaled, axis=1)[:, None]
            + np.sum(scaled * scaled, axis=1)[None, :]
            - 2.0 * scaled @ scaled.T,
            0.0,
        )
        nu = ard_kinds[kind]
        if nu is None:
            return variance * np.exp(-0.5 * sqdist_ard)
        distance_ard = np.sqrt(sqdist_ard + 1e-30)
        if nu == 3.0:
            z = np.sqrt(3.0) * distance_ard
            return variance * (1 + z) * np.exp(-z)
        z = np.sqrt(5.0) * distance_ard
        return variance * (1 + z + z**2 / 3) * np.exp(-z)
    raise ValueError(f"unsupported base kernel: {kind!r}")
