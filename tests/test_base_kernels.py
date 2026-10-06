from __future__ import annotations

import numpy as np
import pytest

from kernaut.contracts.runtime import _evaluate_tree


def _sqdist(x: np.ndarray) -> np.ndarray:
    return np.maximum(
        np.sum(x * x, axis=1)[:, None] + np.sum(x * x, axis=1)[None, :] - 2.0 * x @ x.T, 0.0
    )


@pytest.mark.parametrize(
    "tree",
    [
        {"op": "base", "kind": "rq", "lengthscale": 0.7, "alpha": 1.3},
        {"op": "base", "kind": "periodic", "lengthscale": 0.8, "period": 1.4},
        {
            "op": "base",
            "kind": "spectral_mixture",
            "weights": [1.0, 2.0],
            "means": [0.3, -1.1],
            "scales": [0.5, 2.0],
        },
        {"op": "base", "kind": "ard_matern52", "lengthscales": [0.4, 1.5]},
        {
            "op": "base",
            "kind": "ard_matern32",
            "lengthscales": {"mode": "geometric_ramp", "min": 0.25, "max": 4.0},
        },
        {
            "op": "base",
            "kind": "rff",
            "omega": [[0.5, -1.0], [1.5, 0.3]],
            "bias": [0.1, 2.0],
        },
    ],
)
def test_new_base_kernels_are_symmetric_psd_and_scaled(tree: dict) -> None:
    rng = np.random.default_rng(0)
    x = rng.uniform(size=(12, 2))
    gram = _evaluate_tree(tree, x)
    assert gram.shape == (12, 12)
    assert np.allclose(gram, gram.T)
    assert np.all(np.isfinite(gram))
    assert np.linalg.eigvalsh(gram).min() >= -1e-9


def test_rq_matches_closed_form() -> None:
    x = np.array([[0.0], [0.5], [2.0]])
    lengthscale, alpha = 0.8, 1.25
    gram = _evaluate_tree(
        {"op": "base", "kind": "rq", "variance": 2.0, "lengthscale": lengthscale, "alpha": alpha},
        x,
    )
    expected = 2.0 * (1.0 + _sqdist(x) / (2.0 * alpha * lengthscale**2)) ** (-alpha)
    assert np.allclose(gram, expected)


def test_periodic_matches_closed_form() -> None:
    x = np.linspace(0.0, 1.0, 6).reshape(-1, 1)
    lengthscale, period = 0.9, 1.7
    gram = _evaluate_tree(
        {
            "op": "base",
            "kind": "periodic",
            "variance": 1.5,
            "lengthscale": lengthscale,
            "period": period,
        },
        x,
    )
    distance = np.sqrt(_sqdist(x))
    expected = 1.5 * np.exp(-2.0 * np.sin(np.pi * distance / period) ** 2 / lengthscale**2)
    assert np.allclose(gram, expected)


def test_multivariate_periodic_is_coordinate_wise_product() -> None:
    rng = np.random.default_rng(3)
    x = rng.uniform(size=(6, 3))
    gram = _evaluate_tree(
        {
            "op": "base",
            "kind": "periodic",
            "variance": 2.0,
            "lengthscale": 0.8,
            "period": 1.3,
        },
        x,
    )
    offsets = np.abs(x[:, None, :] - x[None, :, :])
    expected = 2.0 * np.prod(
        np.exp(-2.0 * np.sin(np.pi * offsets / 1.3) ** 2 / 0.8**2),
        axis=-1,
    )
    assert np.allclose(gram, expected)


def test_spectral_mixture_matches_direct_sum() -> None:
    rng = np.random.default_rng(1)
    x = rng.uniform(size=(7, 3))
    weights = np.array([0.6, 1.4])
    means = np.array([0.2, -0.9])
    scales = np.array([0.4, 1.1])
    gram = _evaluate_tree(
        {
            "op": "base",
            "kind": "spectral_mixture",
            "weights": weights.tolist(),
            "means": means.tolist(),
            "scales": scales.tolist(),
        },
        x,
    )
    tau = x[:, None, :] - x[None, :, :]
    expected = sum(
        weight
        * np.exp(-2.0 * np.pi**2 * scale**2 * np.sum(tau**2, axis=-1))
        * np.cos(2.0 * np.pi * np.sum(tau, axis=-1) * mean)
        for weight, mean, scale in zip(weights, means, scales, strict=True)
    )
    assert np.allclose(gram, expected)


def test_ard_matches_isotropic_when_lengthscales_equal() -> None:
    x = np.random.default_rng(2).uniform(size=(6, 4))
    isotropic = _evaluate_tree({"op": "base", "kind": "matern52", "lengthscale": 1.3}, x)
    ard = _evaluate_tree({"op": "base", "kind": "ard_matern52", "lengthscales": [1.3] * 4}, x)
    assert np.allclose(isotropic, ard)


def test_geometric_ramp_adapts_to_dimension() -> None:
    pattern = {"mode": "geometric_ramp", "min": 0.25, "max": 4.0}
    for dimension in (2, 6, 10):
        x = np.random.default_rng(dimension).uniform(size=(5, dimension))
        gram = _evaluate_tree(
            {"op": "base", "kind": "ard_rbf", "variance": 1.0, "lengthscales": pattern}, x
        )
        lengthscales = np.geomspace(0.25, 4.0, dimension)
        scaled = x / lengthscales
        sqdist_ard = _sqdist(scaled)
        assert np.allclose(gram, np.exp(-0.5 * sqdist_ard))


def test_invalid_new_kernel_parameters_rejected() -> None:
    x = np.zeros((3, 2))
    with pytest.raises(ValueError):
        _evaluate_tree({"op": "base", "kind": "rq", "alpha": -1.0}, x)
    with pytest.raises(ValueError):
        _evaluate_tree({"op": "base", "kind": "spectral_mixture", "weights": []}, x)
    with pytest.raises(ValueError):
        _evaluate_tree(
            {"op": "base", "kind": "ard_rbf", "lengthscales": [-1.0, 1.0]},
            x,
        )
