"""Tests for the evaluation metrics."""

from __future__ import annotations

import numpy as np

from ees.metrics import (
    adjacent_agreement,
    bootstrap_ci,
    exact_agreement,
    mae,
    quadratic_weighted_kappa,
)


def test_perfect_agreement_kappa_is_one() -> None:
    y = [10, 20, 30, 40]
    assert quadratic_weighted_kappa(y, y, 0, 40) == 1.0


def test_mae_and_agreement() -> None:
    y_true = [10, 20, 30]
    y_pred = [11, 20, 32]
    assert abs(mae(y_true, y_pred) - 1.0) < 1e-9
    assert exact_agreement(y_true, y_pred) == 1 / 3
    assert adjacent_agreement(y_true, y_pred) == 2 / 3


def test_kappa_bounds_and_order() -> None:
    y_true = [0, 10, 20, 30, 40]
    close = [1, 11, 20, 31, 39]
    far = [40, 30, 20, 10, 0]
    k_close = quadratic_weighted_kappa(y_true, close, 0, 40)
    k_far = quadratic_weighted_kappa(y_true, far, 0, 40)
    assert k_close > k_far
    assert -1.0 <= k_far <= 1.0


def test_bootstrap_ci_brackets_estimate() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 41, 40).tolist()
    y_pred = [min(40, v + int(rng.integers(-2, 3))) for v in y_true]
    lo, hi = bootstrap_ci(quadratic_weighted_kappa, y_true, y_pred, n_boot=200)
    point = quadratic_weighted_kappa(y_true, y_pred, 0, 40)
    assert lo <= point + 1e-9
    assert hi >= point - 1e-9
