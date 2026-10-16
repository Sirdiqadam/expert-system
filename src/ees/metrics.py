"""Evaluation metrics: QWK, MAE, exact/adjacent agreement, bootstrap CIs."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def _as_int_array(values: Sequence[float]) -> np.ndarray:
    return np.rint(np.asarray(values, dtype=float)).astype(int)


def confusion_matrix(
    y_true: Sequence[float], y_pred: Sequence[float], min_rating: int, max_rating: int
) -> np.ndarray:
    """Confusion matrix with rows=true, cols=predicted over the rating range."""

    a = _as_int_array(y_true)
    b = _as_int_array(y_pred)
    n = max_rating - min_rating + 1
    matrix = np.zeros((n, n), dtype=float)
    for t, p in zip(a, b):
        ti = min(max(t - min_rating, 0), n - 1)
        pi = min(max(p - min_rating, 0), n - 1)
        matrix[ti, pi] += 1
    return matrix


def quadratic_weighted_kappa(
    y_true: Sequence[float], y_pred: Sequence[float], min_rating: int = 0, max_rating: int = 40
) -> float:
    """Quadratic weighted kappa (the primary agreement metric)."""

    a = _as_int_array(y_true)
    b = _as_int_array(y_pred)
    if len(a) == 0:
        return float("nan")
    lo = min(int(min(a.min(), b.min())), min_rating)
    hi = max(int(max(a.max(), b.max())), max_rating)
    n = hi - lo + 1
    if n < 2:
        return 1.0 if np.array_equal(a, b) else 0.0
    O = confusion_matrix(a, b, lo, hi)
    weights = np.zeros_like(O)
    for i in range(n):
        for j in range(n):
            weights[i, j] = ((i - j) ** 2) / ((n - 1) ** 2)
    hist_true = O.sum(axis=1)
    hist_pred = O.sum(axis=0)
    total = O.sum()
    if total == 0:
        return float("nan")
    expected = np.outer(hist_true, hist_pred) / total
    denom = float((weights * expected).sum())
    if denom == 0:
        return 1.0
    return float(1.0 - (weights * O).sum() / denom)


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Mean absolute error."""

    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if len(a) == 0:
        return float("nan")
    return float(np.mean(np.abs(a - b)))


def exact_agreement(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    a = _as_int_array(y_true)
    b = _as_int_array(y_pred)
    if len(a) == 0:
        return float("nan")
    return float(np.mean(a == b))


def adjacent_agreement(y_true: Sequence[float], y_pred: Sequence[float], tolerance: int = 1) -> float:
    a = _as_int_array(y_true)
    b = _as_int_array(y_pred)
    if len(a) == 0:
        return float("nan")
    return float(np.mean(np.abs(a - b) <= tolerance))


def bootstrap_ci(
    metric: Callable[[Sequence[float], Sequence[float]], float],
    y_true: Sequence[float],
    y_pred: Sequence[float],
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for a metric."""

    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if len(a) < 2:
        value = metric(a, b)
        return (value, value)
    rng = np.random.default_rng(seed)
    stats: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(a), len(a))
        value = metric(a[idx], b[idx])
        if not np.isnan(value):
            stats.append(value)
    if not stats:
        return (float("nan"), float("nan"))
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return (lo, hi)
