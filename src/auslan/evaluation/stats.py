"""Significance tests and bootstrap intervals from the report's Section 6.4."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.stats import binomtest

from auslan.evaluation.metrics import macro_f1


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> float:
    """Exact two-sided McNemar p-value on paired correct/incorrect predictions."""
    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    discordant = a_only + b_only
    if discordant == 0:
        return 1.0
    return float(binomtest(min(a_only, b_only), discordant, 0.5).pvalue)


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, returned in the input order."""
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, (m - rank) * p_values[idx])
        adjusted[idx] = min(1.0, running_max)
    return adjusted.tolist()


def _stratified_resamples(y_true: np.ndarray, n_boot: int, seed: int):
    """Yield index arrays resampled with replacement within each letter."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y_true == c) for c in np.unique(y_true)]
    for _ in range(n_boot):
        yield np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])


def bootstrap_statistic(
    y_true: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    n_boot: int,
    seed: int,
    alpha: float,
) -> tuple[float, float]:
    draws = [statistic(idx) for idx in _stratified_resamples(y_true, n_boot, seed)]
    low, high = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile CI of macro-F1, resampled within each letter."""
    return bootstrap_statistic(
        y_true, lambda i: macro_f1(y_true[i], y_pred[i], n_classes), n_boot, seed, alpha
    )


def paired_bootstrap_diff(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n_classes: int,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile CI of macro-F1(A) - macro-F1(B) using identical resamples for both."""

    def diff(i: np.ndarray) -> float:
        return macro_f1(y_true[i], pred_a[i], n_classes) - macro_f1(y_true[i], pred_b[i], n_classes)

    return bootstrap_statistic(y_true, diff, n_boot, seed, alpha)
