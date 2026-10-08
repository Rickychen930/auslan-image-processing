import numpy as np
import pytest

from auslan.evaluation.metrics import (
    accuracy,
    confusion_matrix,
    macro_f1,
    most_confused_pairs,
    top_k_accuracy,
)
from auslan.evaluation.stats import (
    bootstrap_ci,
    holm_adjust,
    mcnemar_exact,
    paired_bootstrap_diff,
)


def test_macro_f1_perfect_and_worst():
    y = np.array([0, 1, 2, 0, 1, 2])
    assert macro_f1(y, y, 3) == pytest.approx(1.0)
    assert macro_f1(y, (y + 1) % 3, 3) == pytest.approx(0.0)


def test_macro_f1_counts_each_class_equally():
    y_true = np.array([0] * 90 + [1] * 10)
    y_pred = np.zeros(100, dtype=int)  # always predicts the majority class
    assert accuracy(y_true, y_pred) == pytest.approx(0.9)
    assert macro_f1(y_true, y_pred, 2) == pytest.approx((2 * 0.9 / 1.9) / 2)


def test_macro_f1_ignores_letters_that_never_occur():
    y = np.array([0, 1, 2])
    assert macro_f1(y, y, 26) == pytest.approx(1.0)  # 23 absent letters are not phantom zeros


def test_macro_f1_still_penalises_a_letter_that_is_predicted_but_absent():
    assert macro_f1(np.array([0, 0]), np.array([0, 1]), 3) == pytest.approx((2 / 3 + 0) / 2)


def test_top_k_accuracy():
    proba = np.array([[0.1, 0.5, 0.4], [0.6, 0.3, 0.1]])
    y = np.array([2, 1])
    assert top_k_accuracy(proba, y, 1) == pytest.approx(0.0)
    assert top_k_accuracy(proba, y, 2) == pytest.approx(1.0)


def test_top_k_ties_go_to_the_lower_class_index():
    flat = np.zeros((2, 5))
    assert top_k_accuracy(flat, np.array([1, 4]), 3) == pytest.approx(0.5)


def test_confusion_matrix_and_pairs():
    y_true = np.array([0, 0, 1, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 2])
    cm = confusion_matrix(y_true, y_pred, 3)
    assert cm.tolist() == [[1, 1, 0], [1, 1, 0], [0, 0, 1]]
    pairs = most_confused_pairs(cm, top=1)
    assert pairs[0][2] == 2 and {pairs[0][0], pairs[0][1]} == {0, 1}


def test_mcnemar_identical_classifiers_gives_p_one():
    correct = np.array([True, False, True, True])
    assert mcnemar_exact(correct, correct) == pytest.approx(1.0)


def test_mcnemar_detects_one_sided_difference():
    a = np.array([True] * 30 + [False] * 10)
    b = np.array([False] * 30 + [False] * 10)
    assert mcnemar_exact(a, b) < 0.001


def test_holm_adjust_is_monotone_and_bounded():
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])
    assert all(0 <= p <= 1 for p in adjusted)


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 300)
    pred = np.where(rng.random(300) < 0.8, y, (y + 1) % 3)
    low, high = bootstrap_ci(y, pred, 3, n_boot=200, seed=0)
    assert low < macro_f1(y, pred, 3) < high


def test_paired_bootstrap_diff_positive_when_a_better():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 300)
    good = np.where(rng.random(300) < 0.95, y, (y + 1) % 3)
    bad = np.where(rng.random(300) < 0.5, y, (y + 1) % 3)
    low, high = paired_bootstrap_diff(y, good, bad, 3, n_boot=200, seed=0)
    assert low > 0 and high > low
