"""Classification metrics used by every protocol (macro-F1, top-k, confusion matrix)."""

from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Rows are true classes, columns are predicted classes."""
    flat = y_true * n_classes + y_pred
    return np.bincount(flat, minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def _f1_scores(cm: np.ndarray) -> np.ndarray:
    true_pos = np.diag(cm).astype(float)
    denominator = cm.sum(axis=0) + cm.sum(axis=1)  # 2TP + FP + FN once TP is counted twice
    return np.divide(2 * true_pos, denominator, out=np.zeros(len(cm)), where=denominator > 0)


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    return _f1_scores(confusion_matrix(y_true, y_pred, n_classes))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Mean per-letter F1, so every letter counts equally.

    Only letters that occur in the labels or the predictions are averaged. A letter that is
    missing from a test set therefore does not drag the score down as a phantom zero.
    """
    cm = confusion_matrix(y_true, y_pred, n_classes)
    seen = (cm.sum(axis=0) + cm.sum(axis=1)) > 0
    return float(_f1_scores(cm)[seen].mean()) if seen.any() else 0.0


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def top_k_accuracy(proba: np.ndarray, y_true: np.ndarray, k: int) -> float:
    """Fraction of samples whose true class is among the ``k`` highest-scoring classes.

    Ties (common for kNN and forests, which output many zeros) go to the lower class index.
    """
    best = np.argsort(-proba, axis=1, kind="stable")[:, :k]
    return float(np.mean((best == y_true[:, None]).any(axis=1)))


def most_confused_pairs(cm: np.ndarray, top: int = 5) -> list[tuple[int, int, int]]:
    """Unordered class pairs with the most mutual confusions: ``(i, j, count)``, ``i < j``."""
    n = cm.shape[0]
    pairs = [(i, j, int(cm[i, j] + cm[j, i])) for i in range(n) for j in range(i + 1, n)]
    pairs = [p for p in pairs if p[2] > 0]
    return sorted(pairs, key=lambda p: p[2], reverse=True)[:top]
