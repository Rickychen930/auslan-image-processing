"""Classification metrics used by every protocol (macro-F1, top-k, confusion matrix)."""

from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Rows are true classes, columns are predicted classes."""
    flat = np.asarray(y_true) * n_classes + np.asarray(y_pred)
    return np.bincount(flat, minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, n_classes)
    true_pos = np.diag(cm).astype(float)
    denominator = cm.sum(axis=0) + cm.sum(axis=1)  # 2TP + FP + FN once TP is counted twice
    return np.divide(2 * true_pos, denominator, out=np.zeros(n_classes), where=denominator > 0)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Mean per-letter F1, so every letter counts equally."""
    return float(per_class_f1(y_true, y_pred, n_classes).mean())


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def top_k_accuracy(proba: np.ndarray, y_true: np.ndarray, k: int) -> float:
    """Fraction of samples whose true class is among the ``k`` highest-scoring classes."""
    k = min(k, proba.shape[1])
    top = np.argpartition(-proba, k - 1, axis=1)[:, :k]
    return float(np.mean((top == np.asarray(y_true)[:, None]).any(axis=1)))


def most_confused_pairs(cm: np.ndarray, top: int = 5) -> list[tuple[int, int, int]]:
    """Unordered class pairs with the most mutual confusions: ``(i, j, count)``, ``i < j``."""
    n = cm.shape[0]
    pairs = [(i, j, int(cm[i, j] + cm[j, i])) for i in range(n) for j in range(i + 1, n)]
    pairs = [p for p in pairs if p[2] > 0]
    return sorted(pairs, key=lambda p: p[2], reverse=True)[:top]
