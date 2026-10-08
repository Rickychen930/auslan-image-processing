"""Helpers shared by the report modules: picking runs, intervals, number and table formatting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from auslan.data.classes import AUSLAN_LETTERS
from auslan.evaluation.metrics import macro_f1
from auslan.evaluation.stats import bootstrap_statistic
from auslan.experiments import CORE_METHODS
from auslan.results import ResultStore

N_CLASSES = len(AUSLAN_LETTERS)
ALPHA = 0.05
N_BOOT = 1000


def select(
    table: pd.DataFrame,
    method: str,
    protocol: str,
    test_set: str,
    k: int | None = None,
    tag: str = "",
) -> pd.DataFrame:
    """Result rows of one method, protocol and test set, ordered by seed.

    For P2 and P3, ``k=None`` selects the all-data runs.
    """
    rows = table[
        (table["method"] == method)
        & (table["protocol"] == protocol)
        & (table["test_set"] == test_set)
        & (table["tag"] == tag)
    ]
    if protocol != "P1":
        rows = rows[rows["k"].isna()] if k is None else rows[rows["k"] == k]
    return rows.sort_values("seed")


def final_protocol(table: pd.DataFrame, method: str, tag: str = "") -> str:
    """P3 where the method can transfer from ASL, otherwise P3 equals P2."""
    rows = table[(table["method"] == method) & (table["tag"] == tag) & (table["protocol"] == "P3")]
    return "P3" if len(rows) else "P2"


def core_methods(table: pd.DataFrame) -> list[str]:
    present = set(table.loc[table["tag"] == "", "method"])
    return [m for m in CORE_METHODS if m in present]


def percent(values: np.ndarray) -> str:
    """Mean in percent, with the standard deviation over draws when there are several."""
    if len(values) == 0:
        return "–"
    spread = f" ± {100 * values.std(ddof=1):.1f}" if len(values) > 1 else ""
    return f"{100 * values.mean():.1f}{spread}"


def markdown(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def predictions_of(store: ResultStore, keys: list[str], test_set: str) -> tuple[np.ndarray, list]:
    """True labels and the predicted classes of several runs on one test set.

    Runs are only comparable image by image if every run saw the same images in the same order.
    """
    y_true = store.predictions(keys[0], test_set)[0]
    predicted = []
    for key in keys:
        labels, proba = store.predictions(key, test_set)
        if not np.array_equal(labels, y_true):
            raise ValueError(f"{key} was scored on different images than {keys[0]}")
        predicted.append(proba.argmax(axis=1))
    return y_true, predicted


def mean_f1_interval(store: ResultStore, keys: list[str], test_set: str) -> tuple[float, float]:
    """95% bootstrap interval of macro-F1 averaged over the runs, resampled within each letter.

    It reflects test-image uncertainty only; the spread between draws is the reported s.d.
    """
    y_true, predicted = predictions_of(store, keys, test_set)

    def mean_f1(idx: np.ndarray) -> float:
        return float(np.mean([macro_f1(y_true[idx], p[idx], N_CLASSES) for p in predicted]))

    return bootstrap_statistic(y_true, mean_f1, N_BOOT, seed=0, alpha=ALPHA)
