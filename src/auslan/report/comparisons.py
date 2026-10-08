"""Comparisons between methods: confused letters, significance tests and the gain from ASL."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from auslan.data.classes import AUSLAN_LETTERS
from auslan.evaluation.metrics import macro_f1, most_confused_pairs, per_class_f1
from auslan.evaluation.stats import bootstrap_statistic, holm_adjust, mcnemar_exact
from auslan.experiments import HEADLINE_K
from auslan.report.common import (
    ALPHA,
    N_BOOT,
    N_CLASSES,
    core_methods,
    final_protocol,
    markdown,
    predictions_of,
    select,
)
from auslan.results import ResultStore

PRIMARY_TRANSFER_METHOD = "m4_mobilenetv2"  # the one pre-specified transfer test (S3)


def _main_runs(table: pd.DataFrame, method: str) -> dict[int, str]:
    """seed -> run key for the method's main k=20 runs on Roboflow."""
    rows = select(table, method, final_protocol(table, method), "roboflow", HEADLINE_K)
    return dict(zip(rows["seed"], rows["key"], strict=True))


def confusions(
    store: ResultStore, table: pd.DataFrame
) -> tuple[pd.DataFrame, str, dict[str, np.ndarray]]:
    """Per-letter F1, the most confused letter pairs and the confusion matrices (E4).

    Everything is pooled over the draws of the main k=20 runs on Roboflow.
    """
    f1_by_method, matrices, sections = {}, {}, []
    for method in core_methods(table):
        runs = _main_runs(table, method)
        if not runs:
            continue
        y_true, predicted = predictions_of(store, list(runs.values()), "roboflow")
        y_all, p_all = np.tile(y_true, len(predicted)), np.concatenate(predicted)
        cm = np.zeros((N_CLASSES, N_CLASSES), int)
        np.add.at(cm, (y_all, p_all), 1)
        f1_by_method[method], matrices[method] = per_class_f1(y_all, p_all, N_CLASSES), cm
        pairs = [
            f"{AUSLAN_LETTERS[i]}/{AUSLAN_LETTERS[j]} ({n})" for i, j, n in most_confused_pairs(cm)
        ]
        sections.append(f"- **{method}** ({final_protocol(table, method)}): " + ", ".join(pairs))
    letters = pd.DataFrame(f1_by_method, index=list(AUSLAN_LETTERS))
    text = "Most confused letter pairs (mutual errors, pooled over draws)\n\n" + "\n".join(sections)
    return letters, text + "\n", matrices


def significance(store: ResultStore, table: pd.DataFrame) -> str:
    """McNemar tests and paired bootstrap between methods, then the gain from ASL training."""
    runs = {m: _main_runs(table, m) for m in core_methods(table)}
    runs = {m: r for m, r in runs.items() if r}
    seeds = sorted(set.intersection(*(set(r) for r in runs.values()))) if runs else []
    if len(runs) < 2 or not seeds:
        return "Needs at least two methods with main runs for the same draws.\n" + _transfer_gain(
            store, table
        )

    y_true, correct, predicted = None, {}, {}
    for method, by_seed in runs.items():
        y_true, predicted[method] = predictions_of(store, [by_seed[s] for s in seeds], "roboflow")
        correct[method] = [p == y_true for p in predicted[method]]
    pairs = list(itertools.combinations(runs, 2))
    adjusted = [  # per draw: Holm over all method pairs
        holm_adjust([mcnemar_exact(correct[a][i], correct[b][i]) for a, b in pairs])
        for i in range(len(seeds))
    ]
    rows = []
    for index, (a, b) in enumerate(pairs):
        worst_p = max(draw[index] for draw in adjusted)
        low, high = _difference_interval(y_true, predicted[a], predicted[b])
        if worst_p < ALPHA and (low > 0 or high < 0):
            verdict = f"{a if low > 0 else b} better"
        else:
            verdict = "no clear difference"
        rows.append([a, b, f"{worst_p:.4f}", f"[{100 * low:+.1f}, {100 * high:+.1f}]", verdict])
    note = (
        "Main k=20 runs on Roboflow. McNemar (exact) is Holm-adjusted across method pairs within "
        "each draw; the largest adjusted p over the draws is shown, so a difference counts only if "
        "it holds in all of them. The interval is the paired bootstrap of the macro-F1 difference "
        "(A minus B) averaged over draws; it reflects test-image uncertainty only.\n\n"
    )
    table_md = markdown(["A", "B", "max adjusted p", "ΔF1 95% CI (points)", "verdict"], rows)
    return note + table_md + _transfer_gain(store, table)


def _difference_interval(y_true: np.ndarray, pred_a: list, pred_b: list) -> tuple[float, float]:
    def difference(idx: np.ndarray) -> float:
        return float(
            np.mean(
                [
                    macro_f1(y_true[idx], a[idx], N_CLASSES)
                    - macro_f1(y_true[idx], b[idx], N_CLASSES)
                    for a, b in zip(pred_a, pred_b, strict=True)
                ]
            )
        )

    return bootstrap_statistic(y_true, difference, N_BOOT, seed=0, alpha=ALPHA)


def transfer_gain_ci(
    store: ResultStore, table: pd.DataFrame, method: str
) -> tuple[float, float] | None:
    """95% paired-bootstrap interval of ΔT(20): macro-F1 with ASL training minus without."""
    p2 = select(table, method, "P2", "roboflow", HEADLINE_K)
    p3 = select(table, method, "P3", "roboflow", HEADLINE_K)
    seeds = sorted(set(p2["seed"]) & set(p3["seed"]))
    if not seeds:
        return None
    key2 = dict(zip(p2["seed"], p2["key"], strict=True))
    key3 = dict(zip(p3["seed"], p3["key"], strict=True))
    y_true, without = predictions_of(store, [key2[s] for s in seeds], "roboflow")
    _, with_asl = predictions_of(store, [key3[s] for s in seeds], "roboflow")

    def gain(idx: np.ndarray) -> float:
        return float(
            np.mean(
                [
                    macro_f1(y_true[idx], b[idx], N_CLASSES)
                    - macro_f1(y_true[idx], a[idx], N_CLASSES)
                    for a, b in zip(without, with_asl, strict=True)
                ]
            )
        )

    return bootstrap_statistic(y_true, gain, N_BOOT, seed=0, alpha=ALPHA)


def _transfer_gain(store: ResultStore, table: pd.DataFrame) -> str:
    lines = ["\n### Gain from ASL training, ΔT(20), 95% bootstrap interval\n"]
    for method in core_methods(table):
        interval = transfer_gain_ci(store, table, method)
        if interval is None:
            continue
        kind = "pre-specified test" if method == PRIMARY_TRANSFER_METHOD else "exploratory"
        lines.append(
            f"- {method}: [{100 * interval[0]:+.1f}, {100 * interval[1]:+.1f}] points ({kind})"
        )
    return "\n".join(lines) + "\n"
