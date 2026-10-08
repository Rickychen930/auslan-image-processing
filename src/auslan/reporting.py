"""Turns stored results and predictions into the tables the Task 2 report needs."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from auslan.data.classes import AUSLAN_LETTERS
from auslan.evaluation.metrics import macro_f1, most_confused_pairs, per_class_f1
from auslan.evaluation.stats import bootstrap_statistic, holm_adjust, mcnemar_exact
from auslan.experiments import CORE_METHODS, HEADLINE_K
from auslan.results import ResultStore

N_CLASSES = len(AUSLAN_LETTERS)
ALPHA = 0.05
MIN_FPS = 10  # S4
PRIMARY_TRANSFER_METHOD = "m4_mobilenetv2"  # the one pre-specified transfer test (S3)
CROP_TIMING_FILE = "crop_ms.json"


def build_report(store: ResultStore, artifacts_dir: Path) -> Path:
    """Write every table under ``<artifacts_dir>/report`` and return that folder."""
    table = store.table()
    if table.empty:
        raise RuntimeError("no results yet; run an experiment first")
    table = table.assign(tag=table["tag"].fillna(""))
    out = Path(artifacts_dir) / "report"
    out.mkdir(parents=True, exist_ok=True)
    crop_ms = _crop_ms(artifacts_dir)

    table.to_csv(out / "results.csv", index=False)
    (out / "table9.md").write_text(headline_table(table, crop_ms))
    k_curve(table).to_csv(out / "k_curve.csv")
    letters, pairs = confusions(store, table)
    letters.to_csv(out / "per_letter_f1.csv")
    (out / "confused_pairs.md").write_text(pairs)
    (out / "significance.md").write_text(significance(store, table))
    (out / "success_criteria.md").write_text(success_criteria(store, table, crop_ms))
    return out


def _crop_ms(artifacts_dir: Path) -> float:
    path = Path(artifacts_dir) / CROP_TIMING_FILE
    return float(json.loads(path.read_text())["crop_ms"]) if path.exists() else 0.0


def _select(table: pd.DataFrame, method: str, protocol: str, test_set: str, k=None) -> pd.DataFrame:
    rows = table[
        (table["method"] == method)
        & (table["protocol"] == protocol)
        & (table["test_set"] == test_set)
        & (table["tag"] == "")
    ]
    if protocol != "P1":
        rows = rows[rows["k"].isna()] if k is None else rows[rows["k"] == k]
    return rows.sort_values("seed")


def _final_protocol(table: pd.DataFrame, method: str) -> str:
    """P3 where the method can transfer from ASL, otherwise P3 equals P2."""
    has_p3 = ((table["method"] == method) & (table["protocol"] == "P3")).any()
    return "P3" if has_p3 else "P2"


def _methods(table: pd.DataFrame) -> list[str]:
    present = set(table.loc[table["tag"] == "", "method"])
    return [m for m in CORE_METHODS if m in present]


def _percent(values: np.ndarray) -> str:
    if len(values) == 0:
        return "–"
    spread = f" ± {100 * values.std(ddof=1):.1f}" if len(values) > 1 else ""
    return f"{100 * values.mean():.1f}{spread}"


def _markdown(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def headline_table(table: pd.DataFrame, crop_ms: float) -> str:
    """Table 9: macro-F1 (%) at k=20, gain from ASL training, top-3 and speed."""
    rows = []
    for method in _methods(table):
        p1 = _select(table, method, "P1", "asl_cross")["macro_f1"].to_numpy()
        p2 = _select(table, method, "P2", "roboflow", HEADLINE_K)
        final = _final_protocol(table, method)
        p3 = _select(table, method, final, "roboflow", HEADLINE_K)
        gain = "n/a"
        if final == "P3" and len(p2) and len(p3):
            both = p2.merge(p3, on="seed", suffixes=("_p2", "_p3"))
            gain = _percent((both["macro_f1_p3"] - both["macro_f1_p2"]).to_numpy())
        speed = p3["ms_classify"].mean() + crop_ms if len(p3) else float("nan")
        rows.append(
            [
                method,
                _percent(p1),
                _percent(p2["macro_f1"].to_numpy()),
                _percent(p3["macro_f1"].to_numpy()) + ("" if final == "P3" else " (=P2)"),
                gain,
                _percent(p3["top3"].to_numpy()),
                f"{speed:.0f}" if len(p3) else "–",
            ]
        )
    headers = [
        "Method",
        "P1 ASL Test",
        "P2 Roboflow",
        "P3 Roboflow",
        "ΔT(20)",
        "Top-3 P3",
        "ms/frame",
    ]
    return _markdown(headers, rows)


def k_curve(table: pd.DataFrame) -> pd.DataFrame:
    """Mean macro-F1 on Roboflow against examples per letter (E3); NaN in ``k`` means all data."""
    rows = table[
        (table["test_set"] == "roboflow") & (table["tag"] == "") & (table["protocol"] != "P1")
    ]
    curve = rows.groupby(["method", "protocol", rows["k"].fillna(-1).astype(int)])["macro_f1"]
    return curve.mean().rename("macro_f1").reset_index().rename(columns={"k": "k (-1 = all)"})


def _pooled_predictions(
    store: ResultStore, table: pd.DataFrame, method: str, protocol: str, k: int = HEADLINE_K
) -> tuple[np.ndarray, np.ndarray]:
    runs = _select(table, method, protocol, "roboflow", k)
    pairs = [store.predictions(key, "roboflow") for key in runs["key"]]
    y_true = np.concatenate([y for y, _ in pairs])
    return y_true, np.concatenate([p.argmax(axis=1) for _, p in pairs])


def confusions(store: ResultStore, table: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Per-letter F1 and the most confused letter pairs, pooled over the three draws (E4)."""
    f1_by_method, sections = {}, []
    for method in _methods(table):
        protocol = _final_protocol(table, method)
        if _select(table, method, protocol, "roboflow", HEADLINE_K).empty:
            continue
        y_true, y_pred = _pooled_predictions(store, table, method, protocol)
        f1_by_method[method] = per_class_f1(y_true, y_pred, N_CLASSES)
        cm = np.zeros((N_CLASSES, N_CLASSES), int)
        np.add.at(cm, (y_true, y_pred), 1)
        pairs = [
            f"{AUSLAN_LETTERS[i]}/{AUSLAN_LETTERS[j]} ({n})" for i, j, n in most_confused_pairs(cm)
        ]
        sections.append(f"- **{method}** ({protocol}): " + ", ".join(pairs))
    letters = pd.DataFrame(f1_by_method, index=list(AUSLAN_LETTERS))
    return letters, "Most confused letter pairs (count of mutual errors)\n\n" + "\n".join(
        sections
    ) + "\n"


def _correct_by_seed(store: ResultStore, table: pd.DataFrame, method: str) -> dict[int, np.ndarray]:
    runs = _select(table, method, _final_protocol(table, method), "roboflow", HEADLINE_K)
    out = {}
    for seed, key in zip(runs["seed"], runs["key"], strict=True):
        y_true, proba = store.predictions(key, "roboflow")
        out[int(seed)] = proba.argmax(axis=1) == y_true
    return out


def significance(store: ResultStore, table: pd.DataFrame) -> str:
    """McNemar tests between methods with Holm correction (Section 6.4)."""
    methods = [
        m for m in _methods(table) if not _select(table, m, "P2", "roboflow", HEADLINE_K).empty
    ]
    correct = {m: _correct_by_seed(store, table, m) for m in methods}
    pairs = list(itertools.combinations(methods, 2))
    seeds = sorted(set.intersection(*(set(c) for c in correct.values()))) if methods else []
    adjusted = {
        seed: holm_adjust([mcnemar_exact(correct[a][seed], correct[b][seed]) for a, b in pairs])
        for seed in seeds
    }
    rows = []
    for index, (a, b) in enumerate(pairs):
        worst = max(adjusted[s][index] for s in seeds)
        accuracy_a = np.mean([correct[a][s].mean() for s in seeds])
        accuracy_b = np.mean([correct[b][s].mean() for s in seeds])
        verdict = (
            f"{a if accuracy_a > accuracy_b else b} better" if worst < ALPHA else "no difference"
        )
        rows.append([a, b, f"{worst:.4f}", verdict])
    note = (
        "McNemar (exact) at k=20 on Roboflow, Holm-adjusted across method pairs per draw. "
        "A difference counts only if it holds in all draws, so the largest adjusted p is shown.\n\n"
    )
    return (
        note
        + _markdown(["A", "B", "max adjusted p", "verdict"], rows)
        + _transfer_gain(store, table)
    )


def _transfer_gain_ci(store: ResultStore, table: pd.DataFrame, method: str):
    """95% paired-bootstrap CI of ΔT(20): macro-F1 with ASL training minus without, over draws."""
    p2 = _select(table, method, "P2", "roboflow", HEADLINE_K)
    p3 = _select(table, method, "P3", "roboflow", HEADLINE_K)
    seeds = sorted(set(p2["seed"]) & set(p3["seed"]))
    if not seeds:
        return None
    key2, key3 = (
        dict(zip(p2["seed"], p2["key"], strict=True)),
        dict(zip(p3["seed"], p3["key"], strict=True)),
    )
    y_true = store.predictions(key2[seeds[0]], "roboflow")[0]
    pred2 = [store.predictions(key2[s], "roboflow")[1].argmax(axis=1) for s in seeds]
    pred3 = [store.predictions(key3[s], "roboflow")[1].argmax(axis=1) for s in seeds]

    def gain(idx: np.ndarray) -> float:
        return float(
            np.mean(
                [
                    macro_f1(y_true[idx], b[idx], N_CLASSES)
                    - macro_f1(y_true[idx], a[idx], N_CLASSES)
                    for a, b in zip(pred2, pred3, strict=True)
                ]
            )
        )

    return bootstrap_statistic(y_true, gain, n_boot=1000, seed=0, alpha=ALPHA)


def _transfer_gain(store: ResultStore, table: pd.DataFrame) -> str:
    lines = ["\n### Gain from ASL training, ΔT(20) with 95% bootstrap interval\n"]
    for method in _methods(table):
        interval = _transfer_gain_ci(store, table, method)
        if interval is None:
            continue
        marker = " (pre-specified test)" if method == PRIMARY_TRANSFER_METHOD else " (exploratory)"
        lines.append(
            f"- {method}: [{100 * interval[0]:+.1f}, {100 * interval[1]:+.1f}] points{marker}"
        )
    return "\n".join(lines) + "\n"


def _implementation_check(table: pd.DataFrame) -> str:
    """S1: P1 accuracy on the ASL split within 5 points of the published figures."""
    published = {"m4_mobilenetv2": 99.0, "m4_resnet18": 99.0, "m5_landmarks_svm": 87.6}
    rows = []
    for method, target in published.items():
        scores = _select(table, method, "P1", "asl_test")["accuracy"]
        if scores.empty:
            continue
        accuracy = 100 * scores.mean()
        rows.append(
            [method, f"{accuracy:.1f}", f"{target:.1f}", "yes" if accuracy >= target - 5 else "no"]
        )
    if not rows:
        return "S1 (P1 within 5 points of published ASL accuracy): not run yet\n"
    return "S1: P1 accuracy on the held-out ASL split\n\n" + _markdown(
        ["Method", "accuracy %", "published %", "within 5 points"], rows
    )


def success_criteria(store: ResultStore, table: pd.DataFrame, crop_ms: float) -> str:
    """Table 8: S1 to S4 evaluated; S5 is covered by confused_pairs.md and the k curve."""
    lines = []
    for method in _methods(table):
        protocol = _final_protocol(table, method)
        main = _select(table, method, protocol, "roboflow", HEADLINE_K)
        if main.empty:
            continue
        f1, top3 = main["macro_f1"].mean(), main["top3"].mean()
        fps = 1000 / (main["ms_classify"].mean() + crop_ms) if crop_ms else float("nan")
        passed = f1 >= 0.60 and top3 >= 0.85 and fps >= MIN_FPS
        lines.append(
            [
                method,
                f"{100 * f1:.1f}",
                f"{100 * top3:.1f}",
                f"{fps:.1f}",
                "yes" if passed else "no",
            ]
        )
    s2 = _markdown(["Method", "macro-F1 %", "top-3 %", "fps", "S2 and S4 met"], lines)

    interval = _transfer_gain_ci(store, table, PRIMARY_TRANSFER_METHOD)
    s3 = (
        "S3 (ΔT(20) > 0 for MobileNetV2): not run yet\n"
        if interval is None
        else f"S3 (ΔT(20) > 0 for MobileNetV2): {'met' if interval[0] > 0 else 'not met'}, "
        f"95% CI [{100 * interval[0]:+.1f}, {100 * interval[1]:+.1f}] points\n"
    )
    header = (
        "S2 and S4 must be met by the same method (macro-F1 >= 60, top-3 >= 85, >= 10 fps).\n\n"
    )
    return header + s2 + "\n" + s3 + "\n" + _implementation_check(table)
