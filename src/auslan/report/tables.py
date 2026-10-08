"""Tables built from the stored results: Table 9 and the tables for experiments E1 to E7."""

from __future__ import annotations

import numpy as np
import pandas as pd

from auslan.experiments import HEADLINE_K
from auslan.report.common import (
    core_methods,
    final_protocol,
    markdown,
    mean_f1_interval,
    percent,
    select,
)
from auslan.report.comparisons import PRIMARY_TRANSFER_METHOD, transfer_gain_ci
from auslan.results import ResultStore

MIN_FPS = 10  # S4
MIN_F1, MIN_TOP3 = 0.60, 0.85  # S2
PUBLISHED_ACCURACY = {"m4_mobilenetv2": 99.0, "m4_resnet18": 99.0, "m5_landmarks_svm": 87.6}  # S1
S1_TOLERANCE = 5.0
IMAGE_SIZES = (32, 64, 128, 224)


def _main(table: pd.DataFrame, method: str, tag: str = "") -> pd.DataFrame:
    """The method's k=20 runs on Roboflow, under P3 if it can transfer and under P2 otherwise."""
    return select(table, method, final_protocol(table, method, tag), "roboflow", HEADLINE_K, tag)


def _interval_text(store: ResultStore, rows: pd.DataFrame) -> str:
    if rows.empty:
        return "–"
    low, high = mean_f1_interval(store, rows["key"].tolist(), "roboflow")
    return f"[{100 * low:.1f}, {100 * high:.1f}]"


def headline_table(store: ResultStore, table: pd.DataFrame, crop_ms: float) -> str:
    """Table 9: macro-F1 (%) at k=20, with the gain from ASL training, top-3 and speed."""
    rows = []
    for method in core_methods(table):
        p1 = select(table, method, "P1", "asl_cross")["macro_f1"].to_numpy()
        p2 = select(table, method, "P2", "roboflow", HEADLINE_K)
        main = _main(table, method)
        transfers = final_protocol(table, method) == "P3"
        gain = "n/a"
        if transfers and len(p2) and len(main):
            both = p2.merge(main, on="seed", suffixes=("_p2", "_p3"))
            gain = percent((both["macro_f1_p3"] - both["macro_f1_p2"]).to_numpy())
        label = "" if transfers or main.empty else " (=P2)"
        speed = f"{main['ms_classify'].mean() + crop_ms:.0f}" if len(main) else "–"
        rows.append(
            [
                method,
                percent(p1),
                percent(p2["macro_f1"].to_numpy()),
                percent(main["macro_f1"].to_numpy()) + label,
                _interval_text(store, main),
                gain,
                percent(main["top3"].to_numpy()),
                speed,
            ]
        )
    headers = [
        "Method",
        "P1 ASL Test",
        "P2 Roboflow",
        "P3 Roboflow",
        "95% CI",
        "ΔT(20)",
        "Top-3",
        "ms/frame",
    ]
    note = (
        "Macro-F1 in %, mean ± s.d. over three draws. P3 = P2 where a method cannot transfer, and "
        "a dash means the method was not run there (M5 needs detected hands). 95% CI is the "
        "bootstrap interval of the draw-averaged macro-F1.\n\n"
    )
    return note + markdown(headers, rows)


def cost_table(table: pd.DataFrame, crop_ms: float) -> str:
    """Training time, model size and CPU time per frame (report Section 6.2)."""
    rows = []
    for method in core_methods(table):
        runs = _main(table, method)
        if runs.empty:
            runs = select(table, method, "P1", "asl_cross")
        total = runs["ms_classify"].mean() + crop_ms
        rows.append(
            [
                method,
                f"{runs['fit_seconds'].mean():.0f}",
                f"{runs['model_mb'].mean():.1f}",
                f"{runs['ms_classify'].mean():.1f}",
                f"{crop_ms:.1f}",
                f"{1000 / total:.1f}" if crop_ms else "–",
            ]
        )
    headers = ["Method", "train s", "model MB", "classify ms", "detect+crop ms", "frames/s"]
    note = "CPU timings; 'classify' includes decoding the cached crop and computing features.\n\n"
    return note + markdown(headers, rows)


def k_curve(table: pd.DataFrame) -> pd.DataFrame:
    """Macro-F1 on Roboflow against examples per letter (E3); ``k = -1`` means all data."""
    rows = table[
        (table["test_set"] == "roboflow") & (table["tag"] == "") & (table["protocol"] != "P1")
    ]
    grouped = rows.groupby(["method", "protocol", rows["k"].fillna(-1).astype(int)])["macro_f1"]
    return grouped.agg(["mean", "std", "count"]).reset_index().rename(columns={"k": "k (-1 = all)"})


def cross_dataset_table(table: pd.DataFrame) -> str:
    """E2: same-dataset against cross-dataset macro-F1 (%)."""
    rows = []
    for method in core_methods(table):
        protocol = final_protocol(table, method)
        pairs = [
            (
                "P1",
                select(table, method, "P1", "asl_test"),
                select(table, method, "P1", "asl_cross"),
            ),
            (
                protocol,
                select(table, method, protocol, "kaggle_test", HEADLINE_K),
                _main(table, method),
            ),
        ]
        for name, same, cross in pairs:
            if same.empty or cross.empty:
                continue
            drop = same["macro_f1"].mean() - cross["macro_f1"].mean()
            rows.append(
                [
                    method,
                    name,
                    percent(same["macro_f1"].to_numpy()),
                    percent(cross["macro_f1"].to_numpy()),
                    f"{100 * drop:.1f}",
                ]
            )
    headers = ["Method", "Protocol", "Same dataset", "Other dataset", "Drop (points)"]
    note = (
        "P1: ASL split against ASL Alphabet Test. P2/P3 (k=20): Kaggle split against Roboflow.\n\n"
    )
    return note + markdown(headers, rows)


def variants_table(table: pd.DataFrame) -> str:
    """E5: each design alternative next to the default of its family."""
    tags = table.loc[(table["tag"] != "") & ~table["tag"].str.startswith("size"), ["method", "tag"]]
    rows = []
    for method, tag in tags.drop_duplicates().itertuples(index=False):
        variant_p1 = select(table, method, "P1", "asl_cross", tag=tag)["macro_f1"].to_numpy()
        base_p1 = select(table, method, "P1", "asl_cross")["macro_f1"].to_numpy()
        variant = _main(table, method, tag)["macro_f1"].to_numpy()
        base = _main(table, method)["macro_f1"].to_numpy()
        rows.append(
            [
                f"{method} [{tag}]",
                percent(variant_p1),
                percent(base_p1),
                percent(variant),
                percent(base),
            ]
        )
    headers = ["Variant", "P1 cross: variant", "default", "Roboflow k=20: variant", "default"]
    return markdown(headers, rows)


def style_table(table: pd.DataFrame) -> str:
    """E6: Roboflow with and without the Kaggle-style transform T."""
    rows = []
    for method in core_methods(table):
        protocol = final_protocol(table, method)
        styled = _main(table, method)
        raw = select(table, method, protocol, "roboflow_raw", HEADLINE_K)
        if styled.empty or raw.empty:
            continue
        drop = raw["macro_f1"].mean() - styled["macro_f1"].mean()
        rows.append(
            [
                method,
                protocol,
                percent(styled["macro_f1"].to_numpy()),
                percent(raw["macro_f1"].to_numpy()),
                f"{100 * drop:+.1f}",
            ]
        )
    headers = ["Method", "Protocol", "With T", "Without T", "Change (points)"]
    note = "M1 is always scored on the colour photo, so its two columns are the same.\n\n"
    return note + markdown(headers, rows)


def size_table(table: pd.DataFrame) -> str:
    """E7: macro-F1 on Roboflow at k=20 against the input image size."""
    rows = []
    for method in core_methods(table):
        values = {}
        for size in IMAGE_SIZES:
            tag = "" if size == 128 else f"size{size}"
            values[size] = percent(_main(table, method, tag)["macro_f1"].to_numpy())
        if any(v != "–" for v in values.values()):
            rows.append([method, *values.values()])
    return markdown(["Method", *(f"{s} px" for s in IMAGE_SIZES)], rows)


def success_criteria(store: ResultStore, table: pd.DataFrame, crop_ms: float) -> str:
    """Table 8 checked against the results; S5 is covered by confused_pairs.md."""
    s1 = _implementation_check(table)
    s2 = []
    for method in core_methods(table):
        main = _main(table, method)
        if main.empty:
            continue
        f1, top3 = main["macro_f1"].mean(), main["top3"].mean()
        fps = 1000 / (main["ms_classify"].mean() + crop_ms) if crop_ms else float("nan")
        met = f1 >= MIN_F1 and top3 >= MIN_TOP3 and fps >= MIN_FPS
        s2.append(
            [method, f"{100 * f1:.1f}", f"{100 * top3:.1f}", f"{fps:.1f}", "yes" if met else "no"]
        )
    interval = transfer_gain_ci(store, table, PRIMARY_TRANSFER_METHOD)
    if interval is None:
        s3 = "S3 (ΔT(20) > 0 for MobileNetV2): not run yet\n"
    else:
        verdict = "met" if interval[0] > 0 else "not met"
        s3 = (
            f"S3 (ΔT(20) > 0 for MobileNetV2): {verdict}, "
            f"95% CI [{100 * interval[0]:+.1f}, {100 * interval[1]:+.1f}] points\n"
        )
    header = (
        f"S2 and S4 must be met by the same method (macro-F1 >= {MIN_F1:.0%}, "
        f"top-3 >= {MIN_TOP3:.0%}, >= {MIN_FPS} frames/s).\n\n"
    )
    headers = ["Method", "macro-F1 %", "top-3 %", "frames/s", "S2 and S4 met"]
    return f"{s1}\n{header}{markdown(headers, s2)}\n{s3}"


def _implementation_check(table: pd.DataFrame) -> str:
    """S1: P1 accuracy on the held-out ASL split within 5 points of the published figures."""
    rows = []
    for method, published in PUBLISHED_ACCURACY.items():
        scores = select(table, method, "P1", "asl_test")["accuracy"]
        if scores.empty:
            continue
        accuracy = 100 * float(np.mean(scores))
        ok = abs(accuracy - published) <= S1_TOLERANCE
        rows.append([method, f"{accuracy:.1f}", f"{published:.1f}", "yes" if ok else "no"])
    if not rows:
        return "S1 (P1 within 5 points of published ASL accuracy): not run yet\n"
    headers = ["Method", "accuracy %", "published %", "within 5 points"]
    return "S1: P1 accuracy on the held-out ASL split\n\n" + markdown(headers, rows)
