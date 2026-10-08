"""Failure checks of report Section 6.2: how often hands are found and how often M1 finds none."""

from __future__ import annotations

import pandas as pd

from auslan.config import Settings
from auslan.data.bundle import DataBundle
from auslan.methods.m1_skin import empty_mask_rate
from auslan.report.common import markdown

DATASETS = ("asl_alphabet", "asl_alphabet_test", "auslan_kaggle", "auslan_roboflow")


def detection_rates(settings: Settings) -> str:
    """MediaPipe detection per dataset, and M1's share of empty masks on the test sets."""
    rows = []
    for name in DATASETS:
        catalog = settings.work_dir / name / "catalog.csv"
        if not catalog.exists():
            continue
        frame = pd.read_csv(catalog)
        share = frame["n_hands"].value_counts(normalize=True)
        rows.append(
            [
                name,
                str(len(frame)),
                *(f"{100 * share.get(n, 0.0):.1f}" for n in (0, 1, 2)),
                f"{100 * frame['used_fallback'].mean():.1f}",
            ]
        )
    headers = ["Dataset", "images", "no hand %", "one hand %", "two hands %", "fallback crop %"]
    return (
        "MediaPipe hand detection on the prepared images\n\n"
        + markdown(headers, rows)
        + "\n"
        + _m1_failures(settings)
    )


def _m1_failures(settings: Settings) -> str:
    bundle = DataBundle(settings.work_dir, settings.image_size)
    sets = {
        "ASL test": lambda: bundle.asl("test"),
        "Kaggle Auslan test": lambda: bundle.kaggle("test"),
        "Roboflow (colour)": lambda: bundle.roboflow(styled=False),
    }
    rows = []
    for name, load in sets.items():
        try:
            rows.append([name, f"{100 * empty_mask_rate(load()):.1f}"])
        except FileNotFoundError:
            continue  # that dataset is not prepared yet
    return "M1: images in which no hand-sized blob is found\n\n" + markdown(
        ["Test set", "empty mask %"], rows
    )
