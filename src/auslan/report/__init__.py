"""Builds every table the Task 2 report needs from the stored results."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from auslan.config import Settings
from auslan.data.classes import AUSLAN_LETTERS
from auslan.report import tables
from auslan.report.comparisons import confusions, significance
from auslan.report.failures import detection_rates
from auslan.report.html import render_index
from auslan.results import ResultStore

log = logging.getLogger(__name__)
CROP_TIMING_FILE = "crop_ms.json"


def build_report(store: ResultStore, settings: Settings) -> Path:
    """Write the report files under ``<artifacts_dir>/report`` and return that folder."""
    table = store.table()
    if table.empty:
        raise RuntimeError("no results yet; run an experiment first")
    out = settings.artifacts_dir / "report"
    out.mkdir(parents=True, exist_ok=True)
    crop_ms = _crop_ms(settings.artifacts_dir)

    table.to_csv(out / "results.csv", index=False)
    texts = {
        "table9.md": tables.headline_table(store, table, crop_ms),
        "cost.md": tables.cost_table(table, crop_ms),
        "cross_dataset.md": tables.cross_dataset_table(table),
        "variants.md": tables.variants_table(table),
        "style_effect.md": tables.style_table(table),
        "image_size.md": tables.size_table(table),
        "significance.md": significance(store, table),
        "success_criteria.md": tables.success_criteria(store, table, crop_ms),
        "detection_rates.md": detection_rates(settings),
    }
    for name, text in texts.items():
        (out / name).write_text(text)
    tables.k_curve(table).to_csv(out / "k_curve.csv", index=False)

    letters, pairs, matrices = confusions(store, table)
    letters.to_csv(out / "per_letter_f1.csv")
    (out / "confused_pairs.md").write_text(pairs)
    for method, matrix in matrices.items():
        frame = pd.DataFrame(matrix, index=list(AUSLAN_LETTERS), columns=list(AUSLAN_LETTERS))
        frame.to_csv(out / f"confusion_{method}.csv")
    render_index(out)
    return out


def _crop_ms(artifacts_dir: Path) -> float:
    path = artifacts_dir / CROP_TIMING_FILE
    if not path.exists():
        log.warning(
            "no %s: run `auslan speed`; speed figures will exclude hand detection", path.name
        )
        return 0.0
    return float(json.loads(path.read_text())["crop_ms"])
