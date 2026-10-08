"""The four public datasets of the study and how each one is prepared."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from auslan.data.classes import ASL_LETTERS, AUSLAN_LETTERS

Fallback = Literal["skin", "full"]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: Path
    letters: tuple[str, ...]
    fallback: Fallback  # region used when no hand is found: skin blobs (colour) or whole frame
    split: bool  # True: carve train/val/test; False: the whole set is a test set


def dataset_specs(raw_dir: Path) -> dict[str, DatasetSpec]:
    """Datasets keyed by name, rooted under ``raw_dir`` (see Table 2 of the report)."""
    specs = [
        DatasetSpec("asl_alphabet", raw_dir / "asl_alphabet", ASL_LETTERS, "skin", split=True),
        DatasetSpec(
            "asl_alphabet_test", raw_dir / "asl_alphabet_test", ASL_LETTERS, "skin", split=False
        ),
        DatasetSpec("auslan_kaggle", raw_dir / "auslan_kaggle", AUSLAN_LETTERS, "full", split=True),
        DatasetSpec(
            "auslan_roboflow", raw_dir / "auslan_roboflow", AUSLAN_LETTERS, "skin", split=False
        ),
    ]
    return {s.name: s for s in specs}
