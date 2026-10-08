"""Build an image catalog (path, label, source) from a class-per-folder dataset."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from auslan.data.classes import letter_from_name

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp"})
CATALOG_COLUMNS = ["path", "label", "source"]


def scan_folder_dataset(root: Path, source: str, letters: Iterable[str]) -> pd.DataFrame:
    """List every image whose parent folder names one of ``letters``.

    Folders are matched at any depth, so nested exports (``train/train/A``) work. Folders that
    are not a requested letter (digits, ``del``, ``space``, J/Z for ASL) are ignored.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root not found: {root}")
    wanted = frozenset(letters)
    rows = [
        (str(path.resolve()), letter, source)
        for path in sorted(root.rglob("*"))
        if _is_image(path) and (letter := letter_from_name(path.parent.name)) in wanted
    ]
    return pd.DataFrame(rows, columns=CATALOG_COLUMNS)


def _is_image(path: Path) -> bool:
    return (
        path.suffix.lower() in IMAGE_SUFFIXES and not path.name.startswith(".") and path.is_file()
    )
