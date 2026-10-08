"""Project settings: one immutable object, optionally overridden from a YAML file."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    seeds: tuple[int, ...] = (0, 1, 2)  # repeat runs / k-shot draws (Section 6.4)
    split_seed: int = 0
    k_values: tuple[int, ...] = (5, 10, 20, 50)  # plus "all", handled by the protocols
    val_per_class: int = 10  # validation images per letter under P2/P3
    image_size: int = 128  # CNN input; 224 crops are cached and resized on read
    crop_size: int = 224
    near_duplicate_hamming: int = 4
    workers: int = 4

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def work_dir(self) -> Path:
        return self.data_dir / "work"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"


_PATH_FIELDS = {"data_dir", "artifacts_dir"}
_TUPLE_FIELDS = {"seeds", "k_values"}


def load_settings(path: Path | None) -> Settings:
    """Defaults, overridden by the keys present in ``path``; unknown keys are an error."""
    if path is None:
        return Settings()
    raw = yaml.safe_load(Path(path).read_text()) or {}
    known = {f.name for f in fields(Settings)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"unknown setting(s): {', '.join(unknown)}")
    coerced = {
        key: Path(v) if key in _PATH_FIELDS else tuple(v) if key in _TUPLE_FIELDS else v
        for key, v in raw.items()
    }
    return replace(Settings(), **coerced)
