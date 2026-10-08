"""Run specifications and the on-disk store for their results and predictions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PROTOCOLS = ("P1", "P2", "P3")


@dataclass(frozen=True)
class RunSpec:
    """One training run. P1 trains on ASL; P2/P3 draw ``k`` Kaggle Auslan images per letter.

    ``k=None`` means all training data. Runs with non-default ``options`` or ``image_size``
    must carry a ``tag`` so their results can be told apart.
    """

    method: str
    protocol: str
    k: int | None = None
    seed: int = 0
    options: dict = field(default_factory=dict)
    image_size: int | None = None
    tag: str = ""

    def __post_init__(self) -> None:
        if self.protocol not in PROTOCOLS:
            raise ValueError(f"protocol must be one of {PROTOCOLS}, got {self.protocol!r}")
        if (self.options or self.image_size) and not self.tag:
            raise ValueError("runs with custom options or image size need a tag")

    @property
    def key(self) -> str:
        shots = "" if self.protocol == "P1" else f"-k{self.k if self.k else 'all'}"
        tag = f"-{self.tag}" if self.tag else ""
        return f"{self.method}-{self.protocol}{shots}-s{self.seed}{tag}"


@dataclass(frozen=True)
class Result:
    """Scores of one run on one test set."""

    key: str
    method: str
    protocol: str
    k: int | None
    seed: int
    tag: str
    test_set: str
    n_train: int
    macro_f1: float
    accuracy: float
    top3: float
    fit_seconds: float
    ms_classify: float  # CPU time per frame for features + classifier, excluding cropping
    model_mb: float  # size of the fitted model when pickled


class ResultStore:
    """Append-only ``results.jsonl`` plus one ``.npz`` of predictions per run and test set."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._log = self._root / "results.jsonl"
        self._predictions = self._root / "predictions"
        self._predictions.mkdir(parents=True, exist_ok=True)
        self._keys = set(self.table()["key"])

    def has(self, key: str) -> bool:
        return key in self._keys

    def add(
        self, results: list[Result], predictions: dict[str, tuple[np.ndarray, np.ndarray]]
    ) -> None:
        """Save predictions first, so a logged result always has its predictions on disk."""
        for result in results:
            y_true, proba = predictions[result.test_set]
            np.savez_compressed(
                self._path(result.key, result.test_set),
                y_true=y_true,
                proba=proba.astype(np.float16),
            )
        with self._log.open("a") as handle:
            handle.write("".join(json.dumps(asdict(result)) + "\n" for result in results))
        self._keys.update(result.key for result in results)

    def table(self) -> pd.DataFrame:
        if not self._log.exists():
            return pd.DataFrame(columns=list(Result.__dataclass_fields__))
        table = pd.read_json(self._log, lines=True, dtype={"tag": str})
        return table.assign(tag=table["tag"].fillna(""))

    def predictions(self, key: str, test_set: str) -> tuple[np.ndarray, np.ndarray]:
        data = np.load(self._path(key, test_set))
        return data["y_true"], data["proba"].astype(np.float32)

    def _path(self, key: str, test_set: str) -> Path:
        return self._predictions / f"{key}__{test_set}.npz"
