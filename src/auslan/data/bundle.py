"""The four prepared datasets, exposed as the sample sets the protocols need."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from auslan.data.classes import ASL_SPACE, AUSLAN_SPACE, LabelSpace
from auslan.data.prepare import load_prepared
from auslan.data.sample_set import SampleSet


class DataBundle:
    """Reads prepared catalogs lazily; every sample set uses the same ``image_size``."""

    def __init__(self, work_dir: Path, image_size: int) -> None:
        self._work_dir = Path(work_dir)
        self._image_size = image_size
        self._loaded: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}

    def asl(self, split: str, styled: bool = False) -> SampleSet:
        """Held-out parts of ASL Alphabet; ``styled`` applies the Kaggle-style transform T."""
        return self._set("asl_alphabet", ASL_SPACE, split=split, styled=styled)

    def asl_cross(self) -> SampleSet:
        """ASL Alphabet Test: the same letters on varied backgrounds (P1 cross-dataset test)."""
        return self._set("asl_alphabet_test", ASL_SPACE)

    def kaggle_frame(self, split: str) -> pd.DataFrame:
        """Catalog rows of one Kaggle Auslan split, for k-shot sampling."""
        frame, _ = self._catalog("auslan_kaggle")
        return frame[frame["split"] == split]

    def kaggle_set(self, frame: pd.DataFrame) -> SampleSet:
        _, landmarks = self._catalog("auslan_kaggle")
        return SampleSet(frame, landmarks, AUSLAN_SPACE, self._image_size)

    def kaggle(self, split: str) -> SampleSet:
        return self.kaggle_set(self.kaggle_frame(split))

    def roboflow(self, styled: bool = True) -> SampleSet:
        """The independent Auslan test set; T is applied unless the E6 'raw' variant is wanted."""
        return self._set("auslan_roboflow", AUSLAN_SPACE, styled=styled)

    def _set(
        self, name: str, space: LabelSpace, split: str | None = None, styled: bool = False
    ) -> SampleSet:
        frame, landmarks = self._catalog(name)
        if split is not None:
            frame = frame[frame["split"] == split]
        return SampleSet(frame, landmarks, space, self._image_size, styled=styled)

    def _catalog(self, name: str) -> tuple[pd.DataFrame, np.ndarray]:
        if name not in self._loaded:
            folder = self._work_dir / name
            if not (folder / "catalog.csv").exists():
                raise FileNotFoundError(f"{name} is not prepared yet; run `auslan prepare {name}`")
            self._loaded[name] = load_prepared(folder)
        return self._loaded[name]
