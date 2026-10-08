"""Lazy, read-only view over a prepared catalog (cached hand crops plus landmarks)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd

from auslan.data.classes import LabelSpace
from auslan.data.transforms import kaggle_style


@dataclass(frozen=True)
class Sample:
    crop: np.ndarray  # (H, W, 3) uint8 RGB hand crop
    landmarks: np.ndarray  # (2, 21, 3) float32, zero-filled for hands that were not found
    binarised: bool  # the crop is a black-and-white drawing rather than a colour photo


class SampleSet:
    """Rows of a prepared catalog; crops are decoded on access so large sets fit in memory.

    ``styled`` applies the Kaggle-style transform T to every crop, which also makes them drawings.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        landmarks: np.ndarray,
        label_space: LabelSpace,
        image_size: int,
        styled: bool = False,
    ) -> None:
        self._frame = frame.reset_index(drop=True)
        self._landmarks = landmarks
        self._label_space = label_space
        self._image_size = image_size
        self._styled = styled
        self._crop_paths = self._frame["crop_path"].to_numpy()
        self._landmark_rows = self._frame["lm_index"].to_numpy(dtype=int)
        if "binarised" in self._frame:
            native = self._frame["binarised"].to_numpy(dtype=bool)
        else:
            native = np.zeros(len(self._frame), dtype=bool)
        self._binarised = native | styled
        self._labels = np.array(
            [label_space.index(letter) for letter in self._frame["label"]], dtype=np.int64
        )

    def __len__(self) -> int:
        return len(self._frame)

    def __getitem__(self, index: int) -> Sample:
        crop = self._load_crop(self._crop_paths[index])
        return Sample(
            crop, self._landmarks[self._landmark_rows[index]], bool(self._binarised[index])
        )

    @property
    def labels(self) -> np.ndarray:
        return self._labels

    @property
    def frame(self) -> pd.DataFrame:
        return self._frame

    @property
    def label_space(self) -> LabelSpace:
        return self._label_space

    @property
    def image_size(self) -> int:
        return self._image_size

    @property
    def mostly_binarised(self) -> bool:
        return bool(self._binarised.any() and self._binarised.mean() > 0.5)

    def landmark_array(self) -> np.ndarray:
        """All landmarks at once, in row order, without decoding any image."""
        return self._landmarks[self._landmark_rows]

    def subset(self, rows: Sequence[int]) -> SampleSet:
        return SampleSet(
            self._frame.iloc[list(rows)],
            self._landmarks,
            self._label_space,
            self._image_size,
            self._styled,
        )

    def _load_crop(self, path: str) -> np.ndarray:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"cannot read crop: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[0] != self._image_size or rgb.shape[1] != self._image_size:
            size = (self._image_size, self._image_size)
            rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        return kaggle_style(rgb) if self._styled else rgb
