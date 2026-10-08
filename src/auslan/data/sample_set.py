"""Lazy, read-only view over a prepared catalog (cached hand crops plus landmarks)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd

from auslan.data.classes import LabelSpace

ImageTransform = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Sample:
    crop: np.ndarray  # (H, W, 3) uint8 RGB hand crop
    landmarks: np.ndarray  # (2, 21, 3) float32, zero-filled for hands that were not found


class SampleSet:
    """Rows of a prepared catalog; crops are decoded on access so large sets fit in memory."""

    def __init__(
        self,
        frame: pd.DataFrame,
        landmarks: np.ndarray,
        label_space: LabelSpace,
        image_size: int,
        transform: ImageTransform | None = None,
    ) -> None:
        self._frame = frame.reset_index(drop=True)
        self._landmarks = landmarks
        self._label_space = label_space
        self._image_size = image_size
        self._transform = transform
        self._labels = np.array([label_space.index(x) for x in self._frame["label"]], dtype=np.int64)

    def __len__(self) -> int:
        return len(self._frame)

    def __getitem__(self, index: int) -> Sample:
        row = self._frame.iloc[index]
        return Sample(self._load_crop(row["crop_path"]), self._landmarks[int(row["lm_index"])])

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

    def landmark_array(self) -> np.ndarray:
        """All landmarks at once, in row order, without decoding any image."""
        return self._landmarks[self._frame["lm_index"].to_numpy(dtype=int)]

    def subset(self, rows: Sequence[int]) -> SampleSet:
        return SampleSet(
            self._frame.iloc[list(rows)],
            self._landmarks,
            self._label_space,
            self._image_size,
            self._transform,
        )

    def with_transform(self, transform: ImageTransform | None) -> SampleSet:
        return SampleSet(
            self._frame, self._landmarks, self._label_space, self._image_size, transform
        )

    def _load_crop(self, path: str) -> np.ndarray:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"cannot read crop: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[0] != self._image_size or rgb.shape[1] != self._image_size:
            rgb = cv2.resize(rgb, (self._image_size, self._image_size), interpolation=cv2.INTER_AREA)
        return self._transform(rgb) if self._transform else rgb
