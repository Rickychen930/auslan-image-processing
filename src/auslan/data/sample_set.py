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
        crops: np.ndarray | None = None,
    ) -> None:
        self._crops = crops  # (n, H, W, 3) crops held in memory instead of on disk
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
        crop = self._finish(self._crops[index] if self._crops is not None else self._read(index))
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

    @classmethod
    def from_memory(
        cls,
        crop: np.ndarray,
        landmarks: np.ndarray,
        label_space: LabelSpace,
        image_size: int,
        styled: bool = False,
    ) -> SampleSet:
        """A one-sample set from an image that is not on disk (a camera frame).

        The label is a placeholder, since nothing is known about a live frame.
        """
        frame = pd.DataFrame(
            {"crop_path": [""], "label": [label_space.letters[0]], "lm_index": [0]}
        )
        return cls(frame, landmarks[None], label_space, image_size, styled, crop[None])

    def subset(self, rows: Sequence[int]) -> SampleSet:
        rows = list(rows)
        crops = None if self._crops is None else self._crops[rows]
        return SampleSet(
            self._frame.iloc[rows],
            self._landmarks,
            self._label_space,
            self._image_size,
            self._styled,
            crops,
        )

    def _read(self, index: int) -> np.ndarray:
        path = self._crop_paths[index]
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"cannot read crop: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _finish(self, rgb: np.ndarray) -> np.ndarray:
        """Resize to the model's input size and, for styled sets, draw it like the Kaggle images."""
        if rgb.shape[0] != self._image_size or rgb.shape[1] != self._image_size:
            size = (self._image_size, self._image_size)
            rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        return kaggle_style(rgb) if self._styled else rgb
