"""Image-style transform that makes colour photos resemble the binarised Kaggle Auslan set."""

from __future__ import annotations

import cv2
import numpy as np

BLUR_KERNEL = (5, 5)
THRESHOLD_BLOCK = 11
THRESHOLD_OFFSET = 2


def kaggle_style(rgb: np.ndarray) -> np.ndarray:
    """Greyscale, Gaussian blur and adaptive threshold; returns a 3-channel 0/255 image (T)."""
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(grey, BLUR_KERNEL, 0)
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        THRESHOLD_BLOCK,
        THRESHOLD_OFFSET,
    )
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
