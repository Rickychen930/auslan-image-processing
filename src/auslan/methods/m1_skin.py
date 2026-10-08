"""M1: skin (or edge-map) segmentation, then shape descriptors and a kNN / SVM classifier."""

from __future__ import annotations

import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from auslan.data.sample_set import SampleSet
from auslan.methods.base import FeatureClassifier, register
from auslan.preprocessing.hand_crop import skin_colour_mask

MIN_BLOB_FRACTION = 0.01  # ignore blobs smaller than 1% of the crop
DEFECT_DEPTH_FRACTION = 0.02  # a convexity defect counts if deeper than 2% of the diagonal
N_FEATURES = 7 + 11  # Hu moments + geometric descriptors
_KERNEL = np.ones((3, 3), np.uint8)


def is_binarised(rgb: np.ndarray) -> bool:
    """True for grey, almost pure black/white images such as the Kaggle Auslan edge drawings."""
    channels = rgb.astype(int)
    grey_pixels = np.mean(channels.max(axis=-1) - channels.min(axis=-1) < 8)
    grey = channels.mean(axis=-1)
    extreme_pixels = np.mean((grey < 30) | (grey > 225))
    return bool(grey_pixels > 0.98 and extreme_pixels > 0.9)


def hand_mask(rgb: np.ndarray) -> np.ndarray:
    """Binary hand mask: skin-colour thresholds for photos, closed and filled edges otherwise."""
    if not is_binarised(rgb):
        mask = skin_colour_mask(rgb)
        return cv2.morphologyEx(
            cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL), cv2.MORPH_CLOSE, _KERNEL
        )
    ink = (cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) < 128).astype(np.uint8) * 255  # dark lines
    closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(closed)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return cv2.morphologyEx(filled, cv2.MORPH_OPEN, _KERNEL)


def _defect_depths(contour: np.ndarray) -> list[float]:
    if len(contour) < 4:
        return []
    hull = cv2.convexHull(contour, returnPoints=False)
    try:
        defects = cv2.convexityDefects(contour, hull)
    except cv2.error:  # self-intersecting contours have no valid defects; treat as none
        return []
    if defects is None:
        return []
    return (defects.reshape(-1, 4)[:, 3] / 256.0).tolist()  # depth is stored as fixed-point


def shape_features(rgb: np.ndarray) -> np.ndarray:
    """Hu moments plus area, solidity, circularity, extent, aspect and convexity-defect stats."""
    mask = hand_mask(rgb)
    height, width = mask.shape
    diagonal = float(np.hypot(height, width))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= MIN_BLOB_FRACTION * height * width]
    if not contours:
        return np.zeros(N_FEATURES, np.float32)

    area = sum(cv2.contourArea(c) for c in contours)
    perimeter = sum(cv2.arcLength(c, True) for c in contours)
    points = np.concatenate(contours)
    hull_area = max(cv2.contourArea(cv2.convexHull(points)), 1.0)
    x, y, w, h = cv2.boundingRect(points)
    depths = [
        d for c in contours for d in _defect_depths(c) if d > DEFECT_DEPTH_FRACTION * diagonal
    ]
    moments = cv2.moments(mask, binaryImage=True)
    hu = cv2.HuMoments(moments).ravel()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)  # compress the huge dynamic range
    geometric = [
        area / (height * width),
        len(contours),
        area / hull_area,
        4 * np.pi * area / max(perimeter**2, 1.0),
        area / max(w * h, 1),
        w / max(h, 1),
        len(depths),
        float(np.mean(depths)) / diagonal if depths else 0.0,
        float(np.max(depths)) / diagonal if depths else 0.0,
        moments["m10"] / max(moments["m00"], 1.0) / width,
        moments["m01"] / max(moments["m00"], 1.0) / height,
    ]
    return np.concatenate([hu, geometric]).astype(np.float32)


class SkinShapeClassifier(FeatureClassifier):
    def __init__(self, name: str, estimator: str) -> None:
        super().__init__(name)
        self._estimator_kind = estimator

    def _features(self, samples: SampleSet) -> np.ndarray:
        return np.stack([shape_features(samples[i].crop) for i in range(len(samples))])

    def _make_estimator(self, features: np.ndarray):
        model = (
            SVC(kernel="rbf", C=10.0, gamma="scale")
            if self._estimator_kind == "svm"
            else KNeighborsClassifier(n_neighbors=3, weights="distance")
        )
        return make_pipeline(StandardScaler(), model)


@register("m1_skin_svm")
def _skin_svm() -> SkinShapeClassifier:
    return SkinShapeClassifier("m1_skin_svm", "svm")


@register("m1_skin_knn")
def _skin_knn() -> SkinShapeClassifier:
    return SkinShapeClassifier("m1_skin_knn", "knn")
