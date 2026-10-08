"""M1: skin (or edge-map) segmentation, then shape descriptors and a kNN / SVM classifier."""

from __future__ import annotations

import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from auslan.data.sample_set import SampleSet
from auslan.methods.base import SVM_C_GRID, FeatureClassifier, register
from auslan.preprocessing.hand_crop import clean_mask, skin_colour_mask

MIN_BLOB_FRACTION = 0.01  # ignore blobs smaller than 1% of the crop
DEFECT_DEPTH_FRACTION = 0.02  # a convexity defect counts if deeper than 2% of the diagonal
N_FEATURES = 7 + 11  # Hu moments + geometric descriptors
KNN_NEIGHBOURS = (1, 3, 5)


def hand_mask(rgb: np.ndarray, binarised: bool) -> np.ndarray:
    """Binary hand mask: skin-colour thresholds for photos, closed and filled edges for drawings."""
    if not binarised:
        return clean_mask(skin_colour_mask(rgb))
    ink = (cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) < 128).astype(np.uint8) * 255  # dark lines
    closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(closed)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return clean_mask(filled)


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


def shape_features(rgb: np.ndarray, binarised: bool) -> np.ndarray:
    """Hu moments plus area, solidity, circularity, extent, aspect and convexity-defect stats.

    All zeros means no hand-sized blob was found; ``empty_mask_rate`` counts those.
    """
    mask = hand_mask(rgb, binarised)
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


def empty_mask_rate(samples: SampleSet) -> float:
    """Share of images in which M1 finds no hand (the failure check of the report's Section 6.2)."""
    rows = (samples[i] for i in range(len(samples)))
    return float(np.mean([not shape_features(s.crop, s.binarised).any() for s in rows]))


class SkinShapeClassifier(FeatureClassifier):
    needs_colour = True  # thresholds work on the colour photo, so T is not applied to its test set

    def __init__(self, name: str, estimator: str) -> None:
        super().__init__(name)
        self._estimator_kind = estimator

    def _features(self, samples: SampleSet) -> np.ndarray:
        rows = (samples[i] for i in range(len(samples)))
        return np.stack([shape_features(s.crop, s.binarised) for s in rows])

    def _candidates(self, features: np.ndarray) -> list:
        if self._estimator_kind == "svm":
            models = [SVC(kernel="rbf", C=c, gamma="scale") for c in SVM_C_GRID]
        else:
            models = [KNeighborsClassifier(k, weights="distance") for k in KNN_NEIGHBOURS]
        return [make_pipeline(StandardScaler(), model) for model in models]


@register("m1_skin_svm")
def _skin_svm() -> SkinShapeClassifier:
    return SkinShapeClassifier("m1_skin_svm", "svm")


@register("m1_skin_knn")
def _skin_knn() -> SkinShapeClassifier:
    return SkinShapeClassifier("m1_skin_knn", "knn")
