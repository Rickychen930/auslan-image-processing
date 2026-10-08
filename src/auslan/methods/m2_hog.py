"""M2: HOG descriptors on a greyscale hand crop, reduced with PCA, then an SVM or random forest."""

from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from auslan.data.sample_set import SampleSet
from auslan.methods.base import SVM_C_GRID, FeatureClassifier, register

ORIENTATIONS = 9
PCA_COMPONENTS = 128


class HogClassifier(FeatureClassifier):
    def __init__(
        self, name: str, estimator: str, cell_size: int = 8, kernel: str = "rbf", seed: int = 0
    ) -> None:
        super().__init__(name)
        self._estimator_kind = estimator
        self._cell_size = cell_size
        self._kernel = kernel
        self._seed = seed

    def _features(self, samples: SampleSet) -> np.ndarray:
        return np.stack([self._describe(samples[i].crop) for i in range(len(samples))])

    def _describe(self, rgb: np.ndarray) -> np.ndarray:
        grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return hog(
            grey,
            orientations=ORIENTATIONS,
            pixels_per_cell=(self._cell_size, self._cell_size),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
        ).astype(np.float32)

    def _candidates(self, features: np.ndarray) -> list:
        components = min(PCA_COMPONENTS, features.shape[0] - 1, features.shape[1])
        if self._estimator_kind == "svm":
            models = [SVC(kernel=self._kernel, C=c, gamma="scale") for c in SVM_C_GRID]
        else:
            models = [RandomForestClassifier(300, n_jobs=-1, random_state=self._seed)]
        return [
            make_pipeline(PCA(n_components=components, random_state=self._seed), model)
            for model in models
        ]


@register("m2_hog_svm")
def _hog_svm(cell_size: int = 8, kernel: str = "rbf") -> HogClassifier:
    return HogClassifier("m2_hog_svm", "svm", cell_size, kernel)


@register("m2_hog_rf", seeded=True)
def _hog_rf(cell_size: int = 8, seed: int = 0) -> HogClassifier:
    return HogClassifier("m2_hog_rf", "rf", cell_size, seed=seed)
