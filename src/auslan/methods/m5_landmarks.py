"""M5: classify from the 2 x 21 MediaPipe hand landmarks instead of pixels."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import Dataset, TensorDataset

from auslan.data.sample_set import SampleSet
from auslan.methods.base import Classifier, FeatureClassifier, register
from auslan.methods.torch_base import TorchClassifier

WRIST, MIDDLE_MCP = 0, 9
FINGERTIPS = (4, 8, 12, 16, 20)
# (a, b, c): the angle at joint b between bones b->a and b->c, three per finger
JOINT_TRIPLES = tuple(
    (a, b, c)
    for base in (1, 5, 9, 13, 17)
    for a, b, c in ((0, base, base + 1), (base, base + 1, base + 2), (base + 1, base + 2, base + 3))
)
EPS = 1e-6


def _hand_block(hand: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-hand features, the hand size used for scaling and a presence flag.

    Coordinates are shifted to the wrist and divided by the wrist-to-middle-knuckle distance,
    so the block does not depend on where the hand is or how far it is from the camera.
    """
    present = np.any(hand != 0, axis=(1, 2))
    size = np.linalg.norm(hand[:, MIDDLE_MCP, :2] - hand[:, WRIST, :2], axis=1)
    size = np.where(size > EPS, size, 1.0)
    relative = (hand - hand[:, :1, :]) / size[:, None, None]

    a, b, c = (np.array(idx) for idx in zip(*JOINT_TRIPLES, strict=True))
    bone_1, bone_2 = hand[:, a] - hand[:, b], hand[:, c] - hand[:, b]
    cosine = (bone_1 * bone_2).sum(-1) / (
        np.linalg.norm(bone_1, axis=-1) * np.linalg.norm(bone_2, axis=-1) + EPS
    )
    angles = np.arccos(np.clip(cosine, -1, 1)) / np.pi

    block = np.concatenate([relative.reshape(len(hand), -1), angles], axis=1)
    return block * present[:, None], size, present


def _contact_features(hands: np.ndarray, sizes: np.ndarray, both: np.ndarray) -> np.ndarray:
    """Fingertip-to-fingertip distances and wrist offset between the two hands.

    These describe where the hands touch, which is what separates many Auslan letters.
    """
    scale = sizes.mean(axis=1)
    tips = hands[:, :, FINGERTIPS, :2]  # (n, 2, 5, 2)
    gaps = np.linalg.norm(tips[:, 0, :, None] - tips[:, 1, None, :], axis=-1).reshape(
        len(hands), -1
    )
    wrist_offset = hands[:, 1, WRIST, :2] - hands[:, 0, WRIST, :2]
    features = np.concatenate(
        [gaps, wrist_offset, np.linalg.norm(wrist_offset, axis=1, keepdims=True)], axis=1
    )
    return features / scale[:, None] * both[:, None]


def landmark_features(landmarks: np.ndarray, two_hand_distances: bool = True) -> np.ndarray:
    """Feature matrix for landmarks of shape ``(n, 2, 21, 3)``; absent hands give zeros."""
    blocks, sizes, present = zip(
        *(_hand_block(landmarks[:, hand]) for hand in range(landmarks.shape[1])), strict=True
    )
    parts = [*blocks, np.stack(present, axis=1).astype(np.float64)]
    if two_hand_distances:
        both = np.logical_and(*present)
        parts.append(_contact_features(landmarks, np.stack(sizes, axis=1), both))
    return np.concatenate(parts, axis=1).astype(np.float32)


class LandmarkClassifier(FeatureClassifier):
    uses_landmarks = True

    def __init__(
        self, name: str, estimator: str, two_hand_distances: bool = True, seed: int = 0
    ) -> None:
        super().__init__(name)
        self._estimator_kind = estimator
        self._two_hand = two_hand_distances
        self._seed = seed

    def _features(self, samples: SampleSet) -> np.ndarray:
        return landmark_features(samples.landmark_array(), self._two_hand)

    def _make_estimator(self, features: np.ndarray):
        if self._estimator_kind == "svm":
            return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10.0, gamma="scale"))
        return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=self._seed)


class _Mlp(nn.Module):
    def __init__(self, n_features: int, n_classes: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(n_features, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 128), nn.ReLU()
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):
        return self.head(self.body(x))


class LandmarkMlp(TorchClassifier):
    """MLP on landmark features; it can start from ASL weights (the second hand is all zeros)."""

    head_prefixes = ("head",)
    uses_landmarks = True

    def __init__(self, name: str, two_hand_distances: bool = True, **options: object) -> None:
        super().__init__(name, **options)
        self._two_hand = two_hand_distances
        self._scaler = StandardScaler()

    def _build(self, n_classes: int) -> nn.Module:
        return _Mlp(self._n_features, n_classes)

    def _before_fit(self, train: SampleSet, init: Classifier | None) -> None:
        features = landmark_features(train.landmark_array(), self._two_hand)
        self._n_features = features.shape[1]
        if isinstance(init, LandmarkMlp):
            self._scaler = init._scaler  # pretrained weights expect the ASL feature scaling
        else:
            self._scaler.fit(features)

    def _dataset(self, samples: SampleSet, train: bool) -> Dataset:
        features = landmark_features(samples.landmark_array(), self._two_hand)
        scaled = torch.from_numpy(self._scaler.transform(features).astype(np.float32))
        return TensorDataset(scaled, torch.from_numpy(samples.labels))


@register("m5_landmarks_svm")
def _landmarks_svm(two_hand_distances: bool = True) -> LandmarkClassifier:
    return LandmarkClassifier("m5_landmarks_svm", "svm", two_hand_distances)


@register("m5_landmarks_rf", seeded=True)
def _landmarks_rf(two_hand_distances: bool = True, seed: int = 0) -> LandmarkClassifier:
    return LandmarkClassifier("m5_landmarks_rf", "rf", two_hand_distances, seed)


@register("m5_landmarks_mlp", seeded=True)
def _landmarks_mlp(**options: object) -> LandmarkMlp:
    return LandmarkMlp("m5_landmarks_mlp", **options)
