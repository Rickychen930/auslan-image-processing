"""Classifier interface, registry and the feature-based template shared by classical methods."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, Self

import numpy as np

from auslan.data.sample_set import SampleSet


class Classifier(Protocol):
    """What the protocol runner needs from any method (Liskov: all five are interchangeable)."""

    name: str
    supports_transfer: bool  # True if ``init`` can warm-start ``fit`` (P3), False for P3 == P2

    def fit(
        self, train: SampleSet, val: SampleSet | None = None, init: Classifier | None = None
    ) -> Self: ...

    def predict_proba(self, samples: SampleSet) -> np.ndarray:
        """Per-class scores, shape ``(n, n_classes)``, rows summing to 1."""


_FACTORIES: dict[str, Callable[..., Classifier]] = {}


def register(name: str) -> Callable[[Callable[..., Classifier]], Callable[..., Classifier]]:
    """Decorator that makes a method constructible by name (open for extension, closed for edit)."""

    def decorate(factory: Callable[..., Classifier]) -> Callable[..., Classifier]:
        if name in _FACTORIES:
            raise ValueError(f"method already registered: {name}")
        _FACTORIES[name] = factory
        return factory

    return decorate


def create(name: str, **options: object) -> Classifier:
    if name not in _FACTORIES:
        raise KeyError(f"unknown method {name!r}; available: {', '.join(method_names())}")
    return _FACTORIES[name](**options)


def method_names() -> list[str]:
    return sorted(_FACTORIES)


def softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _scores_to_proba(estimator, features: np.ndarray, n_classes: int) -> np.ndarray:
    """Probabilities from any sklearn estimator; margins are soft-maxed when none are exposed."""
    if hasattr(estimator, "predict_proba"):
        scores = estimator.predict_proba(features)
    else:
        raw = estimator.decision_function(features)
        scores = softmax(np.stack([-raw, raw], axis=1) if raw.ndim == 1 else raw)
    full = np.zeros((len(features), n_classes), dtype=np.float32)
    full[:, estimator.classes_.astype(int)] = scores  # classes absent from training stay at 0
    return full


class FeatureClassifier:
    """Template: hand-crafted features followed by a scikit-learn estimator (no transfer)."""

    supports_transfer = False

    def __init__(self, name: str) -> None:
        self.name = name
        self._estimator = None
        self._n_classes = 0

    def _features(self, samples: SampleSet) -> np.ndarray:
        raise NotImplementedError

    def _make_estimator(self, features: np.ndarray):
        raise NotImplementedError

    def fit(
        self, train: SampleSet, val: SampleSet | None = None, init: Classifier | None = None
    ) -> Self:
        """Fit on ``train``; ``val`` and ``init`` are accepted for interface parity and unused."""
        self._n_classes = len(train.label_space)
        features = self._features(train)
        self._estimator = self._make_estimator(features).fit(features, train.labels)
        return self

    def predict_proba(self, samples: SampleSet) -> np.ndarray:
        if self._estimator is None:
            raise RuntimeError(f"{self.name} must be fitted before predicting")
        return _scores_to_proba(self._estimator, self._features(samples), self._n_classes)
