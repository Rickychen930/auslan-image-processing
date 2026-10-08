"""Classifier interface, registry and the feature-based template shared by classical methods."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, Self

import numpy as np

from auslan.data.sample_set import SampleSet
from auslan.evaluation.metrics import macro_f1

SVM_C_GRID = (1.0, 10.0, 100.0)  # tuned on the validation images, never on a test set


class Classifier(Protocol):
    """What the runner needs from a method."""

    name: str
    supports_transfer: bool  # True if ``init`` can warm-start ``fit`` (P3); False means P3 == P2
    uses_landmarks: bool  # True if the method is blind without detected hands (M5)
    needs_colour: bool  # True if it works from the colour photo, so style transform T is skipped

    def fit(
        self, train: SampleSet, val: SampleSet | None = None, init: Classifier | None = None
    ) -> Self: ...

    def predict_proba(self, samples: SampleSet) -> np.ndarray:
        """Per-class scores, shape ``(n, n_classes)``, rows summing to 1."""


_FACTORIES: dict[str, Callable[..., Classifier]] = {}
_SEEDED: set[str] = set()  # methods with a random component that accept a ``seed``


def register(
    name: str, seeded: bool = False
) -> Callable[[Callable[..., Classifier]], Callable[..., Classifier]]:
    """Make a method constructible by name; ``seeded`` methods receive the run's seed."""

    def decorate(factory: Callable[..., Classifier]) -> Callable[..., Classifier]:
        if name in _FACTORIES:
            raise ValueError(f"method already registered: {name}")
        _FACTORIES[name] = factory
        if seeded:
            _SEEDED.add(name)
        return factory

    return decorate


def create(name: str, seed: int = 0, **options: object) -> Classifier:
    if name not in _FACTORIES:
        raise KeyError(f"unknown method {name!r}; available: {', '.join(method_names())}")
    if name in _SEEDED:
        options["seed"] = seed
    return _FACTORIES[name](**options)


def method_names() -> list[str]:
    return sorted(_FACTORIES)


def softmax(scores: np.ndarray) -> np.ndarray:
    exp = np.exp(scores - scores.max(axis=1, keepdims=True))
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
    """Hand-crafted features followed by a scikit-learn estimator. It cannot transfer from ASL."""

    supports_transfer = False
    uses_landmarks = False
    needs_colour = False

    def __init__(self, name: str) -> None:
        self.name = name
        self._estimator = None
        self._n_classes = 0

    def _features(self, samples: SampleSet) -> np.ndarray:
        raise NotImplementedError

    def _candidates(self, features: np.ndarray) -> list:
        """Unfitted estimators that differ in one hyper-parameter; the best on validation wins."""
        raise NotImplementedError

    def fit(
        self, train: SampleSet, val: SampleSet | None = None, init: Classifier | None = None
    ) -> Self:
        """``init`` is ignored: hand-crafted features cannot start from ASL training."""
        self._n_classes = len(train.label_space)
        features = self._features(train)
        fitted = [c.fit(features, train.labels) for c in self._candidates(features)]
        self._estimator = fitted[0] if val is None else self._best_on(fitted, val)
        return self

    def predict_proba(self, samples: SampleSet) -> np.ndarray:
        if self._estimator is None:
            raise RuntimeError(f"{self.name} must be fitted before predicting")
        return _scores_to_proba(self._estimator, self._features(samples), self._n_classes)

    def _best_on(self, fitted: list, val: SampleSet):
        if len(fitted) == 1:
            return fitted[0]
        features = self._features(val)
        scores = [
            macro_f1(
                val.labels,
                _scores_to_proba(candidate, features, self._n_classes).argmax(axis=1),
                self._n_classes,
            )
            for candidate in fitted
        ]
        return fitted[int(np.argmax(scores))]  # first one wins a tie
