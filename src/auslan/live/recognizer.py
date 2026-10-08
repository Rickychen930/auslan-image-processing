"""Turn camera frames into letter probabilities with a model saved by ``auslan export``."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import joblib
import numpy as np

from auslan.data.classes import LabelSpace
from auslan.data.sample_set import SampleSet
from auslan.methods import Classifier
from auslan.preprocessing.hand_crop import LANDMARKS_PER_HAND, MAX_HANDS, HandCropper

TOP_K = 3
MIN_CONFIDENCE = 0.5  # below this a matching letter does not count as correct
SMOOTHING_FRAMES = 5  # predictions are averaged over this many frames to stop flicker
FRAME_SIZE = 224  # side of the squeezed whole frame, as for the cached Kaggle crops


@dataclass
class LiveModel:
    """A trained classifier plus what it needs to be fed from a camera."""

    classifier: Classifier
    letters: tuple[str, ...]
    input_mode: str  # "hands": crop to the detected hands; "frame": whole frame drawn like Kaggle
    image_size: int
    description: str


@dataclass(frozen=True)
class Verdict:
    status: str  # "idle", "no-hand", "correct" or "wrong"
    message: str


def save_live_model(model: LiveModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_live_model(path: Path) -> LiveModel:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; train it first with `auslan export`")
    return joblib.load(path)


class Recognizer:
    """Classifies camera frames. Frames are never mirrored here: a flip swaps the two hands."""

    def __init__(
        self, model: LiveModel, cropper: HandCropper | None, smoothing: int = SMOOTHING_FRAMES
    ) -> None:
        if model.input_mode == "hands" and cropper is None:
            raise ValueError("a model that reads hands needs a hand cropper")
        self._model = model
        self._cropper = cropper
        self._history: deque[np.ndarray] = deque(maxlen=smoothing)
        self.last_input: np.ndarray | None = None  # the image the model actually saw

    def proba(self, rgb: np.ndarray) -> np.ndarray | None:
        """Probabilities averaged over recent frames, or ``None`` when a needed hand is missing."""
        if self._model.input_mode == "hands":
            found = self._cropper(rgb)
            if self._model.classifier.uses_landmarks and found.n_hands == 0:
                self._history.clear()
                return None
            crop, landmarks, styled = found.crop, found.landmarks, False
        else:
            crop = cv2.resize(rgb, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
            landmarks = np.zeros((MAX_HANDS, LANDMARKS_PER_HAND, 3), np.float32)
            styled = True
        samples = SampleSet.from_memory(
            crop, landmarks, LabelSpace(self._model.letters), self._model.image_size, styled
        )
        self.last_input = samples[0].crop
        self._history.append(self._model.classifier.predict_proba(samples)[0])
        return np.mean(self._history, axis=0)


def top_letters(
    letters: tuple[str, ...], proba: np.ndarray, k: int = TOP_K
) -> list[tuple[str, float]]:
    best = np.argsort(-proba, kind="stable")[:k]
    return [(letters[i], float(proba[i])) for i in best]


def judge(letters: tuple[str, ...], proba: np.ndarray | None, target: str | None) -> Verdict:
    """Say what is seen and, when the learner aims for a ``target`` letter, whether it matches."""
    if proba is None:
        return Verdict("no-hand", "Show your hands to the camera")
    letter, confidence = top_letters(letters, proba, 1)[0]
    if target is None:
        return Verdict("idle", f"I see {letter} ({confidence:.0%})")
    if letter == target and confidence >= MIN_CONFIDENCE:
        return Verdict("correct", f"Correct: {target}")
    return Verdict("wrong", f"Not yet: I see {letter} ({confidence:.0%}), you aim for {target}")
