"""Per-frame timing on the CPU, the way a laptop tutoring app would run."""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from auslan.data.sample_set import SampleSet
from auslan.preprocessing.hand_crop import HandCropper

DEFAULT_FRAMES = 50


def classify_ms(model, samples: SampleSet, frames: int = DEFAULT_FRAMES, seed: int = 0) -> float:
    """Mean milliseconds to classify one already-cropped sample (features + model)."""
    if hasattr(model, "use_device"):
        model.use_device("cpu")
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(samples), size=min(frames, len(samples)), replace=False)
    singles = [samples.subset([int(i)]) for i in picks]
    model.predict_proba(singles[0])  # warm-up: lazy initialisation should not be timed
    start = time.perf_counter()
    for single in singles:
        model.predict_proba(single)
    return (time.perf_counter() - start) / len(singles) * 1000


def crop_ms(
    cropper: HandCropper, paths: Sequence[str | Path], frames: int = DEFAULT_FRAMES
) -> float:
    """Mean milliseconds to read an image and run hand detection and cropping on it."""
    images = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in paths[:frames]]
    cropper(images[0])
    start = time.perf_counter()
    for image in images:
        cropper(image)
    return (time.perf_counter() - start) / len(images) * 1000
