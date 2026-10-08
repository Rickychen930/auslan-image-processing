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
WARM_UP_CALLS = 3


def classify_ms(model, samples: SampleSet, frames: int = DEFAULT_FRAMES, seed: int = 0) -> float:
    """Mean milliseconds to classify one already-cropped sample (decode, features, model).

    Torch models are moved to the CPU first, so call this after all other predictions.
    """
    if hasattr(model, "use_device"):
        model.use_device("cpu")
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(samples), size=min(frames, len(samples)), replace=False)
    singles = [samples.subset([int(i)]) for i in picks]
    for single in singles[:WARM_UP_CALLS]:
        model.predict_proba(single)
    start = time.perf_counter()
    for single in singles:
        model.predict_proba(single)
    return (time.perf_counter() - start) / len(singles) * 1000


def crop_ms(
    cropper: HandCropper, paths: Sequence[str | Path], frames: int = DEFAULT_FRAMES
) -> float:
    """Mean milliseconds to run hand detection and cropping on an already-loaded image."""
    images = []
    for path in paths[:frames]:
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise FileNotFoundError(f"cannot read image: {path}")
        images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    for image in images[:WARM_UP_CALLS]:
        cropper(image)
    start = time.perf_counter()
    for image in images:
        cropper(image)
    return (time.perf_counter() - start) / len(images) * 1000
