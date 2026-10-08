"""Build the cropper used by ``prepare``; kept at module level so worker processes can pickle it."""

from __future__ import annotations

from pathlib import Path

from auslan.preprocessing.hand_crop import FullFrame, HandCropper, SkinRegions
from auslan.preprocessing.mediapipe_detector import MediaPipeFaceDetector, MediaPipeHandDetector

HAND_MODEL = "hand_landmarker.task"
FACE_MODEL = "blaze_face_short_range.tflite"


def build_cropper(models_dir: Path, fallback: str, crop_size: int) -> HandCropper:
    """``fallback`` is "skin" for colour photos and "full" for binarised drawings."""
    models_dir = Path(models_dir)
    hands = MediaPipeHandDetector(models_dir / HAND_MODEL)
    if fallback == "skin":
        region = SkinRegions(MediaPipeFaceDetector(models_dir / FACE_MODEL))
    elif fallback == "full":
        region = FullFrame()
    else:
        raise ValueError(f"unknown fallback {fallback!r}")
    return HandCropper(hands, region, size=crop_size)
