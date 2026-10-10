"""MediaPipe Tasks adapters for the ``HandDetector`` and ``FaceDetector`` protocols."""

from __future__ import annotations

from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from auslan.preprocessing.hand_crop import DETECT_HANDS, Box, HandDetection

DEFAULT_CONFIDENCE = 0.3  # permissive: blurred or occluded two-handed signs score low


def _as_mp_image(rgb: np.ndarray) -> mp.Image:
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))


class MediaPipeHandDetector:
    """Hand landmarks (21 points per hand, up to four hands) via the Tasks HandLandmarker."""

    def __init__(self, model_path: Path, confidence: float = DEFAULT_CONFIDENCE) -> None:
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            num_hands=DETECT_HANDS,
            min_hand_detection_confidence=confidence,
            min_hand_presence_confidence=confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, rgb: np.ndarray) -> HandDetection:
        result = self._landmarker.detect(_as_mp_image(rgb))
        hands = tuple(
            np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
            for landmarks in result.hand_landmarks
        )
        return HandDetection(hands=hands)


class MediaPipeFaceDetector:
    """Face boxes in pixels, used to keep faces out of the skin-colour fallback."""

    def __init__(self, model_path: Path) -> None:
        options = vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path))
        )
        self._detector = vision.FaceDetector.create_from_options(options)

    def faces(self, rgb: np.ndarray) -> list[Box]:
        boxes = []
        for detection in self._detector.detect(_as_mp_image(rgb)).detections:
            b = detection.bounding_box
            boxes.append((b.origin_x, b.origin_y, b.origin_x + b.width, b.origin_y + b.height))
        return boxes
