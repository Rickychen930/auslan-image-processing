"""Shared pre-processing: crop each image to the union of the detected hand boxes.

Detectors are passed in, so the crop logic can be tested without MediaPipe.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

LANDMARKS_PER_HAND = 21
MAX_HANDS = 2
# Look for more hands than are kept: Roboflow frames often show a second signer or a close-up
# inset, and the crop must cover the two hands of one signer, not one hand from each.
DETECT_HANDS = 4
MAX_PAIR_SIZE_RATIO = 2.0  # one signer's hands are of similar size (99% of 2-hand frames: < 2)
MAX_PAIR_GAP = 3.0  # wrist distance in hand sizes (99% of 2-hand frames: < 2.3)
DEFAULT_PADDING = 0.20
Box = tuple[int, int, int, int]  # x0, y0, x1, y1 in pixels


@dataclass(frozen=True)
class HandDetection:
    hands: tuple[np.ndarray, ...]  # each (21, 3): x, y normalised to the image, z relative depth


@dataclass(frozen=True)
class CropResult:
    crop: np.ndarray  # (size, size, 3) uint8
    landmarks: np.ndarray  # (2, 21, 3) float32, ordered left to right by wrist, zero-filled
    n_hands: int
    used_fallback: bool


class HandDetector(Protocol):
    def detect(self, rgb: np.ndarray) -> HandDetection: ...


class FaceDetector(Protocol):
    def faces(self, rgb: np.ndarray) -> Sequence[Box]: ...


class RegionFinder(Protocol):
    def box(self, rgb: np.ndarray) -> Box | None:
        """Pixel box of the region of interest, or ``None`` to keep the whole frame."""


def _pad_box(box: Box, padding: float, height: int, width: int) -> Box:
    x0, y0, x1, y1 = box
    pad_x, pad_y = padding * (x1 - x0), padding * (y1 - y0)
    return (
        max(0, int(np.floor(x0 - pad_x))),
        max(0, int(np.floor(y0 - pad_y))),
        min(width, int(np.ceil(x1 + pad_x))),
        min(height, int(np.ceil(y1 + pad_y))),
    )


def union_box(hands: Sequence[np.ndarray], padding: float, height: int, width: int) -> Box:
    """Padded pixel box around all landmarks of all given hands, clamped to the image."""
    points = np.concatenate([h[:, :2] for h in hands])
    raw = (
        int(np.floor(points[:, 0].min() * width)),
        int(np.floor(points[:, 1].min() * height)),
        int(np.ceil(points[:, 0].max() * width)),
        int(np.ceil(points[:, 1].max() * height)),
    )
    return _pad_box(raw, padding, height, width)


def signer_hands(hands: Sequence[np.ndarray], height: int, width: int) -> list[np.ndarray]:
    """The hands of one signer: the largest plausible pair, else the largest single hand.

    Hands are paired greedily by wrist distance, so two signers side by side are not mixed up;
    a pair must be of similar size and close together. The largest pair is the signer nearest
    the camera, or the close-up inset of a teaching video.
    """
    if len(hands) <= 1:
        return list(hands)
    pixels = np.array([width, height], np.float32)
    points = [h[:, :2] * pixels for h in hands]
    sizes = [float(np.hypot(*(p.max(0) - p.min(0)))) or 1.0 for p in points]
    candidates = []
    for i in range(len(hands)):
        for j in range(i + 1, len(hands)):
            small, large = sorted((sizes[i], sizes[j]))
            gap = float(np.linalg.norm(points[i][0] - points[j][0])) / ((small + large) / 2)
            if large / small <= MAX_PAIR_SIZE_RATIO and gap <= MAX_PAIR_GAP:
                candidates.append((gap, i, j))
    pairs, used = [], set()
    for _, i, j in sorted(candidates):
        if i not in used and j not in used:
            pairs.append((i, j))
            used.update((i, j))
    if pairs:
        i, j = max(pairs, key=lambda pair: sizes[pair[0]] + sizes[pair[1]])
        return [hands[i], hands[j]]
    return [hands[int(np.argmax(sizes))]]


class FullFrame:
    """Fallback for binarised images, where no skin or face can be told apart."""

    def box(self, rgb: np.ndarray) -> Box | None:
        return None


HSV_SKIN = (np.array([0, 30, 60], np.uint8), np.array([25, 255, 255], np.uint8))
YCRCB_SKIN = (np.array([0, 133, 77], np.uint8), np.array([255, 173, 127], np.uint8))


def skin_colour_mask(rgb: np.ndarray) -> np.ndarray:
    """Pixels that fall inside both the HSV and the YCrCb skin ranges (0 or 255)."""
    hsv = cv2.inRange(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV), *HSV_SKIN)
    ycrcb = cv2.inRange(cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb), *YCRCB_SKIN)
    return cv2.bitwise_and(hsv, ycrcb)


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """Remove specks, then close small holes (3 x 3 morphological opening and closing)."""
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)


class SkinRegions:
    """Fallback for colour images: the two largest skin-coloured blobs, with faces masked out."""

    MIN_AREA_FRACTION = 0.005

    def __init__(self, faces: FaceDetector | None = None) -> None:
        self._faces = faces

    def box(self, rgb: np.ndarray) -> Box | None:
        mask = clean_mask(self._skin_mask(rgb))
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        min_area = self.MIN_AREA_FRACTION * mask.size
        blobs = [s for s in stats[1:] if s[cv2.CC_STAT_AREA] >= min_area]
        blobs = sorted(blobs, key=lambda s: s[cv2.CC_STAT_AREA], reverse=True)[:MAX_HANDS]
        if not blobs:
            return None
        x0 = min(s[cv2.CC_STAT_LEFT] for s in blobs)
        y0 = min(s[cv2.CC_STAT_TOP] for s in blobs)
        x1 = max(s[cv2.CC_STAT_LEFT] + s[cv2.CC_STAT_WIDTH] for s in blobs)
        y1 = max(s[cv2.CC_STAT_TOP] + s[cv2.CC_STAT_HEIGHT] for s in blobs)
        return int(x0), int(y0), int(x1), int(y1)

    def _skin_mask(self, rgb: np.ndarray) -> np.ndarray:
        mask = skin_colour_mask(rgb)
        if self._faces is not None:
            for x0, y0, x1, y1 in self._faces.faces(rgb):
                mask[max(0, y0) : y1, max(0, x0) : x1] = 0
        return mask


class HandCropper:
    """Detects hands, crops to their padded union and returns a square crop plus landmarks."""

    def __init__(
        self,
        detector: HandDetector,
        fallback: RegionFinder,
        padding: float = DEFAULT_PADDING,
        size: int = 224,
    ) -> None:
        self._detector = detector
        self._fallback = fallback
        self._padding = padding
        self._size = size

    def __call__(self, rgb: np.ndarray) -> CropResult:
        height, width = rgb.shape[:2]
        detected = signer_hands(self._detector.detect(rgb).hands, height, width)
        hands = sorted(detected, key=lambda h: float(h[0, 0]))  # wrist x, not handedness label
        landmarks = np.zeros((MAX_HANDS, LANDMARKS_PER_HAND, 3), np.float32)
        # MediaPipe gives x and z as fractions of the width and y of the height. Rescale x and z
        # to the height so distances and angles are not skewed on non-square photos.
        units = np.array([width / height, 1.0, width / height], np.float32)
        for slot, hand in enumerate(hands):
            landmarks[slot] = hand * units
        if hands:
            box = union_box(hands, self._padding, height, width)
            return CropResult(self._square_crop(rgb, box), landmarks, len(hands), False)
        region = self._fallback.box(rgb)
        if region is None:  # keep the whole frame, squeezed to a square instead of padded
            whole = cv2.resize(rgb, (self._size, self._size), interpolation=cv2.INTER_AREA)
            return CropResult(whole, landmarks, 0, True)
        box = _pad_box(region, self._padding, height, width)
        return CropResult(self._square_crop(rgb, box), landmarks, 0, True)

    def _square_crop(self, rgb: np.ndarray, box: Box) -> np.ndarray:
        height, width = rgb.shape[:2]
        x0, y0, x1, y1 = box
        side = max(x1 - x0, y1 - y0, 1)
        left = int(round((x0 + x1) / 2 - side / 2))
        top = int(round((y0 + y1) / 2 - side / 2))
        pad_l, pad_t = max(0, -left), max(0, -top)
        pad_r, pad_b = max(0, left + side - width), max(0, top + side - height)
        padded = cv2.copyMakeBorder(rgb, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE)
        patch = padded[top + pad_t : top + pad_t + side, left + pad_l : left + pad_l + side]
        return cv2.resize(patch, (self._size, self._size), interpolation=cv2.INTER_AREA)
