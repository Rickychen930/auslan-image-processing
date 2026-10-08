import numpy as np
import pytest

from auslan.preprocessing.hand_crop import (
    FullFrame,
    HandCropper,
    HandDetection,
    SkinRegions,
    union_box,
)


class FakeDetector:
    def __init__(self, hands):
        self.hands = hands  # list of (21, 3) arrays in normalised image coords

    def detect(self, rgb):
        return HandDetection(hands=tuple(self.hands))


def _hand(cx: float, cy: float, spread: float = 0.1) -> np.ndarray:
    points = np.zeros((21, 3), np.float32)
    points[:, 0] = cx + np.linspace(-spread, spread, 21)
    points[:, 1] = cy + np.linspace(-spread, spread, 21)
    return points


def test_union_box_pads_and_clamps_to_image():
    box = union_box([_hand(0.5, 0.5, 0.1)], padding=0.2, height=100, width=200)
    x0, y0, x1, y1 = box
    assert 0 <= x0 < x1 <= 200 and 0 <= y0 < y1 <= 100
    assert x1 - x0 > 0.2 * 200  # wider than the raw 20% span because of padding
    edge = union_box([_hand(0.0, 0.0, 0.1)], padding=0.2, height=100, width=100)
    assert edge[0] == 0 and edge[1] == 0


def test_two_hands_are_ordered_left_to_right_and_zero_fill_missing():
    rgb = np.zeros((100, 100, 3), np.uint8)
    cropper = HandCropper(FakeDetector([_hand(0.7, 0.5), _hand(0.3, 0.5)]), FullFrame(), size=32)
    result = cropper(rgb)
    assert result.n_hands == 2 and not result.used_fallback
    assert result.crop.shape == (32, 32, 3)
    assert result.landmarks.shape == (2, 21, 3)
    assert result.landmarks[0, :, 0].mean() < result.landmarks[1, :, 0].mean()

    one = HandCropper(FakeDetector([_hand(0.5, 0.5)]), FullFrame(), size=32)(rgb)
    assert one.n_hands == 1
    assert np.any(one.landmarks[0]) and not np.any(one.landmarks[1])


def test_extra_detections_are_truncated_to_two():
    rgb = np.zeros((50, 50, 3), np.uint8)
    hands = [_hand(0.2, 0.5), _hand(0.5, 0.5), _hand(0.8, 0.5)]
    assert HandCropper(FakeDetector(hands), FullFrame(), size=16)(rgb).n_hands == 2


def test_no_hand_uses_full_frame_fallback():
    rgb = np.full((40, 80, 3), 128, np.uint8)
    result = HandCropper(FakeDetector([]), FullFrame(), size=24)(rgb)
    assert result.used_fallback and result.n_hands == 0
    assert result.crop.shape == (24, 24, 3)
    assert not np.any(result.landmarks)


def test_skin_fallback_focuses_on_skin_coloured_blob():
    rgb = np.full((100, 100, 3), 20, np.uint8)  # dark background
    rgb[40:70, 10:40] = (224, 172, 140)  # skin-toned patch
    box = SkinRegions().box(rgb)
    assert box is not None
    x0, y0, x1, y1 = box
    assert x0 <= 12 and x1 >= 38 and y0 <= 42 and y1 >= 68
    assert x1 - x0 < 90  # does not fall back to the whole frame


def test_skin_fallback_returns_none_without_skin():
    assert SkinRegions().box(np.full((60, 60, 3), 10, np.uint8)) is None


def test_skin_fallback_masks_out_detected_faces():
    class FaceOnPatch:
        def faces(self, rgb):
            return [(0, 30, 50, 80)]

    rgb = np.full((100, 100, 3), 20, np.uint8)
    rgb[40:70, 10:40] = (224, 172, 140)  # the only skin region is inside the "face"
    assert SkinRegions(FaceOnPatch()).box(rgb) is None


def test_cropper_rejects_non_positive_size():
    with pytest.raises(ValueError):
        HandCropper(FakeDetector([]), FullFrame(), size=0)
