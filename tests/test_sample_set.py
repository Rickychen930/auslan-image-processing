import numpy as np
import pandas as pd
import pytest
from PIL import Image

from auslan.data.classes import LabelSpace
from auslan.data.sample_set import SampleSet

SPACE = LabelSpace(("A", "B"))


@pytest.fixture
def prepared(tmp_path):
    rows = []
    for i, label in enumerate(["A", "B", "A"]):
        path = tmp_path / f"{i}.jpg"
        Image.fromarray(np.full((224, 224, 3), 60 * (i + 1), np.uint8)).save(path)
        rows.append((str(path), label, i))
    frame = pd.DataFrame(rows, columns=["crop_path", "label", "lm_index"])
    landmarks = np.arange(3 * 2 * 21 * 3, dtype=np.float32).reshape(3, 2, 21, 3)
    return frame, landmarks


def test_len_labels_and_sample_shapes(prepared):
    frame, landmarks = prepared
    ds = SampleSet(frame, landmarks, SPACE, image_size=64)
    assert len(ds) == 3
    assert ds.labels.tolist() == [0, 1, 0]
    sample = ds[1]
    assert sample.crop.shape == (64, 64, 3) and sample.crop.dtype == np.uint8
    assert np.array_equal(sample.landmarks, landmarks[1])


def test_bulk_landmarks_follow_frame_order(prepared):
    frame, landmarks = prepared
    ds = SampleSet(frame.iloc[[2, 0]], landmarks, SPACE, image_size=32)
    assert np.array_equal(ds.landmark_array(), landmarks[[2, 0]])


def test_transform_is_applied_to_crops(prepared):
    frame, landmarks = prepared
    ds = SampleSet(frame, landmarks, SPACE, image_size=32, transform=lambda a: np.zeros_like(a))
    assert not ds[0].crop.any()


def test_subset_preserves_alignment(prepared):
    frame, landmarks = prepared
    ds = SampleSet(frame, landmarks, SPACE, image_size=32)
    sub = ds.subset([2, 1])
    assert sub.labels.tolist() == [0, 1]
    assert np.array_equal(sub[0].landmarks, landmarks[2])


def test_unknown_label_raises(prepared):
    frame, landmarks = prepared
    with pytest.raises(KeyError):
        SampleSet(frame.assign(label="Q"), landmarks, SPACE, image_size=32)
