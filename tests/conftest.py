import cv2
import numpy as np
import pandas as pd
import pytest

from auslan.data.classes import LabelSpace
from auslan.data.sample_set import SampleSet

SKIN = (224, 172, 140)
SIZE = 64


def _draw(class_id: int, rng: np.random.Generator) -> np.ndarray:
    """A skin-coloured square, bar or disc on a dark background, with a little jitter."""
    image = np.full((SIZE, SIZE, 3), 20, np.uint8)
    dx, dy = (int(v) for v in rng.integers(-4, 5, 2))
    if class_id == 0:
        cv2.rectangle(image, (22 + dx, 22 + dy), (42 + dx, 42 + dy), SKIN, -1)
    elif class_id == 1:
        cv2.rectangle(image, (8 + dx, 28 + dy), (56 + dx, 36 + dy), SKIN, -1)
    else:
        cv2.circle(image, (32 + dx, 32 + dy), 12, SKIN, -1)
    return image


def _landmarks(class_id: int, rng: np.random.Generator) -> np.ndarray:
    """Hands with a class-dependent spread; class 1 also shows a second hand."""
    points = np.zeros((2, 21, 3), np.float32)
    spread = (0.05, 0.15, 0.10)[class_id]
    ramp = np.linspace(-spread, spread, 21)
    for hand in range(2 if class_id == 1 else 1):
        points[hand, :, 0] = 0.3 + 0.4 * hand + ramp + rng.normal(0, 0.005, 21)
        points[hand, :, 1] = 0.5 + ramp * (1 + hand) + rng.normal(0, 0.005, 21)
    return points


def _write_prepared(work_dir, name, per_split, seed, letters=("A", "B", "C")):
    """Write a dataset in the layout `prepare` produces: catalog.csv, landmarks.npy, crops."""
    rng = np.random.default_rng(seed)
    folder = work_dir / name
    (folder / "crops").mkdir(parents=True)
    rows, landmarks = [], []
    for split, per_class in per_split.items():
        for class_id, letter in enumerate(letters):
            for _ in range(per_class):
                path = folder / "crops" / f"{len(rows):05d}.png"
                cv2.imwrite(str(path), cv2.cvtColor(_draw(class_id, rng), cv2.COLOR_RGB2BGR))
                rows.append((str(path), letter, name, split, str(path), 1, False, False, len(rows)))
                landmarks.append(_landmarks(class_id, rng))
    columns = [
        "path",
        "label",
        "source",
        "split",
        "crop_path",
        "n_hands",
        "used_fallback",
        "binarised",
        "lm_index",
    ]
    pd.DataFrame(rows, columns=columns).to_csv(folder / "catalog.csv", index=False)
    np.save(folder / "landmarks.npy", np.stack(landmarks))


@pytest.fixture
def prepared_work_dir(tmp_path):
    """A tiny stand-in for the four prepared datasets (letters A, B, C only)."""
    work_dir = tmp_path / "data" / "work"
    _write_prepared(work_dir, "asl_alphabet", {"train": 8, "val": 3, "test": 3}, seed=1)
    _write_prepared(work_dir, "asl_alphabet_test", {"test": 4}, seed=2)
    _write_prepared(work_dir, "auslan_kaggle", {"train": 12, "val": 4, "test": 4}, seed=3)
    _write_prepared(work_dir, "auslan_roboflow", {"test": 4}, seed=4)
    return work_dir


@pytest.fixture
def make_sample_set(tmp_path):
    """Build a separable toy SampleSet; call as ``make_sample_set(n_classes, per_class, seed)``."""

    def build(n_classes: int, per_class: int, seed: int) -> SampleSet:
        rng = np.random.default_rng(seed)
        letters = ("A", "B", "C")[:n_classes]
        rows, landmarks = [], []
        for class_id, letter in enumerate(letters):
            for i in range(per_class):
                path = tmp_path / f"{seed}_{letter}_{i}.png"
                cv2.imwrite(str(path), cv2.cvtColor(_draw(class_id, rng), cv2.COLOR_RGB2BGR))
                rows.append((str(path), letter, len(landmarks)))
                landmarks.append(_landmarks(class_id, rng))
        frame = pd.DataFrame(rows, columns=["crop_path", "label", "lm_index"])
        return SampleSet(frame, np.stack(landmarks), LabelSpace(letters), image_size=SIZE)

    return build
