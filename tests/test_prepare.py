from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from auslan.data.prepare import load_prepared, prepare_source
from auslan.data.registry import DatasetSpec
from auslan.preprocessing.hand_crop import FullFrame, HandCropper, HandDetection


class _OneHand:
    def detect(self, rgb):
        pts = np.zeros((21, 3), np.float32)
        pts[:, 0] = np.linspace(0.3, 0.6, 21)
        pts[:, 1] = np.linspace(0.3, 0.6, 21)
        return HandDetection(hands=(pts,))


def fake_cropper() -> HandCropper:
    return HandCropper(_OneHand(), FullFrame(), size=32)


def exploding_cropper() -> HandCropper:
    raise AssertionError("cropper must not be built when everything is cached")


def _make_dataset(root, per_class=12, letters=("A", "B")):
    seed = 0
    for letter in letters:
        (root / letter).mkdir(parents=True)
        for i in range(per_class):
            seed += 1
            arr = np.random.default_rng(seed).integers(0, 256, (48, 48, 3), dtype=np.uint8)
            Image.fromarray(arr).save(root / letter / f"{i}.png")


@pytest.fixture
def spec(tmp_path):
    root = tmp_path / "raw"
    _make_dataset(root)
    return DatasetSpec(name="demo", root=root, letters=("A", "B"), fallback="full", split=True)


def _prepare(spec, tmp_path, factory=fake_cropper, **kwargs):
    return prepare_source(
        spec,
        work_dir=tmp_path / "work",
        cropper_factory=factory,
        seed=0,
        max_hamming=4,
        workers=1,
        chunk_size=10,
        **kwargs,
    )


def test_prepare_writes_catalog_landmarks_and_crops(spec, tmp_path):
    out = _prepare(spec, tmp_path)
    frame = pd.read_csv(out / "catalog.csv")
    landmarks = np.load(out / "landmarks.npy")
    assert len(frame) == 24
    assert set(frame["split"]) == {"train", "val", "test"}
    assert landmarks.shape == (24, 2, 21, 3)
    assert sorted(frame["lm_index"]) == list(range(24))
    assert all(Path(p).exists() for p in frame["crop_path"])
    assert (frame["n_hands"] == 1).all() and not frame["used_fallback"].any()


def test_prepare_is_idempotent_and_does_not_rebuild_cropper(spec, tmp_path):
    first = _prepare(spec, tmp_path)
    again = _prepare(spec, tmp_path, factory=exploding_cropper)
    assert first == again


def test_unsplit_source_is_all_test(tmp_path):
    root = tmp_path / "raw"
    _make_dataset(root, per_class=5)
    spec = DatasetSpec("demo_test", root, ("A", "B"), "full", split=False)
    out = _prepare(spec, tmp_path)
    assert set(pd.read_csv(out / "catalog.csv")["split"]) == {"test"}


def test_reference_overlap_is_removed(spec, tmp_path):
    out = _prepare(spec, tmp_path)
    reference = pd.read_csv(out / "hashed.csv")
    twin_root = tmp_path / "twin"
    (twin_root / "A").mkdir(parents=True)
    first_ref = reference.loc[reference["label"] == "A", "path"].iloc[0]
    Image.open(first_ref).save(twin_root / "A" / "same.png")
    fresh = np.random.default_rng(999).integers(0, 256, (48, 48, 3), dtype=np.uint8)
    Image.fromarray(fresh).save(twin_root / "A" / "fresh.png")
    twin = DatasetSpec("twin", twin_root, ("A", "B"), "full", split=False)
    out2 = _prepare(twin, tmp_path, reference=reference)
    kept = pd.read_csv(out2 / "catalog.csv")
    assert [Path(p).name for p in kept["path"]] == ["fresh.png"]


def test_load_prepared_returns_aligned_frame_and_landmarks(spec, tmp_path):
    out = _prepare(spec, tmp_path)
    frame, landmarks = load_prepared(out)
    assert len(frame) == landmarks.shape[0] == 24
    assert {"crop_path", "lm_index", "split", "label"} <= set(frame.columns)
