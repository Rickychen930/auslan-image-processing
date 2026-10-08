import numpy as np
import pandas as pd
from PIL import Image

from auslan.data.dedup import (
    add_hashes,
    drop_exact_duplicates,
    drop_near_duplicates,
    drop_near_duplicates_of,
)


def _noise_image(path, seed: int, jitter: int = 0):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (64, 64, 3)).astype(np.int16)
    if jitter:
        arr = arr + np.random.default_rng(seed + 999).integers(-jitter, jitter + 1, arr.shape)
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(path)
    return str(path)


def _catalog(paths_and_labels) -> pd.DataFrame:
    return pd.DataFrame(paths_and_labels, columns=["path", "label"]).assign(source="demo")


def test_add_hashes_adds_md5_and_phash(tmp_path):
    df = _catalog([(_noise_image(tmp_path / "a.png", 1), "A")])
    out = add_hashes(df, workers=1)
    assert len(out["md5"].iloc[0]) == 32
    assert len(out["phash"].iloc[0]) == 16


def test_exact_duplicates_are_dropped_within_a_label(tmp_path):
    first = _noise_image(tmp_path / "a.png", 1)
    second = tmp_path / "copy.png"
    second.write_bytes((tmp_path / "a.png").read_bytes())
    other = _noise_image(tmp_path / "b.png", 2)
    df = add_hashes(_catalog([(first, "A"), (str(second), "A"), (other, "A")]), workers=1)
    assert len(drop_exact_duplicates(df)) == 2


def test_near_duplicates_dropped_but_distinct_images_kept(tmp_path):
    base = _noise_image(tmp_path / "base.png", 1)
    near = _noise_image(tmp_path / "near.png", 1, jitter=3)
    far = _noise_image(tmp_path / "far.png", 2)
    df = add_hashes(_catalog([(base, "A"), (near, "A"), (far, "A")]), workers=1)
    kept = drop_near_duplicates(df, max_hamming=6)
    assert sorted(kept["path"]) == sorted([base, far])


def test_near_duplicates_are_only_compared_within_a_label(tmp_path):
    base = _noise_image(tmp_path / "base.png", 1)
    near = _noise_image(tmp_path / "near.png", 1, jitter=3)
    df = add_hashes(_catalog([(base, "A"), (near, "B")]), workers=1)
    assert len(drop_near_duplicates(df, max_hamming=6)) == 2


def test_drop_near_duplicates_of_reference_checks_across_labels(tmp_path):
    ref = add_hashes(_catalog([(_noise_image(tmp_path / "r.png", 1), "A")]), workers=1)
    clash = _noise_image(tmp_path / "c.png", 1, jitter=3)
    fresh = _noise_image(tmp_path / "f.png", 5)
    test = add_hashes(_catalog([(clash, "B"), (fresh, "B")]), workers=1)
    kept = drop_near_duplicates_of(test, ref, max_hamming=6)
    assert list(kept["path"]) == [fresh]
