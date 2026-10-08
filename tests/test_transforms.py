import numpy as np

from auslan.data.transforms import kaggle_style


def _photo() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)


def test_kaggle_style_keeps_shape_and_is_binary():
    out = kaggle_style(_photo())
    assert out.shape == (64, 64, 3) and out.dtype == np.uint8
    assert set(np.unique(out)) <= {0, 255}


def test_kaggle_style_is_deterministic_and_does_not_mutate_input():
    photo = _photo()
    before = photo.copy()
    assert np.array_equal(kaggle_style(photo), kaggle_style(photo))
    assert np.array_equal(photo, before)
