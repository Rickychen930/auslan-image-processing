from pathlib import Path

import pytest

from auslan.config import Settings, load_settings


def test_defaults_match_the_report():
    s = Settings()
    assert s.k_values == (5, 10, 20, 50)
    assert s.seeds == (0, 1, 2)
    assert s.val_per_class == 10
    assert s.raw_dir == Path("data") / "raw"


def test_load_settings_overrides_only_given_keys(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("image_size: 64\nk_values: [5, 20]\ndata_dir: elsewhere\n")
    s = load_settings(cfg)
    assert s.image_size == 64
    assert s.k_values == (5, 20)
    assert s.raw_dir == Path("elsewhere") / "raw"
    assert s.seeds == (0, 1, 2)


def test_load_settings_rejects_unknown_keys(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("imagesize: 64\n")
    with pytest.raises(ValueError, match="imagesize"):
        load_settings(cfg)


def test_load_settings_without_file_returns_defaults():
    assert load_settings(None) == Settings()
