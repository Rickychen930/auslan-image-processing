from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from auslan.data.catalog import scan_folder_dataset
from auslan.data.classes import ASL_LETTERS, AUSLAN_LETTERS, LabelSpace, letter_from_name


def _write_image(path: Path, value: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((8, 8, 3), value, np.uint8)).save(path)


def test_asl_letters_drop_motion_letters():
    assert len(ASL_LETTERS) == 24
    assert "J" not in ASL_LETTERS and "Z" not in ASL_LETTERS
    assert len(AUSLAN_LETTERS) == 26


@pytest.mark.parametrize(
    ("name", "expected"),
    [("a", "A"), ("B", "B"), ("0", None), ("del", None), ("nothing", None), ("space", None)],
)
def test_letter_from_name(name, expected):
    assert letter_from_name(name) == expected


def test_label_space_round_trip():
    space = LabelSpace(("A", "B", "C"))
    assert len(space) == 3
    assert space.index("B") == 1
    assert space.letter(2) == "C"
    with pytest.raises(KeyError):
        space.index("Z")


def test_scan_keeps_only_requested_letters_and_ignores_digits(tmp_path):
    for folder in ("A", "b", "J", "0", "del"):
        _write_image(tmp_path / folder / "x.jpg")
    df = scan_folder_dataset(tmp_path, source="demo", letters=ASL_LETTERS)
    assert sorted(df["label"]) == ["A", "B"]
    assert set(df["source"]) == {"demo"}
    assert all(Path(p).is_absolute() for p in df["path"])


def test_scan_finds_nested_class_folders(tmp_path):
    _write_image(tmp_path / "train" / "train" / "C" / "1.jpg")
    df = scan_folder_dataset(tmp_path, source="demo", letters=AUSLAN_LETTERS)
    assert list(df["label"]) == ["C"]


def test_scan_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        scan_folder_dataset(tmp_path / "missing", source="demo", letters=ASL_LETTERS)
