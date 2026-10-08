import json

import pytest

from auslan.data import download
from auslan.data.download import organise_coco_by_class


def _coco_export(root, split="train"):
    folder = root / split
    folder.mkdir(parents=True)
    for name in ("one.jpg", "two.jpg", "three.jpg"):
        (folder / name).write_bytes(b"jpeg")
    annotations = {
        "categories": [{"id": 1, "name": "a"}, {"id": 2, "name": "B"}, {"id": 3, "name": "0"}],
        "images": [
            {"id": 10, "file_name": "one.jpg"},
            {"id": 11, "file_name": "two.jpg"},
            {"id": 12, "file_name": "three.jpg"},
        ],
        "annotations": [
            {"image_id": 10, "category_id": 1},
            {"image_id": 11, "category_id": 2},
            {"image_id": 12, "category_id": 3},  # a digit: not part of the fingerspelling set
        ],
    }
    (folder / "_annotations.coco.json").write_text(json.dumps(annotations))


def test_coco_images_are_filed_under_their_letter(tmp_path):
    _coco_export(tmp_path / "export")
    copied = organise_coco_by_class(tmp_path / "export", tmp_path / "out")
    assert copied == 2
    assert [p.name for p in (tmp_path / "out" / "A").iterdir()] == ["train_one.jpg"]
    assert [p.name for p in (tmp_path / "out" / "B").iterdir()] == ["train_two.jpg"]
    assert not (tmp_path / "out" / "0").exists()


def test_coco_splits_do_not_overwrite_each_other(tmp_path):
    _coco_export(tmp_path / "export", "train")
    _coco_export(tmp_path / "export", "valid")
    assert organise_coco_by_class(tmp_path / "export", tmp_path / "out") == 4


def test_roboflow_needs_an_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv(download.ROBOFLOW_KEY_VARIABLE, raising=False)
    with pytest.raises(RuntimeError, match="ROBOFLOW_API_KEY"):
        download.download_roboflow(tmp_path)


def test_unsafe_archive_paths_are_rejected(tmp_path):
    import zipfile

    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape.txt", "no")
    with pytest.raises(RuntimeError, match="unsafe"):
        download._safe_extract(archive, tmp_path / "target")


def test_interrupted_download_leaves_no_file_behind(tmp_path, monkeypatch):
    def broken_transfer(url, destination):
        destination.write_bytes(b"half")
        raise OSError("connection lost")

    monkeypatch.setattr(download.urllib.request, "urlretrieve", broken_transfer)
    target = tmp_path / "model.task"
    with pytest.raises(OSError, match="connection lost"):
        download._fetch("https://example.invalid/model.task", target)
    assert not target.exists()


def _fake_api(monkeypatch, responses):
    """Make _api_get answer from a dict of path -> payload; unknown paths raise like a 404."""

    def get(path, api_key):
        if path not in responses:
            raise RuntimeError(f"Roboflow answered HTTP 404 for /{path}")
        return responses[path]

    monkeypatch.setattr(download, "_api_get", get)


def test_newest_roboflow_version_is_used_when_none_is_given(monkeypatch):
    project = f"{download.ROBOFLOW_WORKSPACE}/{download.ROBOFLOW_PROJECT}"
    _fake_api(
        monkeypatch, {project: {"versions": [{"id": f"{project}/1"}, {"id": f"{project}/3"}]}}
    )
    assert download._latest_version("key") == 3


def test_roboflow_without_versions_asks_for_one(monkeypatch):
    project = f"{download.ROBOFLOW_WORKSPACE}/{download.ROBOFLOW_PROJECT}"
    _fake_api(monkeypatch, {project: {"versions": []}})
    with pytest.raises(RuntimeError, match="--roboflow-version"):
        download._latest_version("key")


def test_export_link_is_read_from_either_response_shape(monkeypatch):
    path = f"{download.ROBOFLOW_WORKSPACE}/{download.ROBOFLOW_PROJECT}/2/coco"
    _fake_api(monkeypatch, {path: {"export": {"link": "https://example.invalid/a.zip"}}})
    assert download._export_link("key", 2).endswith("a.zip")
    _fake_api(
        monkeypatch, {path: {"version": {"export": {"link": "https://example.invalid/b.zip"}}}}
    )
    assert download._export_link("key", 2).endswith("b.zip")


def test_api_errors_do_not_expose_the_key(monkeypatch):
    def refuse(url, *args, **kwargs):
        raise download.urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    monkeypatch.setattr(download.urllib.request, "urlopen", refuse)
    with pytest.raises(RuntimeError, match="404") as caught:
        download._api_get("some/path", "SECRET-KEY")
    assert "SECRET-KEY" not in str(caught.value) and caught.value.__cause__ is None
