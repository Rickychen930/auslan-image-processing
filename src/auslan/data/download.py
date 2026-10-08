"""Downloads: Kaggle datasets, MediaPipe models and the Roboflow Auslan set.

Credentials are read from the environment or the tools' own config files and never logged.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlencode

from auslan.config import FACE_MODEL, HAND_MODEL
from auslan.data.classes import letter_from_name

log = logging.getLogger(__name__)

KAGGLE_DATASETS = {
    "asl_alphabet_test": "danrasband/asl-alphabet-test",
    "asl_alphabet": "grassknoted/asl-alphabet",
    "auslan_kaggle": "rithwikchhugani/auslan-sign-language-fingerspelling-dataset",
}
_MODEL_BASE = "https://storage.googleapis.com/mediapipe-models"
MODEL_URLS = {
    HAND_MODEL: f"{_MODEL_BASE}/hand_landmarker/hand_landmarker/float16/latest/{HAND_MODEL}",
    FACE_MODEL: f"{_MODEL_BASE}/face_detector/blaze_face_short_range/float16/latest/{FACE_MODEL}",
}
ROBOFLOW_API = "https://api.roboflow.com"
ROBOFLOW_WORKSPACE, ROBOFLOW_PROJECT = "workspaces-xbqqe", "auslan"  # universe.roboflow.com/<both>
ROBOFLOW_KEY_VARIABLE = "ROBOFLOW_API_KEY"


def download_kaggle(raw_dir: Path, names: list[str] | None = None) -> None:
    """Download and unzip the Kaggle datasets that are not on disk yet."""
    from kaggle.api.kaggle_api_extended import KaggleApi  # imported late: needs credentials

    api = KaggleApi()
    api.authenticate()
    for name in names or list(KAGGLE_DATASETS):
        target = Path(raw_dir) / name
        if target.is_dir() and any(target.iterdir()):
            log.info("%s already downloaded", name)
            continue
        log.info("downloading %s (%s)", name, KAGGLE_DATASETS[name])
        partial = target.with_name(f"{name}.downloading")  # renamed only once complete
        shutil.rmtree(partial, ignore_errors=True)
        api.dataset_download_files(
            KAGGLE_DATASETS[name], path=str(partial), unzip=True, quiet=False
        )
        partial.rename(target)


def download_models(models_dir: Path) -> None:
    """Fetch the MediaPipe hand and face models used for cropping."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in MODEL_URLS.items():
        target = models_dir / filename
        if not target.exists():
            log.info("downloading %s", filename)
            _fetch(url, target)


def download_roboflow(raw_dir: Path, version: int | None = None) -> int:
    """Export the Roboflow Auslan set as COCO and arrange it into one folder per letter.

    Needs ``ROBOFLOW_API_KEY`` (a free Roboflow account is enough). Uses the newest version
    unless one is given. Returns the image count.
    """
    api_key = os.environ.get(ROBOFLOW_KEY_VARIABLE)
    if not api_key:
        raise RuntimeError(f"set {ROBOFLOW_KEY_VARIABLE} first; see docs/DATASETS.md")
    version = version or _latest_version(api_key)
    log.info("Roboflow %s/%s version %d", ROBOFLOW_WORKSPACE, ROBOFLOW_PROJECT, version)
    link = _export_link(api_key, version)
    with tempfile.TemporaryDirectory() as scratch:
        archive = Path(scratch) / "export.zip"
        _fetch(link, archive)
        extracted = Path(scratch) / "coco"
        _safe_extract(archive, extracted)
        return organise_coco_by_class(extracted, Path(raw_dir) / "auslan_roboflow")


def _api_get(path: str, api_key: str) -> dict:
    url = f"{ROBOFLOW_API}/{path}?{urlencode({'api_key': api_key})}"
    try:
        with urllib.request.urlopen(url) as response:  # noqa: S310 - fixed https endpoint
            return json.load(response)
    except urllib.error.HTTPError as error:
        # Raised without the original error, whose request URL contains the API key.
        raise RuntimeError(
            f"Roboflow answered HTTP {error.code} for /{path}; check the workspace, project and "
            "version, and that the key is the private API key"
        ) from None


def _latest_version(api_key: str) -> int:
    payload = _api_get(f"{ROBOFLOW_WORKSPACE}/{ROBOFLOW_PROJECT}", api_key)
    versions = payload.get("versions") or payload.get("project", {}).get("versions") or []
    numbers = [int(str(v.get("id", v.get("version", "0"))).rsplit("/", 1)[-1]) for v in versions]
    if not numbers:
        raise RuntimeError("Roboflow lists no versions; pass --roboflow-version")
    return max(numbers)


def _export_link(api_key: str, version: int) -> str:
    payload = _api_get(f"{ROBOFLOW_WORKSPACE}/{ROBOFLOW_PROJECT}/{version}/coco", api_key)
    link = payload.get("export", {}).get("link") or payload.get("version", {}).get(
        "export", {}
    ).get("link")
    if not link:
        raise RuntimeError("Roboflow returned no download link; check the version number")
    return link


def _fetch(url: str, target: Path) -> None:
    """Download to a temporary name so an interrupted transfer never leaves a corrupt file."""
    partial = target.with_name(target.name + ".part")
    urllib.request.urlretrieve(url, partial)  # noqa: S310 - https URLs from this module or Roboflow
    partial.replace(target)


def _roboflow_export_link(api_key: str, version: int) -> str:
    url = f"{ROBOFLOW_API}/{ROBOFLOW_WORKSPACE}/{ROBOFLOW_PROJECT}/{version}/coco"
    with urllib.request.urlopen(f"{url}?{urlencode({'api_key': api_key})}") as response:  # noqa: S310
        payload = json.load(response)
    link = payload.get("export", {}).get("link") or payload.get("version", {}).get(
        "export", {}
    ).get("link")
    if not link:
        raise RuntimeError("Roboflow returned no download link; check the version number")
    return link


def _safe_extract(archive: Path, target: Path) -> None:
    root = target.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.namelist():
            if not (root / member).resolve().is_relative_to(root):
                raise RuntimeError(f"unsafe path in archive: {member}")
        bundle.extractall(target)


def organise_coco_by_class(coco_dir: Path, out_dir: Path) -> int:
    """Copy each image into ``out_dir/<LETTER>/``, taking the letter from its first annotation."""
    copied = 0
    for coco_file in sorted(Path(coco_dir).rglob("_annotations.coco.json")):
        data = json.loads(coco_file.read_text())
        names = {c["id"]: letter_from_name(c["name"]) for c in data["categories"]}
        letter_of_image: dict[int, str] = {}
        for box in data["annotations"]:
            letter = names.get(box["category_id"])
            if letter:
                letter_of_image.setdefault(box["image_id"], letter)
        for image in data["images"]:
            letter = letter_of_image.get(image["id"])
            source = coco_file.parent / image["file_name"]
            if letter and source.exists():
                target = Path(out_dir) / letter / f"{coco_file.parent.name}_{source.name}"
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                copied += 1
    return copied
