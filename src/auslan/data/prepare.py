"""Turn a raw dataset into a prepared one: de-duplicated, split, hand-cropped, with landmarks.

Output under ``<work_dir>/<name>/``: ``catalog.csv``, ``landmarks.npy`` and ``crops/*.jpg``.
Every stage is cached, so interrupted runs resume where they stopped. A finished dataset records
the settings it was built with and refuses to be silently reused under different ones.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from auslan.data.catalog import scan_folder_dataset
from auslan.data.dedup import (
    add_hashes,
    drop_exact_duplicates,
    drop_near_duplicates,
    drop_near_duplicates_of,
)
from auslan.data.registry import DatasetSpec
from auslan.data.splits import stratified_split
from auslan.preprocessing.hand_crop import LANDMARKS_PER_HAND, MAX_HANDS, HandCropper

log = logging.getLogger(__name__)

CropperFactory = Callable[[], HandCropper]
JPEG_QUALITY = 95
SUBCHUNK = 250  # rows handed to one worker task
LANDMARK_SHAPE = (MAX_HANDS, LANDMARKS_PER_HAND, 3)

_worker_cropper: HandCropper | None = None


def prepare_source(
    spec: DatasetSpec,
    work_dir: Path,
    cropper_factory: CropperFactory,
    seed: int,
    max_hamming: int,
    workers: int,
    cropper_key: str = "",
    reference: pd.DataFrame | None = None,
    chunk_size: int = 2000,
) -> Path:
    """Prepare one dataset and return its output directory (a no-op when already complete).

    ``cropper_key`` names the cropper settings (it is stored with the result). ``reference`` is a
    hashed catalog whose near-duplicates are removed from this dataset, so a test set stays free
    of images the model was trained on.
    """
    out = work_dir / spec.name
    settings = {"seed": seed, "max_hamming": max_hamming, "cropper": cropper_key}
    if (out / "catalog.csv").exists():
        _check_same_settings(out, settings)
        return out
    (out / "crops").mkdir(parents=True, exist_ok=True)
    frame = _deduplicated(spec, out, max_hamming, workers, reference)
    frame = stratified_split(frame, seed) if spec.split else frame.assign(split="test")
    crop_paths = [str(out / "crops" / f"{i:06d}.jpg") for i in range(len(frame))]
    chunk_key = f"{cropper_key}|{spec.fallback}"
    n_hands, used_fallback, landmarks = _crop_all(
        frame["path"].tolist(), crop_paths, out, cropper_factory, workers, chunk_size, chunk_key
    )
    np.save(out / "landmarks.npy", landmarks)
    (out / "settings.json").write_text(json.dumps(settings))
    frame.assign(
        crop_path=crop_paths,
        n_hands=n_hands,
        used_fallback=used_fallback,
        binarised=spec.fallback == "full",
        lm_index=np.arange(len(frame)),
    ).to_csv(out / "catalog.csv", index=False)
    return out


def load_prepared(out: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Read back the catalog and landmark array written by :func:`prepare_source`."""
    return pd.read_csv(out / "catalog.csv"), np.load(out / "landmarks.npy")


def _check_same_settings(out: Path, settings: dict) -> None:
    stored_file = out / "settings.json"
    if not stored_file.exists():
        stored_file.write_text(json.dumps(settings))  # prepared before settings were recorded
        return
    stored = json.loads(stored_file.read_text())
    if stored != settings:
        raise RuntimeError(
            f"{out.name} was prepared with {stored}, not {settings}; delete {out} to rebuild it"
        )


def _deduplicated(
    spec: DatasetSpec, out: Path, max_hamming: int, workers: int, reference: pd.DataFrame | None
) -> pd.DataFrame:
    hashed_path = out / "hashed.csv"
    if hashed_path.exists():
        hashed = pd.read_csv(hashed_path)
    else:
        scanned = scan_folder_dataset(spec.root, spec.name, spec.letters)
        log.info("%s: %d images found", spec.name, len(scanned))
        hashed = add_hashes(scanned, workers=workers)
        hashed.to_csv(hashed_path, index=False)
    exact = drop_exact_duplicates(hashed)
    near = drop_near_duplicates(exact, max_hamming)
    log.info("%s: %d -> %d exact -> %d near-unique", spec.name, len(hashed), len(exact), len(near))
    if reference is not None:
        near = drop_near_duplicates_of(near, reference, max_hamming)
        log.info("%s: %d after removing overlap with the reference set", spec.name, len(near))
    return near


def _crop_all(
    sources: list[str],
    targets: list[str],
    out: Path,
    factory: CropperFactory,
    workers: int,
    chunk_size: int,
    chunk_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    chunk_dir = out / "chunks"
    chunk_dir.mkdir(exist_ok=True)
    executor = None
    if workers > 1:
        executor = ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(factory,))
    local_cropper = factory() if executor is None else None
    parts = []
    try:
        for number, start in enumerate(range(0, len(sources), chunk_size)):
            src, dst = sources[start : start + chunk_size], targets[start : start + chunk_size]
            signature = hashlib.md5(("|".join(src) + chunk_key).encode()).hexdigest()  # noqa: S324
            cached = chunk_dir / f"{number:05d}.npz"
            part = _read_chunk(cached, signature, dst)
            if part is None:
                part = _crop_chunk(src, dst, executor, local_cropper)
                np.savez(
                    cached,
                    signature=signature,
                    n_hands=part[0],
                    used_fallback=part[1],
                    landmarks=part[2],
                )
                log.info("cropped %d / %d", min(start + chunk_size, len(sources)), len(sources))
            parts.append(part)
    finally:
        if executor is not None:
            executor.shutdown()
    if not parts:
        return np.zeros(0, int), np.zeros(0, bool), np.zeros((0, *LANDMARK_SHAPE), np.float32)
    return tuple(np.concatenate(group) for group in zip(*parts, strict=True))


def _read_chunk(path: Path, signature: str, crop_paths: list[str]):
    """A finished chunk from an earlier run, if it was made with the same inputs and settings."""
    if not path.exists():
        return None
    with np.load(path) as data:
        if str(data["signature"]) != signature or not all(Path(p).exists() for p in crop_paths):
            return None
        return data["n_hands"], data["used_fallback"], data["landmarks"]


def _init_worker(factory: CropperFactory) -> None:
    global _worker_cropper
    _worker_cropper = factory()


def _crop_chunk(src: list[str], dst: list[str], executor, local_cropper: HandCropper | None):
    if local_cropper is not None:  # single process: no pool start-up cost
        return _crop_rows(local_cropper, src, dst)
    pieces = [(src[i : i + SUBCHUNK], dst[i : i + SUBCHUNK]) for i in range(0, len(src), SUBCHUNK)]
    results = list(executor.map(_crop_rows_in_worker, *zip(*pieces, strict=True)))
    return tuple(np.concatenate(group) for group in zip(*results, strict=True))


def _crop_rows_in_worker(src: list[str], dst: list[str]):
    if _worker_cropper is None:
        raise RuntimeError("worker cropper was not initialised")
    return _crop_rows(_worker_cropper, src, dst)


def _crop_rows(cropper: HandCropper, src: list[str], dst: list[str]):
    n_hands, fallbacks, landmarks = [], [], []
    for source_path, target_path in zip(src, dst, strict=True):
        bgr = cv2.imread(source_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise OSError(f"cannot read image: {source_path}")
        result = cropper(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        crop_bgr = cv2.cvtColor(result.crop, cv2.COLOR_RGB2BGR)
        cv2.imwrite(target_path, crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        n_hands.append(result.n_hands)
        fallbacks.append(result.used_fallback)
        landmarks.append(result.landmarks)
    return np.array(n_hands), np.array(fallbacks, dtype=bool), np.stack(landmarks)
