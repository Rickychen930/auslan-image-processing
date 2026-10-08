"""Turn a raw dataset into a prepared one: de-duplicated, split, hand-cropped, with landmarks.

Output under ``<work_dir>/<name>/``: ``catalog.csv``, ``landmarks.npy`` and ``crops/*.jpg``.
Every stage is cached, so interrupted runs resume where they stopped.
"""

from __future__ import annotations

import hashlib
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
from auslan.preprocessing.hand_crop import HandCropper

log = logging.getLogger(__name__)

CropperFactory = Callable[[], HandCropper]
JPEG_QUALITY = 95
SUBCHUNK = 250  # rows handed to one worker task

_worker_cropper: HandCropper | None = None


def prepare_source(
    spec: DatasetSpec,
    work_dir: Path,
    cropper_factory: CropperFactory,
    seed: int,
    max_hamming: int,
    workers: int,
    reference: pd.DataFrame | None = None,
    chunk_size: int = 2000,
) -> Path:
    """Prepare one dataset and return its output directory (a no-op when already complete).

    ``reference`` is a hashed catalog (with ``phash``) whose near-duplicates are removed from
    this dataset, used to keep the independent Auslan test set free of Kaggle images.
    """
    out = Path(work_dir) / spec.name
    if (out / "catalog.csv").exists():
        return out
    (out / "crops").mkdir(parents=True, exist_ok=True)
    frame = _deduplicated(spec, out, max_hamming, workers, reference)
    frame = stratified_split(frame, seed) if spec.split else frame.assign(split="test")
    crop_paths = [str(out / "crops" / f"{i:06d}.jpg") for i in range(len(frame))]
    n_hands, used_fallback, landmarks = _crop_all(
        frame["path"].tolist(), crop_paths, out, cropper_factory, workers, chunk_size
    )
    np.save(out / "landmarks.npy", landmarks)
    frame.assign(
        crop_path=crop_paths,
        n_hands=n_hands,
        used_fallback=used_fallback,
        lm_index=np.arange(len(frame)),
    ).to_csv(out / "catalog.csv", index=False)
    return out


def load_prepared(out: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Read back the catalog and landmark array written by :func:`prepare_source`."""
    return pd.read_csv(Path(out) / "catalog.csv"), np.load(Path(out) / "landmarks.npy")


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
        log.info("%s: %d after removing overlap with reference", spec.name, len(near))
    return near


def _crop_all(
    sources: list[str],
    targets: list[str],
    out: Path,
    factory: CropperFactory,
    workers: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    chunk_dir = out / "chunks"
    chunk_dir.mkdir(exist_ok=True)
    parts = []
    executor = _make_executor(factory, workers)
    try:
        for number, start in enumerate(range(0, len(sources), chunk_size)):
            src, dst = sources[start : start + chunk_size], targets[start : start + chunk_size]
            signature = hashlib.md5("|".join(src).encode()).hexdigest()  # noqa: S324
            cached = chunk_dir / f"{number:05d}.npz"
            if cached.exists() and str(np.load(cached)["signature"]) == signature:
                data = np.load(cached)
                parts.append((data["n_hands"], data["used_fallback"], data["landmarks"]))
                continue
            part = _crop_chunk(src, dst, executor, factory)
            np.savez(
                cached,
                signature=signature,
                n_hands=part[0],
                used_fallback=part[1],
                landmarks=part[2],
            )
            parts.append(part)
            log.info("cropped %d / %d", min(start + chunk_size, len(sources)), len(sources))
    finally:
        if executor is not None:
            executor.shutdown()
    if not parts:
        return np.zeros(0, int), np.zeros(0, bool), np.zeros((0, 2, 21, 3), np.float32)
    return tuple(np.concatenate(group) for group in zip(*parts, strict=True))


def _make_executor(factory: CropperFactory, workers: int) -> ProcessPoolExecutor | None:
    if workers <= 1:
        return None
    return ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(factory,))


def _init_worker(factory: CropperFactory) -> None:
    global _worker_cropper
    _worker_cropper = factory()


def _crop_chunk(src: list[str], dst: list[str], executor, factory: CropperFactory):
    if executor is None:
        return _crop_rows(factory(), src, dst)
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
