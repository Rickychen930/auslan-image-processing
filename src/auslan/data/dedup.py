"""Exact (MD5) and near-duplicate (perceptual hash) removal.

Consecutive video frames and nested dataset copies would otherwise leak between train and test
splits and inflate scores (Kapoor & Narayanan, 2023).
"""

from __future__ import annotations

import hashlib
from io import BytesIO

import imagehash
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image

DEFAULT_MAX_HAMMING = 4  # of 64 phash bits


def _hash_file(path: str) -> tuple[str, str]:
    with open(path, "rb") as handle:
        data = handle.read()
    with Image.open(BytesIO(data)) as image:
        phash = str(imagehash.phash(image.convert("L")))
    return hashlib.md5(data).hexdigest(), phash  # noqa: S324 - content fingerprint, not security


def add_hashes(df: pd.DataFrame, workers: int = -1) -> pd.DataFrame:
    """Return ``df`` with ``md5`` and ``phash`` (16 hex chars) columns."""
    hashes = Parallel(n_jobs=workers)(delayed(_hash_file)(p) for p in df["path"])
    md5, phash = zip(*hashes, strict=True) if hashes else ((), ())
    return df.assign(md5=list(md5), phash=list(phash))


def drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the first file for every (label, content) pair."""
    return df.drop_duplicates(subset=["label", "md5"], keep="first").reset_index(drop=True)


def _as_uint64(hex_hashes: pd.Series) -> np.ndarray:
    return np.array([int(h, 16) for h in hex_hashes], dtype=np.uint64)


def _hamming(hashes: np.ndarray, value: np.uint64) -> np.ndarray:
    return np.bitwise_count(hashes ^ value)


def drop_near_duplicates(df: pd.DataFrame, max_hamming: int = DEFAULT_MAX_HAMMING) -> pd.DataFrame:
    """Greedily keep an image only if it is not within ``max_hamming`` bits of a kept one.

    Comparison is restricted to the same letter, since near-identical frames come from one video.
    """
    keep = np.zeros(len(df), dtype=bool)
    hashes = _as_uint64(df["phash"])
    for positions in df.groupby("label").indices.values():
        kept = np.empty(len(positions), dtype=np.uint64)
        count = 0
        for pos in positions:  # df order is preserved inside a group
            if count and (_hamming(kept[:count], hashes[pos]) <= max_hamming).any():
                continue
            kept[count] = hashes[pos]
            count += 1
            keep[pos] = True
    return df[keep].reset_index(drop=True)


def drop_near_duplicates_of(
    df: pd.DataFrame, reference: pd.DataFrame, max_hamming: int = DEFAULT_MAX_HAMMING
) -> pd.DataFrame:
    """Remove rows of ``df`` that nearly match any ``reference`` image (any label)."""
    ref_hashes = _as_uint64(reference["phash"])
    hashes = _as_uint64(df["phash"])
    clash = np.array([(_hamming(ref_hashes, h) <= max_hamming).any() for h in hashes], dtype=bool)
    return df[~clash].reset_index(drop=True)
