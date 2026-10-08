"""Stratified train/val/test splits and k-shot sampling."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_RATIOS = (0.70, 0.15, 0.15)


def stratified_split(
    df: pd.DataFrame, seed: int, ratios: tuple[float, float, float] = DEFAULT_RATIOS
) -> pd.DataFrame:
    """Return ``df`` with a ``split`` column (train/val/test), stratified by ``label``."""
    train_ratio, val_ratio, test_ratio = ratios
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(f"ratios must sum to 1, got {ratios}")
    labels = df["label"].to_numpy()
    positions = np.arange(len(df))
    train_pos, rest_pos = train_test_split(
        positions, train_size=train_ratio, stratify=labels, random_state=seed
    )
    val_pos, test_pos = train_test_split(
        rest_pos,
        train_size=val_ratio / (val_ratio + test_ratio),
        stratify=labels[rest_pos],
        random_state=seed,
    )
    split = np.empty(len(df), dtype=object)
    split[train_pos], split[val_pos], split[test_pos] = "train", "val", "test"
    return df.assign(split=split)


def sample_k_shot(df: pd.DataFrame, k: int | None, seed: int) -> pd.DataFrame:
    """Draw ``k`` images per letter (all of them when ``k`` is ``None`` or exceeds the supply)."""
    rng = np.random.default_rng(seed)
    parts = []
    for _, group in df.groupby("label", sort=True):
        take = len(group) if k is None else min(k, len(group))
        parts.append(group.iloc[rng.permutation(len(group))[:take]])
    return pd.concat(parts).reset_index(drop=True)


def validation_subset(df: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    """Fixed-size validation set so tuning data stays comparable to the k training images."""
    return sample_k_shot(df, per_class, seed)
