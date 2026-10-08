import pandas as pd
import pytest

from auslan.data.splits import sample_k_shot, stratified_split, validation_subset


def _frame(per_class: int = 100, letters: str = "ABC") -> pd.DataFrame:
    rows = [(f"{c}{i}.jpg", c) for c in letters for i in range(per_class)]
    return pd.DataFrame(rows, columns=["path", "label"])


def test_stratified_split_ratios_and_balance():
    out = stratified_split(_frame(), seed=0)
    counts = out["split"].value_counts()
    assert counts["train"] == 210 and counts["val"] == 45 and counts["test"] == 45
    for _, group in out.groupby("split"):
        assert group["label"].value_counts().nunique() == 1  # perfectly balanced


def test_stratified_split_is_deterministic_and_disjoint():
    a = stratified_split(_frame(), seed=3)
    b = stratified_split(_frame(), seed=3)
    assert a["split"].tolist() == b["split"].tolist()
    assert a["path"].is_unique


def test_stratified_split_rejects_bad_ratios():
    with pytest.raises(ValueError):
        stratified_split(_frame(), ratios=(0.5, 0.5, 0.5), seed=0)


def test_k_shot_draws_k_per_class_and_varies_with_seed():
    train = _frame()
    one = sample_k_shot(train, k=5, seed=0)
    two = sample_k_shot(train, k=5, seed=1)
    assert one["label"].value_counts().tolist() == [5, 5, 5]
    assert set(one["path"]) != set(two["path"])
    assert set(one["path"]) == set(sample_k_shot(train, k=5, seed=0)["path"])


def test_k_shot_none_returns_everything_and_large_k_is_capped():
    train = _frame(per_class=10)
    assert len(sample_k_shot(train, k=None, seed=0)) == 30
    assert len(sample_k_shot(train, k=50, seed=0)) == 30


def test_validation_subset_is_fixed_size_per_class():
    out = validation_subset(_frame(), per_class=10, seed=0)
    assert out["label"].value_counts().tolist() == [10, 10, 10]
