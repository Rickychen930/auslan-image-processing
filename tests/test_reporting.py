import pytest

from auslan.config import Settings
from auslan.report import build_report
from auslan.results import ResultStore, RunSpec
from auslan.runner import Runner

METHODS = ("m1_skin_svm", "m5_landmarks_svm", "m5_landmarks_mlp")


@pytest.fixture
def finished_runs(prepared_work_dir, tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        image_size=32,
        val_per_class=2,
    )
    store = ResultStore(settings.artifacts_dir)
    runner = Runner(settings, store)
    for seed in (0, 1):
        for method in METHODS:
            runner.run(RunSpec(method, "P1", seed=seed))
            runner.run(RunSpec(method, "P2", 20, seed))
        runner.run(RunSpec("m5_landmarks_mlp", "P3", 20, seed))
    return store, settings


def test_report_writes_every_table(finished_runs):
    store, settings = finished_runs
    out = build_report(store, settings)
    for name in (
        "results.csv",
        "table9.md",
        "k_curve.csv",
        "per_letter_f1.csv",
        "confused_pairs.md",
        "significance.md",
        "success_criteria.md",
        "cost.md",
        "cross_dataset.md",
        "variants.md",
        "style_effect.md",
        "image_size.md",
        "detection_rates.md",
        "confusion_m5_landmarks_mlp.csv",
    ):
        assert (out / name).stat().st_size > 0, name


def test_headline_table_marks_methods_that_cannot_transfer(finished_runs):
    store, settings = finished_runs
    table9 = (build_report(store, settings) / "table9.md").read_text()
    rows = [line for line in table9.splitlines() if line.startswith("|")][2:]  # skip header
    lines = {line.split("|")[1].strip(): line for line in rows}
    assert set(lines) == set(METHODS)
    assert "(=P2)" in lines["m1_skin_svm"] and "n/a" in lines["m1_skin_svm"]
    assert "(=P2)" not in lines["m5_landmarks_mlp"] and "n/a" not in lines["m5_landmarks_mlp"]


def test_significance_covers_all_pairs_and_transfer_gain(finished_runs):
    store, settings = finished_runs
    text = (build_report(store, settings) / "significance.md").read_text()
    assert text.count("| m") >= 3 * 2  # three method pairs, each names two methods
    assert "m5_landmarks_mlp" in text and "bootstrap" in text


def test_report_without_results_fails_clearly(tmp_path):
    with pytest.raises(RuntimeError, match="no results"):
        build_report(ResultStore(tmp_path), Settings(artifacts_dir=tmp_path))
