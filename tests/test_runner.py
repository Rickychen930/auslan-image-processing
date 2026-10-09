import numpy as np
import pandas as pd
import pytest

from auslan.config import Settings
from auslan.methods import create
from auslan.results import ResultStore, RunSpec
from auslan.runner import Runner

FAST = {"max_steps": 10, "device": "cpu", "batch_size": 16}


@pytest.fixture
def setup(prepared_work_dir, tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        image_size=32,
        val_per_class=2,
        seeds=(0,),
    )
    store = ResultStore(settings.artifacts_dir)
    return Runner(settings, store), store, settings


def test_run_spec_keys_are_stable_and_distinguish_variants():
    assert RunSpec("m3_cnn", "P1", seed=2).key == "m3_cnn-P1-s2"
    assert RunSpec("m3_cnn", "P2", 20, 1).key == "m3_cnn-P2-k20-s1"
    assert RunSpec("m3_cnn", "P3", None, 0).key == "m3_cnn-P3-kall-s0"
    assert RunSpec("m2_hog_svm", "P1", options={"cell_size": 6}, tag="cell6").key.endswith("-cell6")


def test_run_spec_rejects_untagged_variants_and_unknown_protocols():
    with pytest.raises(ValueError, match="tag"):
        RunSpec("m2_hog_svm", "P1", options={"cell_size": 6})
    with pytest.raises(ValueError, match="protocol"):
        RunSpec("m3_cnn", "P4")


def test_p1_scores_the_held_out_split_and_the_cross_dataset_set(setup):
    runner, store, _ = setup
    assert runner.run(RunSpec("m1_skin_svm", "P1"))
    table = store.table()
    assert set(table["test_set"]) == {"asl_test", "asl_cross"}
    assert (table["n_train"] == 24).all()  # 8 per letter x 3 letters
    assert (table["macro_f1"][table["test_set"] == "asl_test"] > 0.8).all()


def test_finished_runs_are_not_repeated(setup):
    runner, store, _ = setup
    spec = RunSpec("m1_skin_svm", "P1")
    runner.run(spec)
    assert runner.run(spec) is False
    assert len(store.table()) == 2


def test_p2_draws_k_per_letter_and_scores_three_test_sets(setup):
    runner, store, _ = setup
    runner.run(RunSpec("m5_landmarks_svm", "P2", 5, 0))
    table = store.table()
    assert set(table["test_set"]) == {"kaggle_test", "roboflow", "roboflow_raw"}
    assert (table["n_train"] == 15).all()
    assert (table["macro_f1"] > 0.8).all()  # the toy letters are easy to tell apart
    y_true, proba = store.predictions("m5_landmarks_svm-P2-k5-s0", "roboflow")
    assert proba.shape == (12, 24) and len(y_true) == 12  # one column per Auslan letter


def test_landmark_methods_are_skipped_when_no_hands_are_found_in_training(setup):
    runner, store, settings = setup
    catalog = settings.work_dir / "auslan_kaggle" / "catalog.csv"
    pd.read_csv(catalog).assign(n_hands=0).to_csv(catalog, index=False)
    assert runner.run(RunSpec("m5_landmarks_svm", "P2", 5, 0)) is False
    assert runner.run(RunSpec("m5_landmarks_svm", "P1")) is True  # ASL detection is unaffected
    assert runner.run(RunSpec("m1_skin_svm", "P2", 5, 0)) is True  # pixel methods still run


def test_p3_is_skipped_for_methods_that_cannot_transfer(setup):
    runner, store, _ = setup
    assert runner.run(RunSpec("m1_skin_svm", "P3", 5, 0)) is False
    assert store.table().empty


def test_p3_pretrains_on_asl_once_per_method_and_seed(setup):
    runner, store, settings = setup
    assert runner.run(RunSpec("m3_cnn", "P3", 5, 0, options=FAST, tag="fast"))
    saved = list((settings.artifacts_dir / "pretrained").glob("*.joblib"))
    assert len(saved) == 1
    stamp = saved[0].stat().st_mtime_ns
    # same method, options and seed, different k: the ASL weights must be reused, not retrained
    assert runner.run(RunSpec("m3_cnn", "P3", 10, 0, options=FAST, tag="fast"))
    assert len(list(saved[0].parent.glob("*.joblib"))) == 1
    assert saved[0].stat().st_mtime_ns == stamp
    assert np.isfinite(store.table()["macro_f1"]).all()


def test_only_m1_is_tested_on_the_unstyled_colour_photos(setup):
    runner, _, _ = setup
    bundle = runner._bundle(32)
    for method, drawings in (("m1_skin_svm", False), ("m3_cnn", True)):
        spec = RunSpec(method, "P2", 5, 0)
        tests = runner._plan(spec, bundle, create(method))[2]
        assert tests["roboflow"][0].binarised is drawings
        assert not tests["roboflow_raw"][0].binarised


def test_missing_dataset_error_names_the_fix(tmp_path):
    settings = Settings(data_dir=tmp_path / "empty", artifacts_dir=tmp_path / "art")
    runner = Runner(settings, ResultStore(settings.artifacts_dir))
    with pytest.raises(FileNotFoundError, match="auslan prepare"):
        runner.run(RunSpec("m1_skin_svm", "P1"))
