import pytest

from auslan.config import Settings
from auslan.experiments import CORE_METHODS, IMAGE_SIZES, plan


def test_e1_runs_every_method_under_p1_and_both_few_shot_protocols():
    specs = plan("E1", Settings())
    # per method: 3 P1 seeds + (3 seeds at k=20 + the all-data point once) x (P2, P3)
    assert len(specs) == len(CORE_METHODS) * (3 + 2 * (3 + 1))
    assert {s.protocol for s in specs} == {"P1", "P2", "P3"}
    assert {s.k for s in specs if s.protocol != "P1"} == {20, None}


def test_all_data_point_is_drawn_once():
    all_data = [s for s in plan("E1", Settings()) if s.protocol == "P2" and s.k is None]
    assert {s.seed for s in all_data} == {0}


def test_e3_follows_the_configured_k_values():
    settings = Settings(k_values=(5, 10))
    ks = {s.k for s in plan("E3", settings, methods=["m3_cnn"])}
    assert ks == {5, 10, None}


def test_e5_variants_are_tagged_so_results_do_not_collide():
    specs = plan("E5", Settings())
    assert specs and all(s.tag for s in specs)
    assert len({s.key for s in specs}) == len(specs)


def test_e7_varies_image_size_and_leaves_landmark_methods_out():
    specs = plan("E7", Settings())
    assert {s.image_size for s in specs} == set(IMAGE_SIZES)
    assert not any(s.method.startswith("m5_") for s in specs)


def test_everything_plan_has_unique_keys_and_starts_with_e1():
    specs = plan("all", Settings())
    assert len({s.key for s in specs}) == len(specs)
    assert specs[0] == plan("E1", Settings())[0]


def test_method_and_seed_filters_apply():
    specs = plan("E1", Settings(), methods=["m3_cnn"], seeds=[5])
    assert {s.method for s in specs} == {"m3_cnn"} and {s.seed for s in specs} <= {5}


def test_protocol_filter_keeps_only_requested_protocols():
    specs = plan("E1", Settings(), protocols=["P1"])
    assert specs and {s.protocol for s in specs} == {"P1"}


def test_unknown_experiment_lists_the_choices():
    with pytest.raises(KeyError, match="E1"):
        plan("E9", Settings())
