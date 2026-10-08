import numpy as np
import pytest
import torch
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier

from auslan.methods import create, method_names
from auslan.methods.base import FeatureClassifier
from auslan.methods.m1_skin import empty_mask_rate, hand_mask, shape_features
from auslan.methods.m5_landmarks import CONTACT_FEATURES, feature_count, landmark_features
from auslan.methods.torch_base import augment_batch

FAST_TORCH = {"max_steps": 80, "device": "cpu", "batch_size": 16, "lr": 2e-3}
# method name -> (constructor options, minimum toy accuracy)
CASES = {
    "m1_skin_svm": ({}, 0.9),
    "m1_skin_knn": ({}, 0.9),
    "m2_hog_svm": ({}, 0.9),
    "m2_hog_rf": ({}, 0.9),
    "m3_cnn": (FAST_TORCH, 0.6),
    # depthwise convolutions are very slow on CPU, so this one only checks the plumbing;
    # learning is covered by ResNet-18 and M3, which share the same training loop
    "m4_mobilenetv2": ({**FAST_TORCH, "max_steps": 10, "pretrained": False}, 0.0),
    "m4_resnet18": ({**FAST_TORCH, "pretrained": False}, 0.6),
    "m5_landmarks_svm": ({}, 0.9),
    "m5_landmarks_rf": ({}, 0.8),
    "m5_landmarks_mlp": ({"max_steps": 150, "device": "cpu", "batch_size": 16}, 0.9),
}


def test_registry_lists_all_ten_methods():
    assert method_names() == sorted(CASES)


def test_unknown_method_names_the_choices():
    with pytest.raises(KeyError, match="m3_cnn"):
        create("nope")


@pytest.mark.parametrize("name", sorted(CASES))
def test_method_learns_a_separable_toy_task(name, make_sample_set):
    options, minimum = CASES[name]
    train, val, test = (make_sample_set(3, n, seed) for n, seed in ((12, 0), (4, 1), (8, 2)))
    model = create(name, **options).fit(train, val=val)
    proba = model.predict_proba(test)
    assert proba.shape == (len(test), 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)
    assert np.mean(proba.argmax(axis=1) == test.labels) >= minimum


def test_predicting_before_fitting_fails_clearly():
    with pytest.raises(RuntimeError, match="fitted"):
        create("m3_cnn", device="cpu").predict_proba(None)


@pytest.mark.parametrize("name", ["m3_cnn", "m4_resnet18", "m5_landmarks_mlp"])
def test_transfer_to_a_new_label_space(name, make_sample_set):
    options = {**CASES[name][0], "max_steps": 20}
    source = create(name, **options).fit(make_sample_set(2, 10, 0))
    assert source.supports_transfer
    target = create(name, **options).fit(make_sample_set(3, 10, 1), init=source)
    assert target.predict_proba(make_sample_set(3, 4, 2)).shape == (12, 3)


@pytest.mark.parametrize("name", ["m1_skin_svm", "m2_hog_svm", "m5_landmarks_svm"])
def test_hand_crafted_methods_do_not_transfer(name):
    assert not create(name).supports_transfer


def test_transfer_rejects_a_different_method(make_sample_set):
    cnn = create("m3_cnn", **CASES["m3_cnn"][0]).fit(make_sample_set(2, 6, 0))
    with pytest.raises(TypeError):
        create("m4_resnet18", **CASES["m4_resnet18"][0]).fit(make_sample_set(2, 6, 1), init=cnn)


def test_shape_features_separate_shapes_and_handle_blank_images(make_sample_set):
    ds = make_sample_set(3, 1, 0)
    square, bar, disc = (shape_features(ds[i].crop, binarised=False) for i in range(3))
    assert not np.allclose(square, bar) and not np.allclose(square, disc)
    blank_page = np.full((64, 64, 3), 255, np.uint8)
    assert not shape_features(blank_page, binarised=True).any()


def test_drawings_use_the_edge_mask_and_photos_the_skin_mask():
    page = np.full((64, 64, 3), 255, np.uint8)
    page[20:44, 20] = page[20:44, 43] = page[20, 20:44] = page[43, 20:44] = 0  # hollow square
    assert hand_mask(page, binarised=True)[32, 32] == 255  # the outline is filled in
    assert not hand_mask(page, binarised=False).any()  # a white page has no skin colour


def test_empty_mask_rate_counts_images_without_a_hand(make_sample_set):
    assert empty_mask_rate(make_sample_set(3, 2, 0)) == 0.0


def test_landmark_features_ignore_position_and_scale():
    rng = np.random.default_rng(0)
    one = np.zeros((1, 2, 21, 3), np.float32)
    one[0, 0] = rng.random((21, 3))
    moved = one.copy()
    moved[0, 0, :, :2] = one[0, 0, :, :2] * 2.5 + 0.3
    moved[0, 0, :, 2] = one[0, 0, :, 2] * 2.5
    assert np.allclose(landmark_features(one), landmark_features(moved), atol=1e-4)


def test_two_hand_distances_are_zero_unless_both_hands_are_present():
    rng = np.random.default_rng(1)
    pair = rng.random((1, 2, 21, 3)).astype(np.float32)
    single = pair.copy()
    single[0, 1] = 0
    with_distances = landmark_features(single, two_hand_distances=True)
    without = landmark_features(single, two_hand_distances=False)
    assert with_distances.shape[1] == without.shape[1] + CONTACT_FEATURES
    assert with_distances.shape[1] == feature_count() and without.shape[1] == feature_count(False)
    assert not with_distances[0, without.shape[1] :].any()
    assert landmark_features(pair)[0, without.shape[1] :].any()


def test_only_m1_asks_for_the_colour_photo():
    needs_colour = {name: create(name).needs_colour for name in method_names()}
    assert {name for name, flag in needs_colour.items() if flag} == {"m1_skin_svm", "m1_skin_knn"}


def test_hyper_parameters_are_chosen_on_the_validation_set(make_sample_set):
    class TwoCandidates(FeatureClassifier):
        def _features(self, samples):
            return landmark_features(samples.landmark_array())

        def _candidates(self, features):  # the first is useless, the second is good
            return [
                DummyClassifier(strategy="constant", constant=0),
                RandomForestClassifier(50, random_state=0),
            ]

    train, val, test = (make_sample_set(3, n, seed) for n, seed in ((10, 0), (4, 1), (6, 2)))
    chosen = TwoCandidates("two").fit(train, val=val).predict_proba(test).argmax(axis=1)
    untuned = TwoCandidates("two").fit(train).predict_proba(test).argmax(axis=1)
    assert np.mean(chosen == test.labels) > 0.8 > np.mean(untuned == test.labels)


def test_training_on_an_empty_set_fails_instead_of_looping(make_sample_set):
    empty = make_sample_set(2, 2, 0).subset([])
    with pytest.raises(ValueError, match="empty"):
        create("m3_cnn", device="cpu").fit(empty)


def test_transfer_copies_the_backbone_and_replaces_the_head(make_sample_set):
    source = create(
        "m3_cnn",
        **CASES["m3_cnn"][0],
    ).fit(make_sample_set(2, 6, 0))
    fresh = source._build(3)
    source._transfer(fresh, source)
    copied = torch.equal(fresh.features[0][0].weight, source._model.features[0][0].weight.cpu())
    assert copied and fresh.head[-1].out_features == 3


def test_augmentation_keeps_page_colour_and_never_mirrors():
    torch.manual_seed(0)
    white_page = torch.ones(8, 3, 32, 32)
    assert torch.allclose(augment_batch(white_page, fill=1.0), white_page, atol=0.35)
    assert augment_batch(white_page, fill=1.0).min() > 0.6  # no black corners from rotation

    left_bright = torch.zeros(16, 3, 32, 32)
    left_bright[..., :16] = 1.0
    out = augment_batch(left_bright, fill=0.0)
    assert out.shape == left_bright.shape and out.min() >= 0 and out.max() <= 1
    assert (out[..., :16].mean(dim=(1, 2, 3)) > out[..., 16:].mean(dim=(1, 2, 3))).all()
