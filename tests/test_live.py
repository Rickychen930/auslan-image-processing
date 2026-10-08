import numpy as np
import pytest

from auslan.config import Settings
from auslan.live.camera import COLOURS, draw_overlay
from auslan.live.export import export_model
from auslan.live.recognizer import (
    LiveModel,
    Recognizer,
    Verdict,
    judge,
    load_live_model,
    top_letters,
)
from auslan.preprocessing.hand_crop import FullFrame, HandCropper, HandDetection

LETTERS = ("A", "B", "C")


class Fixed:
    """A classifier that always answers with the same probabilities."""

    uses_landmarks = False

    def __init__(self, proba, uses_landmarks=False):
        self._proba = np.array(proba, np.float32)
        self.uses_landmarks = uses_landmarks

    def predict_proba(self, samples):
        return np.tile(self._proba, (len(samples), 1))


class Hands:
    def __init__(self, n_hands):
        self._n = n_hands

    def detect(self, rgb):
        hand = np.linspace(0.3, 0.6, 63, dtype=np.float32).reshape(21, 3)
        return HandDetection(hands=tuple(hand for _ in range(self._n)))


def _model(proba, mode, uses_landmarks=False):
    return LiveModel(Fixed(proba, uses_landmarks), LETTERS, mode, 32, "test model")


def test_top_letters_are_ordered_by_probability():
    assert top_letters(LETTERS, np.array([0.1, 0.6, 0.3]), 2) == [("B", 0.6), ("C", 0.3)]


@pytest.mark.parametrize(
    ("proba", "target", "status"),
    [
        (None, "A", "no-hand"),
        (np.array([0.7, 0.2, 0.1]), None, "idle"),
        (np.array([0.7, 0.2, 0.1]), "A", "correct"),
        (np.array([0.7, 0.2, 0.1]), "B", "wrong"),
        (np.array([0.4, 0.35, 0.25]), "A", "wrong"),  # right letter, but not confident enough
    ],
)
def test_judge_reports_what_is_seen_and_whether_it_matches(proba, target, status):
    assert judge(LETTERS, proba, target).status == status


def test_whole_frame_mode_feeds_the_model_a_drawn_image():
    recognizer = Recognizer(_model([0.1, 0.8, 0.1], "frame"), cropper=None)
    proba = recognizer.proba(np.full((60, 90, 3), 180, np.uint8))
    assert proba.argmax() == 1
    assert set(np.unique(recognizer.last_input)) <= {0, 255}  # drawn like the Kaggle images


def test_hand_mode_needs_a_cropper():
    with pytest.raises(ValueError, match="cropper"):
        Recognizer(_model([1, 0, 0], "hands"), cropper=None)


def test_landmark_models_report_no_hand_instead_of_guessing():
    cropper = HandCropper(Hands(0), FullFrame(), size=32)
    recognizer = Recognizer(_model([1, 0, 0], "hands", uses_landmarks=True), cropper)
    assert recognizer.proba(np.zeros((40, 40, 3), np.uint8)) is None


def test_predictions_are_averaged_over_recent_frames():
    model = _model([1.0, 0.0, 0.0], "frame")
    recognizer = Recognizer(model, cropper=None, smoothing=2)
    frame = np.zeros((20, 20, 3), np.uint8)
    recognizer.proba(frame)
    model.classifier = Fixed([0.0, 1.0, 0.0])
    assert recognizer.proba(frame) == pytest.approx([0.5, 0.5, 0.0])
    assert recognizer.proba(frame) == pytest.approx([0.0, 1.0, 0.0])  # the first frame dropped out


def test_overlay_keeps_the_frame_size_and_colours_the_verdict():
    preview = np.zeros((240, 320, 3), np.uint8)
    out = draw_overlay(
        preview, [("A", 0.9), ("B", 0.1)], Verdict("correct", "Correct: A"), 12.0, None
    )
    assert out.shape == preview.shape and not preview.any()  # the input is left untouched
    assert (out == np.array(COLOURS["correct"], np.uint8)).all(axis=2).any()


def test_export_trains_saves_and_loads_a_working_model(prepared_work_dir, tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data", artifacts_dir=tmp_path / "artifacts", image_size=32
    )
    path = export_model(settings, "asl-landmarks")
    model = load_live_model(path)
    assert model.input_mode == "hands" and len(model.letters) == 24
    cropper = HandCropper(Hands(1), FullFrame(), size=32)
    proba = Recognizer(model, cropper).proba(np.zeros((40, 40, 3), np.uint8))
    assert proba.shape == (24,) and proba.sum() == pytest.approx(1.0, abs=1e-4)


def test_missing_model_names_the_fix(tmp_path):
    with pytest.raises(FileNotFoundError, match="auslan export"):
        load_live_model(tmp_path / "none.joblib")
