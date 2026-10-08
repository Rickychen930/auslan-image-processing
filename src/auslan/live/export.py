"""Train a model for the live demo and save it under ``artifacts/live``."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from auslan.config import Settings
from auslan.data.bundle import DataBundle
from auslan.data.classes import ASL_SPACE, AUSLAN_SPACE
from auslan.live.recognizer import LiveModel, save_live_model
from auslan.methods import create

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Preset:
    method: str
    dataset: str  # "asl" or "auslan"
    input_mode: str  # how a camera frame is turned into model input, see LiveModel
    description: str


PRESETS = {
    "auslan-resnet": Preset(
        "m4_resnet18",
        "auslan",
        "frame",
        "ResNet-18 trained on the Kaggle Auslan frames; the whole frame is drawn like them",
    ),
    "asl-landmarks": Preset(
        "m5_landmarks_mlp",
        "asl",
        "hands",
        "MLP on MediaPipe hand landmarks, trained on the 24 one-handed ASL letters",
    ),
    "asl-resnet": Preset(
        "m4_resnet18", "asl", "hands", "ResNet-18 on hand crops, trained on the 24 ASL letters"
    ),
}


def export_model(settings: Settings, name: str, seed: int = 0) -> Path:
    """Train on the dataset's full training split and save the model for ``auslan live``."""
    preset = PRESETS[name]
    bundle = DataBundle(settings.work_dir, settings.image_size)
    if preset.dataset == "asl":
        space, train, val = ASL_SPACE, bundle.asl("train"), bundle.asl("val")
    else:
        space, train, val = AUSLAN_SPACE, bundle.kaggle("train"), bundle.kaggle("val")
    log.info("training %s on %d images (%s)", preset.method, len(train), preset.dataset)
    classifier = create(preset.method, seed=seed).fit(train, val=val)
    if hasattr(classifier, "use_device"):
        classifier.use_device("cpu")  # the camera loop runs on the CPU, like the speed test
    model = LiveModel(
        classifier, space.letters, preset.input_mode, settings.image_size, preset.description
    )
    path = settings.artifacts_dir / "live" / f"{name}.joblib"
    save_live_model(model, path)
    return path
