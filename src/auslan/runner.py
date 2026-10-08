"""Runs one ``RunSpec`` end to end under protocol P1, P2 or P3 (report, Section 6.1)."""

from __future__ import annotations

import logging
import time
from dataclasses import replace

import joblib
import numpy as np

from auslan.config import Settings
from auslan.data.bundle import DataBundle
from auslan.data.sample_set import SampleSet
from auslan.data.splits import sample_k_shot, validation_subset
from auslan.evaluation.metrics import accuracy, macro_f1, top_k_accuracy
from auslan.evaluation.speed import classify_ms
from auslan.methods import Classifier, create
from auslan.results import Result, ResultStore, RunSpec

log = logging.getLogger(__name__)

TOP_K = 3  # a tutor can offer three letters
# Landmark methods learn nothing from images where MediaPipe finds no hand. The report's risk
# register says to report M5 under P1 only if detection on the Kaggle drawings is too low.
MIN_DETECTION = 0.10
Plan = tuple[SampleSet, SampleSet, dict[str, SampleSet]]  # train, validation, test sets


def _detection_rate(samples: SampleSet) -> float:
    return float((samples.frame["n_hands"] > 0).mean())


class Runner:
    def __init__(self, settings: Settings, store: ResultStore) -> None:
        self._settings = settings
        self._store = store
        self._bundles: dict[int, DataBundle] = {}
        self._pretrained_dir = settings.artifacts_dir / "pretrained"

    def run(self, spec: RunSpec) -> bool:
        """Train and evaluate ``spec``; returns False if it was skipped."""
        if self._store.has(spec.key):
            log.info("skip %s: already done", spec.key)
            return False
        model = create(spec.method, seed=spec.seed, **spec.options)
        if spec.protocol == "P3" and not model.supports_transfer:
            log.info("skip %s: hand-crafted features cannot transfer, so P3 equals P2", spec.key)
            return False
        bundle = self._bundle(spec.image_size or self._settings.image_size)
        train, val, tests = self._plan(spec, bundle)
        if (
            model.uses_landmarks
            and spec.protocol != "P1"
            and _detection_rate(train) < MIN_DETECTION
        ):
            log.warning("skip %s: hands are found in too few training images", spec.key)
            return False
        init = self._asl_pretrained(spec, bundle) if spec.protocol == "P3" else None

        started = time.perf_counter()
        model.fit(train, val=val, init=init)
        fit_seconds = time.perf_counter() - started

        results, predictions = self._evaluate(spec, model, train, tests, fit_seconds)
        self._store.add(results, predictions)
        scores = ", ".join(f"{r.test_set} F1={r.macro_f1:.3f}" for r in results)
        log.info("done %s in %.0fs: %s", spec.key, fit_seconds, scores)
        return True

    def _bundle(self, image_size: int) -> DataBundle:
        if image_size not in self._bundles:
            self._bundles[image_size] = DataBundle(self._settings.work_dir, image_size)
        return self._bundles[image_size]

    def _plan(self, spec: RunSpec, bundle: DataBundle) -> Plan:
        if spec.protocol == "P1":
            tests = {"asl_test": bundle.asl("test"), "asl_cross": bundle.asl_cross()}
            return bundle.asl("train"), bundle.asl("val"), tests
        drawn = sample_k_shot(bundle.kaggle_frame("train"), spec.k, spec.seed)
        val_frame = validation_subset(
            bundle.kaggle_frame("val"), self._settings.val_per_class, spec.seed
        )
        tests = {
            "kaggle_test": bundle.kaggle("test"),
            "roboflow": bundle.roboflow(styled=True),  # the main Auslan test
            "roboflow_raw": bundle.roboflow(styled=False),  # E6: without the style transform
        }
        return bundle.kaggle_set(drawn), bundle.kaggle_set(val_frame), tests

    def _asl_pretrained(self, spec: RunSpec, bundle: DataBundle) -> Classifier:
        """ASL-trained model that P3 starts from, trained once per method, seed and image size."""
        size = spec.image_size or self._settings.image_size
        tag = f"-{spec.tag}" if spec.tag else ""
        path = self._pretrained_dir / f"{spec.method}{tag}-s{spec.seed}-{size}px.joblib"
        if path.exists():
            return joblib.load(path)
        log.info("pretraining %s on ASL (style T applied)", path.stem)
        model = create(spec.method, seed=spec.seed, **spec.options)
        model.fit(bundle.asl("train", styled=True), val=bundle.asl("val", styled=True))
        if hasattr(model, "use_device"):
            model.use_device("cpu")  # keep the saved weights loadable on any machine
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
        return model

    def _evaluate(
        self,
        spec: RunSpec,
        model: Classifier,
        train: SampleSet,
        tests: dict[str, SampleSet],
        fit_seconds: float,
    ) -> tuple[list[Result], dict[str, tuple[np.ndarray, np.ndarray]]]:
        n_classes = len(train.label_space)
        results, predictions = [], {}
        for name, samples in tests.items():
            proba = model.predict_proba(samples)
            predictions[name] = (samples.labels, proba)
            results.append(
                Result(
                    key=spec.key,
                    method=spec.method,
                    protocol=spec.protocol,
                    k=spec.k,
                    seed=spec.seed,
                    tag=spec.tag,
                    test_set=name,
                    n_train=len(train),
                    macro_f1=macro_f1(samples.labels, proba.argmax(axis=1), n_classes),
                    accuracy=accuracy(samples.labels, proba.argmax(axis=1)),
                    top3=top_k_accuracy(proba, samples.labels, TOP_K),
                    fit_seconds=fit_seconds,
                    ms_classify=0.0,
                )
            )
        speed = classify_ms(model, next(reversed(tests.values())))  # last set, moves model to CPU
        return [replace(r, ms_classify=speed) for r in results], predictions
