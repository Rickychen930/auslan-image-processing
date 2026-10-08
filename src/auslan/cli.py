"""Command line entry point: ``auslan download | prepare | run | speed | report | methods``."""

from __future__ import annotations

import argparse
import json
import logging
from functools import partial
from pathlib import Path

import pandas as pd

from auslan.config import Settings, load_settings
from auslan.data import download
from auslan.data.prepare import prepare_source
from auslan.data.registry import dataset_specs
from auslan.evaluation.speed import crop_ms
from auslan.experiments import EXPERIMENTS, plan
from auslan.methods import method_names
from auslan.preprocessing.factory import build_cropper
from auslan.report import CROP_TIMING_FILE, build_report
from auslan.results import PROTOCOLS, ResultStore
from auslan.runner import Runner

log = logging.getLogger("auslan")
PREPARE_ORDER = ("asl_alphabet", "asl_alphabet_test", "auslan_kaggle", "auslan_roboflow")
# A test set must not contain images of the set the model learns from.
OVERLAP_REFERENCE = {"asl_alphabet_test": "asl_alphabet", "auslan_roboflow": "auslan_kaggle"}


def _download(settings: Settings, args: argparse.Namespace) -> None:
    download.download_models(settings.models_dir)
    download.download_kaggle(settings.raw_dir)
    if args.roboflow:
        count = download.download_roboflow(settings.raw_dir, args.roboflow_version)
        log.info("Roboflow: %d images arranged by letter", count)
    else:
        log.info("Roboflow skipped; see docs/DATASETS.md (needs a free API key)")


def _prepare(settings: Settings, args: argparse.Namespace) -> None:
    specs = dataset_specs(settings.raw_dir)
    unknown = set(args.names) - set(specs)
    if unknown:
        raise SystemExit(f"unknown dataset(s): {', '.join(sorted(unknown))}")
    for name in args.names or PREPARE_ORDER:
        spec = specs[name]
        if not spec.root.is_dir():
            log.warning("skipping %s: %s does not exist", name, spec.root)
            continue
        reference = None
        if name in OVERLAP_REFERENCE:
            reference = pd.read_csv(settings.work_dir / OVERLAP_REFERENCE[name] / "hashed.csv")
        factory = partial(build_cropper, settings.models_dir, spec.fallback, settings.crop_size)
        prepare_source(
            spec,
            settings.work_dir,
            factory,
            seed=settings.split_seed,
            max_hamming=settings.near_duplicate_hamming,
            workers=settings.workers,
            cropper_key=f"{spec.fallback}-{settings.crop_size}",
            reference=reference,
        )


def _run(settings: Settings, args: argparse.Namespace) -> None:
    specs = plan(args.experiment, settings, args.methods, args.seeds, args.protocols)
    if args.limit:
        specs = specs[: args.limit]
    log.info("%d runs planned", len(specs))
    if args.dry_run:
        print("\n".join(spec.key for spec in specs))
        return
    runner = Runner(settings, ResultStore(settings.artifacts_dir))
    for number, spec in enumerate(specs, 1):
        log.info("[%d/%d] %s", number, len(specs), spec.key)
        runner.run(spec)


def _speed(settings: Settings, args: argparse.Namespace) -> None:
    """Time hand detection plus cropping on colour photos (the same for every method)."""
    name = "auslan_roboflow"
    if not (settings.work_dir / name / "hashed.csv").exists():
        log.warning("Roboflow is not prepared; timing on ASL Alphabet Test photos instead")
        name = "asl_alphabet_test"
    paths = pd.read_csv(settings.work_dir / name / "hashed.csv")["path"].tolist()
    cropper = build_cropper(settings.models_dir, "skin", settings.crop_size)
    milliseconds = crop_ms(cropper, paths)
    target = settings.artifacts_dir / CROP_TIMING_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"crop_ms": milliseconds}))
    log.info("detection and cropping: %.1f ms per frame", milliseconds)


def _report(settings: Settings, args: argparse.Namespace) -> None:
    folder = build_report(ResultStore(settings.artifacts_dir), settings)
    log.info("report written to %s", folder)


def _methods(settings: Settings, args: argparse.Namespace) -> None:
    print("\n".join(method_names()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="auslan", description=__doc__)
    parser.add_argument("--config", type=Path, help="YAML file overriding the defaults")
    commands = parser.add_subparsers(dest="command", required=True)

    get = commands.add_parser("download", help="fetch datasets and MediaPipe models")
    get.add_argument("--roboflow", action="store_true", help="also fetch Roboflow (needs a key)")
    get.add_argument("--roboflow-version", type=int, help="default: the newest version")
    get.set_defaults(handler=_download)

    prep = commands.add_parser("prepare", help="de-duplicate, split and crop the datasets")
    prep.add_argument("names", nargs="*", help=f"any of: {', '.join(PREPARE_ORDER)} (default: all)")
    prep.set_defaults(handler=_prepare)

    run = commands.add_parser("run", help="train and evaluate an experiment")
    run.add_argument("experiment", choices=[*EXPERIMENTS, "all"])
    run.add_argument("--methods", nargs="+", choices=method_names())
    run.add_argument("--seeds", nargs="+", type=int)
    run.add_argument("--protocols", nargs="+", choices=PROTOCOLS, help="e.g. P1 before Roboflow")
    run.add_argument("--limit", type=int, help="only the first N runs")
    run.add_argument("--dry-run", action="store_true", help="list the runs and stop")
    run.set_defaults(handler=_run)

    commands.add_parser("speed", help="time detection and cropping").set_defaults(handler=_speed)
    commands.add_parser("report", help="build the result tables").set_defaults(handler=_report)
    commands.add_parser("methods", help="list method names").set_defaults(handler=_methods)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.handler(load_settings(args.config), args)
