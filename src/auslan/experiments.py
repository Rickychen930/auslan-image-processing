"""The experiments of Table 7, expressed as lists of runs.

E4 (confusions) and E6 (image style) need no extra runs: every P2/P3 run is also scored on
Roboflow without the style transform, and E4 is computed from the saved predictions.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from auslan.config import Settings
from auslan.results import RunSpec

HEADLINE_K = 20
# Quick hand-crafted methods first, so a valid comparison exists even if deep runs overrun.
CORE_METHODS = (
    "m1_skin_svm",
    "m2_hog_svm",
    "m5_landmarks_svm",
    "m5_landmarks_mlp",
    "m4_mobilenetv2",
    "m4_resnet18",
    "m3_cnn",
)
IMAGE_SIZES = (224, 64, 32)  # E7; the default size (128) is already covered by E1


def _few_shot(
    methods: Sequence[str], ks: Sequence[int | None], seeds: Sequence[int], **extra
) -> list[RunSpec]:
    """P2 and P3 runs for every method, k and seed (the ``all`` point is drawn once)."""
    specs = []
    for method in methods:
        for k in ks:
            for seed in seeds[:1] if k is None else seeds:
                specs += [RunSpec(method, p, k, seed, **extra) for p in ("P2", "P3")]
    return specs


def e1(settings: Settings, methods: Sequence[str], seeds: Sequence[int]) -> list[RunSpec]:
    """All methods under P1, P2 and P3 at k=20, plus P2/P3 on all data once."""
    p1 = [RunSpec(m, "P1", seed=s) for m in methods for s in seeds]
    return p1 + _few_shot(methods, [HEADLINE_K, None], seeds)


def e3(settings: Settings, methods: Sequence[str], seeds: Sequence[int]) -> list[RunSpec]:
    """The k curve."""
    return _few_shot(methods, [*settings.k_values, None], seeds)


def e5(settings: Settings, methods: Sequence[str], seeds: Sequence[int]) -> list[RunSpec]:
    """Within-family alternatives from Section 4, at k=20 (and P1 for the ASL check)."""
    variants = [
        ("m1_skin_knn", {}, "knn"),
        ("m2_hog_svm", {"cell_size": 6}, "cell6"),
        ("m2_hog_svm", {"cell_size": 12}, "cell12"),
        ("m2_hog_rf", {}, "rf"),
        ("m5_landmarks_rf", {}, "rf"),
        ("m5_landmarks_svm", {"two_hand_distances": False}, "no-contact"),
        ("m5_landmarks_mlp", {"two_hand_distances": False}, "no-contact"),
    ]
    specs = []
    for method, options, tag in variants:
        specs += [RunSpec(method, "P1", seed=s, options=options, tag=tag) for s in seeds]
        specs += _few_shot([method], [HEADLINE_K], seeds, options=options, tag=tag)
    return specs


def e7(settings: Settings, methods: Sequence[str], seeds: Sequence[int]) -> list[RunSpec]:
    """Image size versus accuracy at k=20; landmark methods do not depend on pixels."""
    pixel_methods = [m for m in methods if not m.startswith("m5_")]
    specs = []
    for size in IMAGE_SIZES:
        specs += _few_shot(pixel_methods, [HEADLINE_K], seeds, image_size=size, tag=f"size{size}")
    return specs


EXPERIMENTS: dict[str, Callable[[Settings, Sequence[str], Sequence[int]], list[RunSpec]]] = {
    "E1": e1,
    "E3": e3,
    "E5": e5,
    "E7": e7,
}
PRIORITY_ORDER = ("E1", "E3", "E5", "E7")


def plan(
    experiment: str,
    settings: Settings,
    methods: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    protocols: Sequence[str] | None = None,
) -> list[RunSpec]:
    """Runs for one experiment id, or for everything in priority order with ``"all"``."""
    chosen = tuple(methods) if methods else CORE_METHODS
    chosen_seeds = tuple(seeds) if seeds else settings.seeds
    names = PRIORITY_ORDER if experiment == "all" else (experiment,)
    unknown = [n for n in names if n not in EXPERIMENTS]
    if unknown:
        raise KeyError(f"unknown experiment {unknown[0]!r}; choose from {', '.join(EXPERIMENTS)}")
    specs = [spec for n in names for spec in EXPERIMENTS[n](settings, chosen, chosen_seeds)]
    if protocols:
        specs = [spec for spec in specs if spec.protocol in protocols]
    return list({spec.key: spec for spec in specs}.values())  # drop duplicates, keep order
