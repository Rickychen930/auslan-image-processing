# Auslan fingerspelling recognition

Code for the UTS 42177 *Image Processing and Pattern Recognition* project (Group 9). It answers
the question set in the Task 1 report: **how well do methods trained on one-handed ASL transfer to
the two-handed Auslan alphabet, and does ASL training reduce the number of Auslan examples a
method needs?**

Five method families are trained and tested under one shared pipeline, so differences in the
results come from the methods and not from data handling.

| ID | Method | Registry names |
|----|--------|----------------|
| M1 | Skin / edge-map segmentation, shape features (Hu moments, convexity), kNN or SVM | `m1_skin_svm`, `m1_skin_knn` |
| M2 | HOG, PCA, SVM or random forest | `m2_hog_svm`, `m2_hog_rf` |
| M3 | Four-block CNN from random weights | `m3_cnn` |
| M4 | ImageNet-pretrained MobileNetV2 / ResNet-18, fine-tuned | `m4_mobilenetv2`, `m4_resnet18` |
| M5 | MediaPipe 2 x 21 hand landmarks, SVM / random forest / MLP | `m5_landmarks_svm`, `m5_landmarks_rf`, `m5_landmarks_mlp` |

## Quick start

```bash
uv sync                                  # Python 3.12 environment, see docs/SETUP.md
uv run auslan download                   # Kaggle datasets + MediaPipe models (~7.2 GB)
uv run auslan prepare                    # de-duplicate, split, crop hands, cache landmarks
uv run auslan run E1                     # headline experiment: all methods, P1/P2/P3
uv run auslan speed                      # time hand detection and cropping
uv run auslan report                     # tables under artifacts/report/
```

The Roboflow test set needs a free API key, see [docs/DATASETS.md](docs/DATASETS.md). Runs are
resumable: finished runs are skipped, so an interrupted `auslan run` can simply be started again.
`uv run auslan run E1 --dry-run` lists the runs without training anything.

## Protocols and experiments

| Protocol | Train | Test |
|----------|-------|------|
| P1 | ASL Alphabet (24 letters) | held-out ASL split and ASL Alphabet Test |
| P2 | `k` Kaggle Auslan images per letter, learned from scratch (M4 keeps ImageNet weights) | held-out Kaggle split and Roboflow Auslan |
| P3 | as P2, but starting from the weights learned on ASL (new 26-class output layer) | as P2 |

Hand-crafted methods (M1, M2, M5 SVM/RF) cannot reuse ASL training, so for them P3 equals P2 and
the runner skips it. The experiments of the report's Table 7 are in `src/auslan/experiments.py`:

| ID | Command | What it runs |
|----|---------|--------------|
| E1, E2 | `auslan run E1` | every method under P1, P2, P3 at k=20 (and on all data once); same-dataset versus cross-dataset scores come from the same runs |
| E3 | `auslan run E3` | the k curve: 5, 10, 20, 50 and all |
| E4 | `auslan report` | per-letter F1 and most confused pairs, from saved predictions |
| E5 | `auslan run E5` | within-family alternatives (HOG cell size, kNN, random forest, landmark contact features) |
| E6 | `auslan run E1` | every P2/P3 run is also scored on Roboflow without the style transform |
| E7 | `auslan run E7` | image size 224 / 64 / 32 against the default 128 |

`auslan report` writes `table9.md` (the Task 2 results template), `k_curve.csv`,
`per_letter_f1.csv`, `confused_pairs.md`, `significance.md` (McNemar with Holm correction and the
paired bootstrap for ΔT) and `success_criteria.md` (S2 to S4 checked against the targets).

## Layout

```
src/auslan/
  data/           label spaces, catalog, de-duplication, splits, prepared sample sets, downloads
  preprocessing/  hand cropping, MediaPipe adapters, skin-colour mask
  methods/        the five families behind one Classifier interface + a name registry
  evaluation/     metrics, McNemar / Holm / bootstrap, timing
  runner.py       one run under P1, P2 or P3
  experiments.py  E1 to E7 as lists of runs
  reporting.py    result tables
  cli.py          the `auslan` command
tests/            unit tests and a toy end-to-end run
docs/             setup, datasets, related work
```

Design notes:

- **One interface.** A method only has to provide `fit(train, val, init)` and `predict_proba`.
  The runner, metrics and reports do not know which family they are handling, and a new method is
  one decorated function (`@register("name")`).
- **No leakage.** Exact and near-duplicate images (perceptual hash) are removed before splitting.
  The test sets are never used for tuning; early stopping looks at 10 validation images per letter.
  Roboflow is cleaned of anything that nearly matches a Kaggle image.
- **Hands are never flipped.** Horizontal flips would swap the two hands, so augmentation uses
  small rotations, crops and colour jitter only. Hands are ordered by wrist position, not by the
  detector's left/right label.
- **Failures are measured.** `catalog.csv` records how many hands were found per image and whether
  the fallback crop was used, which is the detection rate the report asks for.

## Differences from the report to keep in mind

- CNNs use 128 x 128 inputs by default (the report's risk register allows this for laptop-sized
  training); crops are cached at 224 and E7 covers the other sizes.
- The near-duplicate threshold (4 of 64 hash bits, `near_duplicate_hamming`) removes many
  consecutive video frames, so "all data" is smaller than the nominal 1,400 images per letter.
  The real counts are printed by `auslan prepare` and listed in docs/DATASETS.md.
- Frozen-backbone variants (an example in E5) are not implemented.

## Development

```bash
uv run pytest            # unit tests; the toy end-to-end tests take a few minutes
uv run ruff check src tests
uv run ruff format src tests
```
