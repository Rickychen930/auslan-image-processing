# v2 results: Roboflow crop fix and full re-run (run on 10 October 2026)

Every experiment with the Roboflow crop fix: E5, E1, E3 and E7 in that order, then `auslan speed`
and `auslan report`, on a MacBook (Apple silicon, MPS). Produced from branch
`fix/roboflow-signer-crop` (PR #1, commit `8ae26dd`, since merged into `main` as `294a483`) on top
of `main` at `ee25836`. The
experiments took 5 h 3 min. Training and evaluation code is the same as Brian's run in
[`results/brian/e1`](../../brian/e1/README.md); only the Roboflow crops differ.

**Same as Brian's run:**
- Both languages use the same 24 letters (no J or Z), so chance is 4.2%.
- Roboflow is version 18: 250 test images (7 to 25 per letter).
- MediaPipe finds no hand in any Kaggle Auslan image, so M5 runs under P1 only.
- Prepared counts match exactly: ASL 16,980, ASL Test 704, Kaggle Auslan 11,028, Roboflow 250.

**Different from Brian's run:**
- **Roboflow crops.** 115 of 250 Roboflow images show 3 or 4 hands (two signers, or a close-up
  inset beside the full-body signer), and the old crop often took one hand from each. The fix keeps
  one signer's pair; 90 of 250 crops changed (evidence in [`../crop-fix`](../crop-fix/README.md)).
- **Machine.** Speed was measured on this Mac: detection takes 12 ms per frame here and 48 ms on
  Brian's M2 Pro, so frame rates are not comparable between the two runs.
- **Coverage.** Brian ran E1 (E2, E4 and E6 come from the same runs); this run adds E3, E5 and E7.

## Summary

The research question (Task 1 report, Section 1.2) asks how well methods trained on one-handed ASL
transfer to the two-handed Auslan alphabet, and whether ASL pretraining reduces the Auslan examples
that learned methods need compared with hand-crafted methods that cannot transfer.

- **Transfer to real Auslan images fails for every method.** On the independent Roboflow test the
  best macro-F1 is 10.7% at k=20 (13.1% at 224 px), against 80 to 85% on the Kaggle split. The
  experiments point to the gap between the two Auslan datasets, not to the methods or the pipeline.
- **ASL pretraining reduces the data a CNN trained from scratch needs** (about 2.5 times fewer
  Auslan images on the Kaggle split) and gives MobileNetV2 a small, significant gain on Roboflow
  (S3 met). It adds nothing to ImageNet-pretrained models on the Kaggle split.
- **Hand landmarks are the only representation that survives a change of dataset** (ASL: −0.1
  points), but they cannot be used on the Kaggle Auslan images, so this could not be tested on Auslan.

## Table 9: macro-F1 %, mean over three draws, k=20 (Brian → v2)

| Method | P1 ASL Test | P2 Roboflow | P3 Roboflow | P3 95% CI (v2) | ΔT(20) | Top-3 P3 | ms/frame |
|---|---|---|---|---|---|---|---|
| M1 skin + SVM | 5.0 → 4.9 | 3.1 → 3.3 | = P2 | [2.0, 4.6] | n/a | 10.8 → 11.2 | 51 → 14 |
| M2 HOG + SVM | 43.9 → 44.0 | 6.6 → 7.1 | = P2 | [4.9, 9.2] | n/a | 21.3 → 24.0 | 53 → 15 |
| M3 CNN (no ImageNet) | 24.1 → 24.0 | 0.7 → 0.6 | 1.7 → 0.9 | [0.7, 1.3] | +1.0 → +0.3 | 14.4 → 14.7 | 52 → 14 |
| M4 MobileNetV2 | 67.8 → 67.9 | 5.2 → 6.3 | 6.8 → **9.2** | [6.9, 10.9] | +1.5 → **+2.8** | 22.9 → 25.9 | 88 → 28 |
| M4 ResNet-18 | 68.7 → 70.3 | 8.3 → 9.2 | 8.0 → **10.7** | [8.1, 12.7] | −0.3 → +1.5 | 22.0 → 24.9 | 56 → 16 |
| M5 landmarks + SVM | 83.8 → 83.8 | – | – | – | n/a | – | – → 14 |
| M5 landmarks + MLP | 69.9 → 70.2 | – | – | – | – | – | – → 12 |

Standard deviations and per-draw intervals are in [`table9.md`](table9.md). Speed includes hand
detection. Top-3 for M1 and M2 is their P2 score.

## Success criteria and hypotheses

| Item | Target | Brian | v2 |
|---|---|---|---|
| S1 implementation check | within 5 points of published | met for M4 (99.7% vs 99%), not for M5 SVM (78.8% vs 87.6%) | same |
| S2 Roboflow, P3, k=20 | macro-F1 ≥ 60%, top-3 ≥ 85% | not met: best 8.0%, top-3 22% | not met: best 10.7%, top-3 25.9% |
| S3 value of ASL training | ΔT(20) > 0 for MobileNetV2 | not met: CI [−0.8, +3.7] | **met: CI [+0.5, +5.0]** |
| S4 real-time speed | ≥ 10 frames/s | met (11 to 20 fps) | met (36 to 82 fps, other machine) |
| S5 explained failures | confused pairs and drop, with example images | partly | partly (no per-letter example images) |
| H1 landmarks lose least on a new dataset | – | supported on ASL | same; untestable on Auslan |
| H2 M4 gains most from ASL at small k, less as k grows | – | not tested | mostly not supported: M3 gains most |
| H3 M1 fails most where the hands overlap | – | not tested | inconclusive: M1 near chance everywhere |

## Experiments: rationale, results and discussion

Each rationale comes from Table 7 and the challenges D1 to D6 in the Task 1 report.

### E1: all methods under P1, P2 and P3 at k=20

**Rationale.** The mandatory headline comparison: which method is best, and how much each loses.
P1 doubles as an implementation check (S1) against published ASL results, so a large gap there
would point to a bug rather than to a finding. The P3 Roboflow column is the primary test of a
usable tutor (S2), and the paired P2/P3 runs give the pre-specified transfer test (S3).

**Results.** See Table 9. On ASL, M5 SVM is best on the cross-dataset test (83.8%), followed by
the two M4 models (68 to 70%). On Roboflow, ResNet-18 P3 is best (10.7%, top-3 24.9%), and only
one pair of methods differs significantly: ResNet-18 better than M1 (Holm-adjusted p = 0.017,
[`significance.md`](significance.md)). MobileNetV2's ΔT(20) is +2.8 points, CI [+0.5, +5.0].

**Discussion.** S1 confirms the M4 implementation (99.7% against a published 99%). M5 SVM falls
8.8 points short of Shin et al. (2021); their figure uses accuracy on all 29 classes, so part of
the gap is the different class set, but our MediaPipe also finds no hand in 21.5% of ASL Alphabet
images, which leaves M5 with zero-filled features there. S2 fails by a wide margin: no method is
usable as a tutor on unseen Auslan images. S3 passes only with the crop fix and only just, on
scores barely above chance, so it should be reported as a small but significant gain.

### E2: same dataset against another dataset

**Rationale.** No public two-handed dataset identifies its signers (D4), so a strict unseen-signer
test is impossible. Testing on a different dataset (other people, cameras, backgrounds; D5) is the
closest substitute and tests H1.

**Results** ([`cross_dataset.md`](cross_dataset.md)):

| Method | ASL: same → other dataset | Auslan (k=20): Kaggle → Roboflow |
|---|---|---|
| M5 landmarks + SVM | 83.9 → 83.8 (−0.1) | not testable |
| M5 landmarks + MLP | 71.4 → 70.2 (−1.2) | not testable |
| M4 ResNet-18 | 99.7 → 70.3 (−29.4) | P3 85.1 → 10.7 (−74.4) |
| M4 MobileNetV2 | 99.7 → 67.9 (−31.8) | P3 84.6 → 9.2 (−75.5) |
| M2 HOG + SVM | 98.3 → 44.0 (−54.3) | P2 80.1 → 7.1 (−72.9) |
| M3 CNN | 95.8 → 24.0 (−71.8) | P3 38.1 → 0.9 (−37.2) |
| M1 skin + SVM | 61.1 → 4.9 (−56.2) | P2 34.6 → 3.3 (−31.3) |

**Discussion.** H1 is supported on ASL: landmarks describe the hand's geometry and ignore
background, lighting and clothing, while every pixel-based method learns part of the recording
setting. The ASL Alphabet set is one signer against a plain wall, so near-perfect same-dataset
scores mean little. The Auslan drop is two to three times larger than the ASL drop for the same
methods, because the two Auslan sets differ in kind, not only in setting: Kaggle Auslan is
binarised edge drawings of a few sessions, Roboflow is colour video frames of about eight
teaching videos. The Kaggle split also shares sessions with its training part, so its scores are
an optimistic in-source reference, as the Task 1 report anticipated.

### E3: the k curve

**Rationale.** Auslan data is scarce (D6). E3 answers the second half of the research question by
training on k = 5, 10, 20, 50 and all images per letter, under P2 (Auslan only) and P3 (from ASL),
and tests H2.

**Results** (Kaggle split, P2 → P3; [`k_curve.csv`](k_curve.csv)):

| Method | k=5 | k=10 | k=20 | k=50 | all |
|---|---|---|---|---|---|
| M3 CNN | 27.3 → 34.6 | 27.7 → 37.7 | 25.9 → 38.1 | 39.5 → 55.8 | 76.7 → 83.0 |
| M4 MobileNetV2 | 59.3 → 54.2 | 73.3 → 71.6 | 85.6 → 84.6 | 94.9 → 94.7 | 99.2 → 99.1 |
| M4 ResNet-18 | 56.1 → 54.4 | 73.4 → 71.4 | 87.0 → 85.1 | 95.7 → 95.4 | 99.5 → 99.4 |
| M2 HOG + SVM (P2) | 47.6 | 63.8 | 80.1 | 93.5 | 99.8 |

Roboflow, P2 → P3:

| Method | k=5 | k=10 | k=20 | k=50 | all |
|---|---|---|---|---|---|
| M4 MobileNetV2 | 5.6 → 8.3 | 6.9 → 10.9 | 6.3 → 9.2 | 10.8 → 9.4 | 14.0 → 7.3 |
| M4 ResNet-18 | 7.3 → 7.6 | 7.5 → 8.7 | 9.2 → 10.7 | 10.4 → 12.9 | 8.2 → 11.6 |
| M2 HOG + SVM (P2) | 8.7 | 7.6 | 7.1 | 6.5 | 5.5 |

M1 and M3 stay between 0.6 and 3.3% on Roboflow at every k. "all" is one draw (249 to 484
training images per letter after de-duplication).

**Discussion.** ASL pretraining helps most where nothing else is pretrained: M3 gains 6 to 16
points at every k, and with ASL at k=20 (38.1%) it matches M3 from scratch at k=50 (39.5%), about
2.5 times fewer Auslan images. For M4 the gain is zero or negative on the Kaggle split (−0.1 to
−5.1): ImageNet already supplies general edge and shape features, and one-handed ASL shapes add
nothing for two-handed Auslan drawings. H2 is therefore mostly not supported. Only MobileNetV2 on
Roboflow follows the predicted pattern (a gain of 2.6 to 4.0 at k ≤ 20 that turns negative at
k ≥ 50); ResNet-18's Roboflow gain grows with k instead. On Roboflow no curve rises with k, and
HOG falls (8.7 to 5.5): more images from the same source make the models fit that source more
closely, which is further evidence that the problem is the dataset gap, not the amount of data.

### E4: per-letter results and confused pairs

**Rationale.** Some letters differ only in which fingertip is touched (D2), and skin masks merge
overlapping hands (D1, H3). E4 checks whether particular letters are hard for every method.

**Results** (Roboflow, P3 or P2 at k=20, pooled over draws;
[`confused_pairs.md`](confused_pairs.md), [`per_letter_f1.csv`](per_letter_f1.csv)):

| Method | Most confused pairs | Best letters (F1 %) | Letters at 0% |
|---|---|---|---|
| M4 ResNet-18 | A/Y (25), A/V, A/W, A/G | B 32, F 31, W 28 | 3 (N, O, U) |
| M4 MobileNetV2 | A/Y (19), V/W, A/T, A/V | B 27, X 24, N 18 | 4 |
| M2 HOG + SVM | G/S (19), G/Y, G/P, D/G, G/T | W 26, F 24, B 17 | 8 |
| M1 skin + SVM | C/K (12), F/L, E/I, E/Q, E/R | W 13, F 11, V 8 | 7 |
| M3 CNN | F/Y (38), F/V, F/Q, C/F, F/W | V 11, F 8, M 7 | 19 |

**Discussion.** Most errors are not fine-grained confusions between similar handshapes. Each
method's pairs share one letter (A for M4, G for HOG, F for the CNN): under the domain shift the
models collapse their predictions onto a few letters, which is the typical sign of inputs unlike
anything seen in training. The letters that work best (B, F, W) have distinctive two-handed
outlines that survive the style change. H3 cannot be judged: M1 is near chance on almost every
letter, so overlapping-hand letters do not stand out. S5 still needs example images per pair.

### E5: design variants within each family

**Rationale.** Section 4 of the Task 1 report names alternatives for each family. E5 checks which
design choices matter, so the defaults can be defended in the Task 2 report.

**Results** (ASL Test is P1 on ASL Alphabet Test; Kaggle is the Kaggle split at k=20, P2 or P3
where shown; Roboflow is 0.7 to 8.5% for every variant; [`variants.md`](variants.md)):

| Variant | ASL Test: variant / default | Kaggle: variant / default | Verdict |
|---|---|---|---|
| M3 CNN, 5 blocks | 38.7 / 24.0 | P3 75.7 / 38.1 | much better |
| M3 CNN, 3 blocks | 9.1 / 24.0 | P3 10.4 / 38.1 | much worse |
| M3 CNN, no augmentation | 17.3 / 24.0 | P3 55.8 / 38.1 | worse across datasets, better in-source |
| M4 MobileNetV2, frozen | 31.6 / 67.9 | 43.9 / 85.6 | fine-tuning essential |
| M4 ResNet-18, frozen | 28.6 / 70.3 | 34.8 / 87.0 | fine-tuning essential |
| M2 HOG, linear kernel | 33.5 / 44.0 | 79.0 / 80.1 | RBF default better |
| M2 HOG, cell 12 px | 47.3 / 44.0 | 80.0 / 80.1 | no clear difference |
| M2 HOG, cell 6 px | 41.2 / 44.0 | 78.6 / 80.1 | no clear difference |
| M2 HOG, random forest | 46.6 / 44.0 | 75.5 / 80.1 | SVM better in-source |
| M5 landmarks, random forest | 80.8 / 83.8 | – | SVM default better |
| M5 SVM, no two-hand distances | 83.7 / 83.8 | – | no difference (ASL is one-handed) |
| M5 MLP, no two-hand distances | 69.8 / 70.2 | – | no difference |
| M1 skin, kNN | 4.9 / 4.9 | 31.9 / 34.6 | no difference |

**Discussion.** Depth is the choice that matters most. The 4-block default is too shallow for
128 px input: its receptive field likely cannot cover a whole two-handed sign, and the 5-block network
fixes that (E7 points the same way). Fine-tuning is essential for M4: a frozen ImageNet backbone
halves the ASL Test score, and P3 equals P2 because ASL training then only touches the output
layer, which P3 replaces. Augmentation trades in-source accuracy for robustness to new data,
which is the goal here, so the default keeps it. The two-hand distance features (the direct test
of D1) cannot be judged on one-handed ASL; they need Auslan landmarks. Differences under about
3 points (HOG cell size, M1 kNN) are within draw-to-draw variation.

### E6: Roboflow with and without the style transform T

**Rationale.** The Kaggle Auslan images are processed before release (D6). E6 measures how much of
the cross-dataset drop comes from that processing, by scoring Roboflow both as is and with T
(greyscale, Gaussian blur and adaptive threshold, which makes a photo resemble a Kaggle drawing).

**Results** (k=20; [`style_effect.md`](style_effect.md)):

| Method | With T | Without T | Change |
|---|---|---|---|
| M4 ResNet-18 (P3) | 10.7 | 3.6 | −7.1 |
| M4 MobileNetV2 (P3) | 9.2 | 3.2 | −5.9 |
| M2 HOG + SVM (P2) | 7.1 | 10.6 | +3.5 |
| M3 CNN (P3) | 0.9 | 0.9 | 0.0 |
| M1 skin + SVM (P2) | 3.3 | 3.3 | 0.0 (always uses the colour photo) |

**Discussion.** Without T, the M4 models fall to chance: a network trained only on black-and-white
drawings cannot read a colour photo, so the style difference alone accounts for most of their
remaining score. T is not enough, though: the transformed photos have thicker, solid outlines and
include faces and backgrounds, unlike the thin dotted Kaggle edges (see the crop images in
[`../crop-fix`](../crop-fix/README.md)). HOG does better without T, likely because the adaptive
threshold adds speckle that disturbs its gradient histograms. The Task 1 report planned a blur
matched by Laplacian variance; once the Kaggle images turned out to be binarised, not just blurred,
the transform had to binarise as well.

### E7: image size

**Rationale.** Small fingertip differences may need detail (D2), while a laptop tutor needs small,
fast inputs (S4). E7 compares 32, 64, 128 (default) and 224 px for the pixel-based methods.

**Results** (Kaggle split P2 / Roboflow; Roboflow is P3 for M3 and M4, P2 for HOG;
[`image_size.md`](image_size.md)):

| Method | 32 px | 64 px | 128 px (default) | 224 px |
|---|---|---|---|---|
| M4 ResNet-18 | 71.3 / 3.3 | 81.2 / 5.8 | 87.0 / 10.7 | **88.5 / 13.1** |
| M4 MobileNetV2 | 55.3 / 4.3 | 76.0 / 6.5 | 85.6 / 9.2 | **86.9 / 10.8** |
| M2 HOG + SVM | 63.3 / 0.5 | 78.2 / 7.8 | **80.1** / 7.1 | 76.9 / 5.1 |
| M3 CNN | **50.7** / 2.8 | 22.5 / 2.5 | 25.9 / 0.9 | 19.4 / 0.4 |

M1 stays at 1.8 to 4.2% on Roboflow at every size.

**Discussion.** The M4 models improve with size up to 224 px, the resolution their ImageNet
weights were trained at; ResNet-18 at 224 px gives the best Roboflow score of the project (13.1%),
still far from S2. 128 px is a reasonable trade-off for a laptop. HOG peaks at 64 to 128 px,
likely where its 8 px cells best match the scale of the fingers. M3 does best at 32 px and worst at 224 px, which
confirms the E5 finding that its fixed four blocks are too shallow for larger inputs.

## What changed against Brian's run

Everything not scored on Roboflow reproduces Brian's numbers: P1 within 1.6 points, the Kaggle
test split identical. A controlled check re-scored the same models with old and new crops; Roboflow
F1 rose in 10 of 12 runs (the two drops were −0.1 and −0.2), so the differences come from the
crops alone. The fix raises Roboflow scores by up to 2.7 points (M3 P3 is the one drop, 1.7 to 0.9,
both near zero), lifts M4 top-3 from 22–23% to 25–26%, and moves the S3 interval above zero. The
crop fix is worth keeping because it removes a measurement error, but it changes no conclusion
except S3.

## Overall discussion

**Why Roboflow fails.** Five experiments rule out the obvious explanations one by one: cropping
(the crop fix, +0 to +2.7 points), the amount of training data (E3, flat or falling with k), the
design of each method (E5, every variant at 0.7 to 8.5%), image size (E7, at most 13.1%) and
Brian's checks of framing and mirroring. What remains is the dataset gap. Kaggle Auslan is
binarised close-ups from a few sessions, with no colour, faces cut off and the same clothing
throughout; Roboflow is colour frames from about eight teaching videos with different people,
rooms, on-screen graphics and faces in view. A model can score 99% on the Kaggle split by
recognising the recording set-up, and that knowledge does not transfer.

**What transfers from ASL.** Pretraining on ASL is worth most when a model starts from nothing
(M3) and little when it already has ImageNet features (M4). The gain on Roboflow (S3) is real but
small. Landmarks transfer best between datasets (H1), and they are the representation least
affected by the style gap, so they are the most promising route for Auslan once colour training
images with detectable hands exist.

**Limitations.** Roboflow has 250 images (7 to 25 per letter), so its intervals are wide and only
one method pair differs significantly. The "all" points are single draws. No dataset identifies
its signers, so cross-dataset testing replaces a strict unseen-signer test, and the Kaggle split
likely shares signers with its training part. M5 could not be run on Auslan, which leaves H1, the
MLP's transfer and the two-hand distance features untested there. Speed was measured on a
different machine from Brian's run.

**What would close the gap.** Colour Auslan training images from several people (a few team
members recording 24 letters each would give thousands of frames), held out by person; landmarks
(M5) trained on them; and, as a clearly separate protocol, fine-tuning on part of Roboflow split by
video. The first two keep Roboflow as an untouched test set.

## Open decisions for the team

- PR #1 (the crop fix) is merged into `main`; everyone with prepared data rebuilds Roboflow once:
  `rm -rf data/work/auslan_roboflow && uv run auslan prepare auslan_roboflow`.
- Re-crop ASL with the new code too? Only 0.7% of ASL images show two hands, so P1 would barely move.
- Change the M3 default to 5 blocks (or a 32 px input)? Either roughly doubles its scores.
- Which run feeds Table 9 in the Task 2 report: Brian's E1 (old crops) or this one?
- S5 still needs example images for the five most-confused pairs per method.

## Files

| File | Content |
|---|---|
| `table9.md`, `success_criteria.md`, `significance.md` | headline results and tests (E1) |
| `cross_dataset.md`, `style_effect.md` | E2 (same vs other dataset), E6 (Roboflow with and without the style transform) |
| `confused_pairs.md`, `per_letter_f1.csv`, `confusion_<method>.csv` | E4 |
| `k_curve.csv`, `variants.md`, `image_size.md` | E3, E5, E7 |
| `cost.md`, `detection_rates.md` | training time, model size, speed; MediaPipe detection rates |
| `results.csv`, `results.jsonl` | every run and test set; `results.jsonl` rebuilds the tables with `auslan report` |

To reproduce: copy `configs/default.yaml` to `artifacts/v2.yaml`, set `artifacts_dir: artifacts/v2`,
and run `uv run auslan --config artifacts/v2.yaml run all` (about 5 hours on an M-series Mac).
