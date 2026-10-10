# v2 results: Roboflow crop fix and full re-run (run on 10 October 2026)

Every experiment with the Roboflow crop fix: E5, E1, E3 and E7 in that order, then `auslan speed`
and `auslan report`, on a MacBook (Apple silicon, MPS). Produced from branch
`fix/roboflow-signer-crop` (PR #1, commit `8ae26dd`) on top of `main` at `ee25836`. The
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
- **Coverage.** Brian ran E1; this run adds E3, E5 and E7, so every table below is filled.

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
detection.

## Success criteria and hypotheses

| Item | Target | Brian | v2 |
|---|---|---|---|
| S1 implementation check | within 5 points of published | met for M4 (99.7% vs 99%), not for M5 SVM (78.8% vs 87.6%) | same |
| S2 Roboflow, P3, k=20 | macro-F1 ≥ 60%, top-3 ≥ 85% | not met: best 8.0%, top-3 22% | not met: best 10.7%, top-3 25.9% |
| S3 value of ASL training | ΔT(20) > 0 for MobileNetV2 | not met: CI [−0.8, +3.7] | **met: CI [+0.5, +5.0]** |
| S4 real-time speed | ≥ 10 frames/s | met (11 to 20 fps) | met (36 to 82 fps, other machine) |
| S5 explained failures | confused pairs and drop, with example images | partly | partly (no per-letter example images) |
| H1 landmarks lose least | – | supported on ASL | same; untestable on Auslan |
| H2 M4 gains most at small k | – | not tested | mostly not supported: M3 gains most |
| H3 M1 fails on overlapping hands | – | not tested | inconclusive: M1 near chance everywhere |

S3 passes by a small margin on scores barely above chance: report it as a small but significant
transfer gain. One method pair now differs on Roboflow, ResNet-18 better than M1 (Holm-adjusted
p = 0.017, [`significance.md`](significance.md)); Brian had none.

## What changed against Brian's run

Everything not scored on Roboflow reproduces Brian's numbers: P1 within 1.6 points, the Kaggle
test split identical. A controlled check re-scored the same models with old and new crops; Roboflow
F1 rose in 10 of 12 runs (the two drops were −0.1 and −0.2), so the differences come from the
crops alone. The fix raises Roboflow scores by up to 2.7 points (M3 P3 is the one drop, 1.7 to 0.9,
both near zero), lifts M4 top-3 from 22–23% to 25–26%, and moves the S3 interval above zero.

## Findings

1. **Landmarks survive the change of dataset (supports H1).** M5 SVM scores 83.9% on the ASL split
   and 83.8% on ASL Alphabet Test. Pixel methods drop 29 to 72 points
   ([`cross_dataset.md`](cross_dataset.md)).
2. **Nothing generalises from Kaggle Auslan to Roboflow, even with clean crops.** Methods at 80 to
   85% on the Kaggle split fall to 1 to 11% on Roboflow, and more data (E3), other designs (E5) and
   other image sizes (E7) all leave the best score between 9 and 13%. The gap is between the
   datasets: binarised close-ups from a few sessions against colour video of many signers.
3. **ASL training helps the CNN trained from scratch, not the ImageNet models, on the Kaggle split.**

   | Method | k=5 | k=10 | k=20 | k=50 | all |
   |---|---|---|---|---|---|
   | M3 CNN, P2 → P3 | 27.3 → 34.6 | 27.7 → 37.7 | 25.9 → 38.1 | 39.5 → 55.8 | 76.7 → 83.0 |
   | M4 MobileNetV2, P2 → P3 | 59.3 → 54.2 | 73.3 → 71.6 | 85.6 → 84.6 | 94.9 → 94.7 | 99.2 → 99.1 |
   | M4 ResNet-18, P2 → P3 | 56.1 → 54.4 | 73.4 → 71.4 | 87.0 → 85.1 | 95.7 → 95.4 | 99.5 → 99.4 |

   Starting from ASL with k=20 (38.1%) matches M3 trained from scratch with k=50 (39.5%), about 2.5
   times fewer Auslan images. "all" is one draw.
4. **On Roboflow, ASL training gives MobileNetV2 a small, significant gain (S3 met).**

   | Method | k=5 | k=10 | k=20 | k=50 | all |
   |---|---|---|---|---|---|
   | M4 MobileNetV2, P2 → P3 | 5.6 → 8.3 | 6.9 → 10.9 | 6.3 → 9.2 | 10.8 → 9.4 | 14.0 → 7.3 |
   | M4 ResNet-18, P2 → P3 | 7.3 → 7.6 | 7.5 → 8.7 | 9.2 → 10.7 | 10.4 → 12.9 | 8.2 → 11.6 |
   | M2 HOG + SVM (P2 only) | 8.7 | 7.6 | 7.1 | 6.5 | 5.5 |

   MobileNetV2 gains at k ≤ 20 and loses at k ≥ 50, the pattern H2 predicted; ResNet-18's gain
   instead grows with k. M1 and M3 stay at 0.6 to 3.3% throughout ([`k_curve.csv`](k_curve.csv)).
5. **The default M3 is under-sized (E5, E7).** A 5-block CNN beats the 4-block default by 15 points
   on ASL Test (38.7 against 24.0) and doubles the Kaggle P3 score (75.7 against 38.1). The default
   also scores twice as high at 32 px as at 128 px (50.7 against 25.9 on the Kaggle split).
6. **Fine-tuning is essential for M4 (E5).** Freezing the backbone roughly halves the ASL Test score
   (MobileNetV2 67.9 to 31.6, ResNet-18 70.3 to 28.6).
7. **The style transform helps M4 but hurts HOG (E6).** Without T, ResNet-18 P3 drops from 10.7 to
   3.6 on Roboflow, while HOG rises from 7.1 to 10.6 ([`style_effect.md`](style_effect.md)).
8. **Larger images help M4 (E7).** ResNet-18 P3 reaches 13.1% on Roboflow at 224 px, the best
   Roboflow score in the project ([`image_size.md`](image_size.md)).

## E5: design variants (macro-F1 %, three draws)

ASL Test is P1 on ASL Alphabet Test; Kaggle is the Kaggle split at k=20 (P2, or P3 where shown).
Roboflow is 0.7 to 8.5% for every variant ([`variants.md`](variants.md)).

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

## E7: image size (Kaggle split / Roboflow, k=20)

| Method | 32 px | 64 px | 128 px (default) | 224 px |
|---|---|---|---|---|
| M4 ResNet-18 | 71.3 / 3.3 | 81.2 / 5.8 | 87.0 / 10.7 | **88.5 / 13.1** |
| M4 MobileNetV2 | 55.3 / 4.3 | 76.0 / 6.5 | 85.6 / 9.2 | **86.9 / 10.8** |
| M2 HOG + SVM | 63.3 / 0.5 | 78.2 / 7.8 | **80.1** / 7.1 | 76.9 / 5.1 |
| M3 CNN | **50.7** / 2.8 | 22.5 / 2.5 | 25.9 / 0.9 | 19.4 / 0.4 |

The Kaggle column is P2; Roboflow is P3 for M3 and M4, P2 for HOG. M1 stays at 1.8 to 4.2% on
Roboflow at every size.

## Open decisions for the team

- Merge PR #1, then rebuild Roboflow once:
  `rm -rf data/work/auslan_roboflow && uv run auslan prepare auslan_roboflow`.
- Re-crop ASL with the new code too? Only 0.7% of ASL images show two hands, so P1 would barely move.
- Change the M3 default to 5 blocks (or a 32 px input)? Either roughly doubles its scores.
- Which run feeds Table 9 in the Task 2 report: Brian's E1 (old crops) or this one?
- S5 still needs example images for the five most-confused pairs per method.

## Files

| File | Content |
|---|---|
| `table9.md`, `success_criteria.md`, `significance.md` | headline results and tests |
| `cross_dataset.md`, `style_effect.md` | E2 (same vs other dataset), E6 (Roboflow with and without the style transform) |
| `confused_pairs.md`, `per_letter_f1.csv`, `confusion_<method>.csv` | E4 |
| `k_curve.csv`, `variants.md`, `image_size.md` | E3, E5, E7 (all filled) |
| `cost.md`, `detection_rates.md` | training time, model size, speed; MediaPipe detection rates |
| `results.csv`, `results.jsonl` | every run and test set; `results.jsonl` rebuilds the tables with `auslan report` |

To reproduce: copy `configs/default.yaml` to `artifacts/v2.yaml`, set `artifacts_dir: artifacts/v2`,
and run `uv run auslan --config artifacts/v2.yaml run all` (about 5 hours on an M-series Mac).
