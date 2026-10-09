# Brian's test: E1 results (run on 8–9 October 2026)

Headline experiment E1: every core method under P1, P2 and P3, k=20 with seeds 0, 1 and 2, plus
P2/P3 on all Kaggle data with seed 0. Produced by `auslan run E1`, `auslan speed` and
`auslan report` on an Apple M2 Pro (MPS), from commit `39b9613` plus the 24-letter change; later
commits on `main` up to `4ff4b1a` do not touch training or evaluation. E1 took 78 minutes on top
of an earlier 23-minute trial of M2 and M3.

**Setup that differs from the Task 1 report** (details in the top-level README):
- Both languages use the same 24 letters (no J or Z), so chance is 4.2%.
- Roboflow is version 18: 250 test images (7 to 25 per letter).
- MediaPipe finds no hand in any Kaggle Auslan image, so M5 runs under P1 only.

## Table 9: macro-F1 %, mean over three draws, k=20

| Method | P1 ASL Test | P2 Roboflow | P3 Roboflow | ΔT(20) | Top-3 P3 | ms/frame |
|---|---|---|---|---|---|---|
| M1 skin + SVM | 5.0 | 3.1 | = P2 | n/a | 10.8 | 51 |
| M2 HOG + SVM | 43.9 | 6.6 | = P2 | n/a | 21.3 | 53 |
| M3 CNN (no ImageNet) | 24.1 | 0.7 | 1.7 | +1.0 | 14.4 | 52 |
| M4 MobileNetV2 | 67.8 | 5.2 | 6.8 | +1.5 | 22.9 | 88 |
| M4 ResNet-18 | 68.7 | 8.3 | 8.0 | −0.3 | 22.0 | 56 |
| M5 landmarks + SVM | 83.8 | – | – | n/a | – | – |
| M5 landmarks + MLP | 69.9 | – | – | n/a | – | – |

Standard deviations and 95% intervals are in [`table9.md`](table9.md). Speed includes 48 ms of
hand detection per frame.

## Success criteria ([`success_criteria.md`](success_criteria.md))

| Criterion | Result |
|---|---|
| S1 implementation check | met for MobileNetV2 and ResNet-18 (99.7% vs 99% published); not for M5 SVM (78.8% vs 87.6%) |
| S2 Roboflow macro-F1 ≥ 60%, top-3 ≥ 85% | not met: best is ResNet-18 at 8.0%, top-3 22% |
| S3 ΔT(20) > 0 for MobileNetV2 | not met: 95% CI [−0.8, +3.7] points |
| S4 ≥ 10 frames/s | met by every method (11 to 20 frames/s) |

No pair of methods differs significantly on Roboflow ([`significance.md`](significance.md)).

## Findings

1. **Landmarks survive the change of dataset (supports H1).** Under P1, M5 SVM scores 83.9% on
   the ASL split and 83.8% on ASL Alphabet Test. MobileNetV2 drops 31.9 points and HOG 54.4
   ([`cross_dataset.md`](cross_dataset.md)).
2. **ASL training helps the CNN trained from scratch, not the ImageNet models.** On the Kaggle
   test split (same source as training):

   | Method | k | P2 | P3 | ΔT |
   |---|---|---|---|---|
   | M3 CNN | 20 | 26.1 | 37.6 | +11.5 (positive in all three draws) |
   | M3 CNN | all | 53.5 | 83.3 | +29.7 (one draw) |
   | M4 MobileNetV2 | 20 | 85.3 | 84.4 | −0.8 |
   | M4 ResNet-18 | 20 | 87.2 | 85.8 | −1.4 |

   M4 already starts from ImageNet weights under P2, so ASL adds little. The pre-specified S3
   test (MobileNetV2) therefore looks at the method least likely to gain.
3. **Nothing generalises from Kaggle Auslan to Roboflow.** Methods that reach 80–86% on the
   Kaggle split fall to 1–8% on Roboflow, and their predictions collapse onto 2 to 4 letters.
   Two causes were tested and ruled out on the M2/M3 trial:
   - Framing: feeding Roboflow as full frames, like Kaggle, did not help.
   - Mirroring: flipping Roboflow left-right did not help.

   The Kaggle images come from a few recording sessions (different clothes and rooms), with the
   hands close to the camera and a binarised style. Signers cannot be identified. This points
   to a domain gap between the datasets rather than a pipeline bug.

## Files

| File | Content |
|---|---|
| `table9.md`, `success_criteria.md`, `significance.md` | headline results and tests |
| `cross_dataset.md`, `style_effect.md` | E2 (same vs other dataset), E6 (Roboflow with and without the style transform) |
| `confused_pairs.md`, `per_letter_f1.csv`, `confusion_<method>.csv` | E4 |
| `cost.md`, `detection_rates.md` | training time, model size, speed; MediaPipe detection rates |
| `results.csv`, `results.jsonl` | every run and test set; `results.jsonl` rebuilds the tables with `auslan report` |
| `k_curve.csv`, `image_size.md`, `variants.md` | only partly filled: E3, E7 and E5 have not been run yet |
