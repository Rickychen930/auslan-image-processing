# Warit — Auslan Experiment Results

## Overview

This folder contains my experiment results for the Auslan Fingerspelling Recognition project (UTS 42177, Group 9).

I ran the experiments locally on Windows using the shared project pipeline. The results cover the main model comparisons, few-shot learning, design alternatives, and image resolution experiments.

## Experiments Completed

| Experiment | Description | Status |
|---|---|---|
| E1 | Main model comparison across P1, P2, and P3 | Completed |
| E2 | Same-dataset vs. cross-dataset evaluation | Generated from E1 results |
| E3 | Data efficiency with 5, 10, 20, 50, and all training samples per letter | Completed |
| E4 | Per-letter F1 and confusion analysis | Generated from saved predictions |
| E5 | Design alternatives and ablation studies | Completed |
| E6 | Image style transformation comparison | Generated from P2/P3 evaluations |
| E7 | Image resolution comparison | Completed |

Some runs were skipped automatically by the pipeline because the corresponding methods or protocols were not applicable.

## Files

```text
results/warit/
├── README.md
├── results.jsonl
├── crop_ms.json
└── report/
    ├── index.html
    ├── results.csv
    ├── table9.md
    ├── cross_dataset.md
    ├── k_curve.csv
    ├── per_letter_f1.csv
    ├── confused_pairs.md
    ├── significance.md
    ├── variants.md
    ├── style_effect.md
    ├── image_size.md
    ├── cost.md
    ├── detection_rates.md
    ├── success_criteria.md
    └── confusion_*.csv
```

- `results.jsonl` contains the recorded experiment metrics.
- `crop_ms.json` contains the hand detection and cropping benchmark.
- `report/index.html` is the main HTML report with the experiment results.
- Other files in `report/` contain the detailed tables and evaluation results.

Model checkpoints and individual prediction files are not included in this folder.

## Key Findings

- ResNet-18 achieved the highest Roboflow macro-F1 in the main k=20 comparison, with 10.7% under P3.
- The models performed much better on the held-out Kaggle Auslan split than on the independent Roboflow dataset, indicating a substantial cross-dataset generalisation gap.
- ASL pretraining produced small improvements on Roboflow for the neural models, but the MobileNetV2 transfer gain was not statistically conclusive.
- Increasing the number of Auslan training examples improved ResNet-18 performance, but did not resolve the generalisation problem.
- All benchmarked models exceeded the target of 10 FPS on CPU, while the main Roboflow accuracy targets were not met.

## How to View the Results

Open [`report/index.html`](report/index.html) in a web browser to view the generated results.

The report includes model comparisons, learning curves, statistical tests, confusion analysis, and computational cost.

## Notes

- Experiments were run locally on Windows.
- Results were generated using the shared repository code.
- The Roboflow dataset was used as an independent test set.
- The results are intended for comparison, analysis, and preparation of the Task 2 report.