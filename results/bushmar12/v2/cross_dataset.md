P1: ASL split against ASL Alphabet Test. P2/P3 (k=20): Kaggle split against Roboflow.

| Method | Protocol | Same dataset | Other dataset | Drop (points) |
|---|---|---|---|---|
| m1_skin_svm | P1 | 61.1 ± 0.0 | 4.9 ± 0.0 | 56.2 |
| m1_skin_svm | P2 | 34.6 ± 0.8 | 3.3 ± 0.4 | 31.3 |
| m2_hog_svm | P1 | 98.3 ± 0.0 | 44.0 ± 0.0 | 54.3 |
| m2_hog_svm | P2 | 80.1 ± 2.3 | 7.1 ± 1.5 | 72.9 |
| m5_landmarks_svm | P1 | 83.9 ± 0.0 | 83.8 ± 0.0 | 0.1 |
| m5_landmarks_mlp | P1 | 71.4 ± 0.6 | 70.2 ± 0.7 | 1.2 |
| m4_mobilenetv2 | P1 | 99.7 ± 0.1 | 67.9 ± 6.1 | 31.8 |
| m4_mobilenetv2 | P3 | 84.6 ± 2.4 | 9.2 ± 1.1 | 75.5 |
| m4_resnet18 | P1 | 99.7 ± 0.1 | 70.3 ± 1.6 | 29.4 |
| m4_resnet18 | P3 | 85.1 ± 1.9 | 10.7 ± 1.2 | 74.4 |
| m3_cnn | P1 | 95.8 ± 0.3 | 24.0 ± 2.6 | 71.8 |
| m3_cnn | P3 | 38.1 ± 0.5 | 0.9 ± 0.7 | 37.2 |
