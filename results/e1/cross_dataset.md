P1: ASL split against ASL Alphabet Test. P2/P3 (k=20): Kaggle split against Roboflow.

| Method | Protocol | Same dataset | Other dataset | Drop (points) |
|---|---|---|---|---|
| m1_skin_svm | P1 | 61.0 ± 0.0 | 5.0 ± 0.0 | 56.0 |
| m1_skin_svm | P2 | 34.6 ± 0.8 | 3.1 ± 0.5 | 31.6 |
| m2_hog_svm | P1 | 98.3 ± 0.0 | 43.9 ± 0.0 | 54.4 |
| m2_hog_svm | P2 | 80.1 ± 2.3 | 6.6 ± 1.4 | 73.4 |
| m5_landmarks_svm | P1 | 83.9 ± 0.0 | 83.8 ± 0.0 | 0.1 |
| m5_landmarks_mlp | P1 | 71.3 ± 0.6 | 69.9 ± 0.6 | 1.3 |
| m4_mobilenetv2 | P1 | 99.7 ± 0.0 | 67.8 ± 5.4 | 31.9 |
| m4_mobilenetv2 | P3 | 84.4 ± 1.8 | 6.8 ± 1.5 | 77.7 |
| m4_resnet18 | P1 | 99.7 ± 0.0 | 68.7 ± 1.8 | 31.0 |
| m4_resnet18 | P3 | 85.8 ± 1.8 | 8.0 ± 0.7 | 77.8 |
| m3_cnn | P1 | 95.8 ± 0.3 | 24.1 ± 2.8 | 71.7 |
| m3_cnn | P3 | 37.6 ± 1.5 | 1.7 ± 1.1 | 35.9 |
