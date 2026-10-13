P1: ASL split against ASL Alphabet Test. P2/P3 (k=20): Kaggle split against Roboflow.

| Method | Protocol | Same dataset | Other dataset | Drop (points) |
|---|---|---|---|---|
| m1_skin_svm | P1 | 60.8 ± 0.0 | 5.1 ± 0.0 | 55.7 |
| m1_skin_svm | P2 | 34.6 ± 0.8 | 3.3 ± 0.4 | 31.3 |
| m2_hog_svm | P1 | 98.3 ± 0.0 | 43.9 ± 0.0 | 54.4 |
| m2_hog_svm | P2 | 80.1 ± 2.3 | 7.1 ± 1.5 | 72.9 |
| m5_landmarks_svm | P1 | 83.9 ± 0.0 | 83.7 ± 0.0 | 0.3 |
| m5_landmarks_mlp | P1 | 71.5 ± 0.6 | 70.2 ± 0.2 | 1.3 |
| m4_mobilenetv2 | P1 | 99.7 ± 0.1 | 67.1 ± 4.8 | 32.6 |
| m4_mobilenetv2 | P3 | 82.9 ± 2.4 | 7.3 ± 0.1 | 75.6 |
| m4_resnet18 | P1 | 99.7 ± 0.1 | 69.3 ± 4.6 | 30.4 |
| m4_resnet18 | P3 | 85.9 ± 1.6 | 9.9 ± 3.2 | 76.0 |
| m3_cnn | P1 | 95.9 ± 0.8 | 24.6 ± 5.8 | 71.2 |
| m3_cnn | P3 | 36.8 ± 1.5 | 1.4 ± 0.7 | 35.4 |
