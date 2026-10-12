P1: ASL split against ASL Alphabet Test. P2/P3 (k=20): Kaggle split against Roboflow.

| Method | Protocol | Same dataset | Other dataset | Drop (points) |
|---|---|---|---|---|
| m1_skin_svm | P1 | 60.8 ± 0.0 | 5.1 ± 0.0 | 55.7 |
| m1_skin_svm | P2 | 34.7 ± 0.8 | 3.3 ± 0.4 | 31.3 |
| m2_hog_svm | P1 | 98.3 ± 0.0 | 43.9 ± 0.0 | 54.4 |
| m2_hog_svm | P2 | 80.1 ± 2.3 | 7.1 ± 1.5 | 72.9 |
| m5_landmarks_svm | P1 | 83.9 ± 0.0 | 83.7 ± 0.0 | 0.3 |
| m5_landmarks_mlp | P1 | 71.3 ± 0.4 | 70.5 ± 0.4 | 0.8 |
| m4_mobilenetv2 | P1 | 99.7 ± 0.1 | 69.9 ± 3.8 | 29.8 |
| m4_mobilenetv2 | P3 | 84.0 ± 2.3 | 10.0 ± 1.2 | 74.1 |
| m4_resnet18 | P1 | 99.7 ± 0.1 | 71.5 ± 3.4 | 28.2 |
| m4_resnet18 | P3 | 85.5 ± 1.7 | 10.7 ± 2.5 | 74.9 |
| m3_cnn | P1 | 96.0 ± 0.3 | 22.6 ± 3.6 | 73.4 |
| m3_cnn | P3 | 34.8 ± 1.0 | 1.8 ± 1.0 | 33.0 |
