P1: ASL split against ASL Alphabet Test. P2/P3 (k=20): Kaggle split against Roboflow.

| Method | Protocol | Same dataset | Other dataset | Drop (points) |
|---|---|---|---|---|
| m1_skin_svm | P1 | 60.9 ± 0.0 | 5.0 ± 0.0 | 55.8 |
| m1_skin_svm | P2 | 34.7 ± 0.8 | 3.3 ± 0.4 | 31.3 |
| m2_hog_svm | P1 | 98.3 ± 0.0 | 44.0 ± 0.0 | 54.3 |
| m2_hog_svm | P2 | 80.1 ± 2.3 | 7.1 ± 1.5 | 72.9 |
| m5_landmarks_svm | P1 | 83.9 ± 0.0 | 83.8 ± 0.0 | 0.1 |
| m5_landmarks_mlp | P1 | 71.5 ± 0.8 | 70.2 ± 0.4 | 1.3 |
| m4_mobilenetv2 | P1 | 99.7 ± 0.0 | 69.7 ± 4.6 | 30.1 |
| m4_mobilenetv2 | P3 | 84.3 ± 3.0 | 9.8 ± 3.1 | 74.5 |
| m4_resnet18 | P1 | 99.7 ± 0.1 | 69.8 ± 3.1 | 29.9 |
| m4_resnet18 | P3 | 85.5 ± 1.5 | 10.5 ± 2.9 | 75.0 |
| m3_cnn | P1 | 96.0 ± 0.7 | 25.1 ± 5.7 | 70.8 |
| m3_cnn | P3 | 37.0 ± 1.5 | 1.4 ± 0.7 | 35.6 |
