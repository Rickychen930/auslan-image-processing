S1: P1 accuracy on the held-out ASL split

| Method | accuracy % | published % | within 5 points |
|---|---|---|---|
| m4_mobilenetv2 | 99.7 | 99.0 | yes |
| m4_resnet18 | 99.7 | 99.0 | yes |
| m5_landmarks_svm | 78.8 | 87.6 | no |

S2 and S4 must be met by the same method (macro-F1 >= 60%, top-3 >= 85%, >= 10 frames/s).

| Method | macro-F1 % | top-3 % | frames/s | S2 and S4 met |
|---|---|---|---|---|
| m1_skin_svm | 3.3 | 11.2 | 71.1 | no |
| m2_hog_svm | 7.1 | 24.0 | 66.0 | no |
| m4_mobilenetv2 | 9.2 | 25.9 | 35.6 | no |
| m4_resnet18 | 10.7 | 24.9 | 60.9 | no |
| m3_cnn | 0.9 | 14.7 | 71.6 | no |

S3 (ΔT(20) > 0 for MobileNetV2): met, 95% CI [+0.5, +5.0] points
