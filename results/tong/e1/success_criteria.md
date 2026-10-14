S1: P1 accuracy on the held-out ASL split

| Method | accuracy % | published % | within 5 points |
|---|---|---|---|
| m4_mobilenetv2 | 99.7 | 99.0 | yes |
| m4_resnet18 | 99.7 | 99.0 | yes |
| m5_landmarks_svm | 78.8 | 87.6 | no |

S2 and S4 must be met by the same method (macro-F1 >= 60%, top-3 >= 85%, >= 10 frames/s).

| Method | macro-F1 % | top-3 % | frames/s | S2 and S4 met |
|---|---|---|---|---|
| m1_skin_svm | 3.3 | 11.2 | 17.3 | no |
| m2_hog_svm | 7.1 | 24.1 | 15.0 | no |
| m4_mobilenetv2 | 9.8 | 24.1 | 17.9 | no |
| m4_resnet18 | 10.5 | 27.3 | 16.7 | no |
| m3_cnn | 1.4 | 14.1 | 19.9 | no |

S3 (ΔT(20) > 0 for MobileNetV2): not met, 95% CI [-2.4, +2.3] points
