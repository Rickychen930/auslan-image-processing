S1: P1 accuracy on the held-out ASL split

| Method | accuracy % | published % | within 5 points |
|---|---|---|---|
| m4_mobilenetv2 | 99.7 | 99.0 | yes |
| m4_resnet18 | 99.7 | 99.0 | yes |
| m5_landmarks_svm | 78.8 | 87.6 | no |

S2 and S4 must be met by the same method (macro-F1 >= 60%, top-3 >= 85%, >= 10 frames/s).

| Method | macro-F1 % | top-3 % | frames/s | S2 and S4 met |
|---|---|---|---|---|
| m1_skin_svm | 3.1 | 10.8 | 19.6 | no |
| m2_hog_svm | 6.6 | 21.3 | 19.0 | no |
| m4_mobilenetv2 | 6.8 | 22.9 | 11.3 | no |
| m4_resnet18 | 8.0 | 22.0 | 18.0 | no |
| m3_cnn | 1.7 | 14.4 | 19.4 | no |

S3 (ΔT(20) > 0 for MobileNetV2): not met, 95% CI [-0.8, +3.7] points
