Macro-F1 in %, mean ± s.d. over three draws. P3 = P2 where a method cannot transfer, and a dash means the method was not run there (M5 needs detected hands). 95% CI is the bootstrap interval of the draw-averaged macro-F1.

| Method | P1 ASL Test | P2 Roboflow | P3 Roboflow | 95% CI | ΔT(20) | Top-3 | ms/frame |
|---|---|---|---|---|---|---|---|
| m1_skin_svm | 5.0 ± 0.0 | 3.3 ± 0.4 | 3.3 ± 0.4 (=P2) | [2.1, 4.6] | n/a | 11.2 ± 1.4 | 58 |
| m2_hog_svm | 44.0 ± 0.0 | 7.1 ± 1.5 | 7.1 ± 1.5 (=P2) | [4.8, 9.2] | n/a | 24.1 ± 0.8 | 67 |
| m5_landmarks_svm | 83.8 ± 0.0 | – | – | – | n/a | – | – |
| m5_landmarks_mlp | 70.2 ± 0.4 | – | – | – | n/a | – | – |
| m4_mobilenetv2 | 69.7 ± 4.6 | 9.8 ± 2.1 | 9.8 ± 3.1 | [7.6, 11.5] | 0.0 ± 1.2 | 24.1 ± 2.9 | 56 |
| m4_resnet18 | 69.8 ± 3.1 | 8.4 ± 3.6 | 10.5 ± 2.9 | [8.1, 12.5] | 2.0 ± 6.2 | 27.3 ± 4.6 | 60 |
| m3_cnn | 25.1 ± 5.7 | 0.8 ± 0.7 | 1.4 ± 0.7 | [0.9, 2.1] | 0.6 ± 1.1 | 14.1 ± 1.8 | 50 |
