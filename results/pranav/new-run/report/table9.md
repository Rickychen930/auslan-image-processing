Macro-F1 in %, mean ± s.d. over three draws. P3 = P2 where a method cannot transfer, and a dash means the method was not run there (M5 needs detected hands). 95% CI is the bootstrap interval of the draw-averaged macro-F1.

| Method | P1 ASL Test | P2 Roboflow | P3 Roboflow | 95% CI | ΔT(20) | Top-3 | ms/frame |
|---|---|---|---|---|---|---|---|
| m1_skin_svm | 5.1 ± 0.0 | 3.3 ± 0.4 | 3.3 ± 0.4 (=P2) | [2.0, 4.6] | n/a | 11.2 ± 1.4 | 53 |
| m2_hog_svm | 43.9 ± 0.0 | 7.1 ± 1.5 | 7.1 ± 1.5 (=P2) | [4.9, 9.2] | n/a | 24.1 ± 0.8 | 67 |
| m5_landmarks_svm | 83.7 ± 0.0 | – | – | – | n/a | – | – |
| m5_landmarks_mlp | 70.2 ± 0.2 | – | – | – | n/a | – | – |
| m4_mobilenetv2 | 67.1 ± 4.8 | 7.9 ± 1.4 | 7.3 ± 0.1 | [5.3, 9.1] | -0.6 ± 1.5 | 22.3 ± 1.0 | 57 |
| m4_resnet18 | 69.3 ± 4.6 | 8.0 ± 2.1 | 9.9 ± 3.2 | [7.4, 12.1] | 2.0 ± 5.2 | 27.1 ± 3.4 | 57 |
| m3_cnn | 24.6 ± 5.8 | 1.0 ± 1.0 | 1.4 ± 0.7 | [0.8, 2.1] | 0.4 ± 1.5 | 13.3 ± 0.6 | 52 |
