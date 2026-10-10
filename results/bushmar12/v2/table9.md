Macro-F1 in %, mean ± s.d. over three draws. P3 = P2 where a method cannot transfer, and a dash means the method was not run there (M5 needs detected hands). 95% CI is the bootstrap interval of the draw-averaged macro-F1.

| Method | P1 ASL Test | P2 Roboflow | P3 Roboflow | 95% CI | ΔT(20) | Top-3 | ms/frame |
|---|---|---|---|---|---|---|---|
| m1_skin_svm | 4.9 ± 0.0 | 3.3 ± 0.4 | 3.3 ± 0.4 (=P2) | [2.0, 4.6] | n/a | 11.2 ± 1.4 | 14 |
| m2_hog_svm | 44.0 ± 0.0 | 7.1 ± 1.5 | 7.1 ± 1.5 (=P2) | [4.9, 9.2] | n/a | 24.0 ± 1.1 | 15 |
| m5_landmarks_svm | 83.8 ± 0.0 | – | – | – | n/a | – | – |
| m5_landmarks_mlp | 70.2 ± 0.7 | – | – | – | n/a | – | – |
| m4_mobilenetv2 | 67.9 ± 6.1 | 6.3 ± 2.6 | 9.2 ± 1.1 | [6.9, 10.9] | 2.8 ± 2.3 | 25.9 ± 1.2 | 28 |
| m4_resnet18 | 70.3 ± 1.6 | 9.2 ± 0.9 | 10.7 ± 1.2 | [8.1, 12.7] | 1.5 ± 1.7 | 24.9 ± 1.9 | 16 |
| m3_cnn | 24.0 ± 2.6 | 0.6 ± 0.2 | 0.9 ± 0.7 | [0.7, 1.3] | 0.3 ± 0.7 | 14.7 ± 2.0 | 14 |
