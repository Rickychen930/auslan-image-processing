Macro-F1 in %, mean ± s.d. over three draws. P3 = P2 where a method cannot transfer, and a dash means the method was not run there (M5 needs detected hands). 95% CI is the bootstrap interval of the draw-averaged macro-F1.

| Method | P1 ASL Test | P2 Roboflow | P3 Roboflow | 95% CI | ΔT(20) | Top-3 | ms/frame |
|---|---|---|---|---|---|---|---|
| m1_skin_svm | 5.0 ± 0.0 | 3.1 ± 0.5 | 3.1 ± 0.5 (=P2) | [1.9, 4.2] | n/a | 10.8 ± 0.4 | 51 |
| m2_hog_svm | 43.9 ± 0.0 | 6.6 ± 1.4 | 6.6 ± 1.4 (=P2) | [4.5, 8.7] | n/a | 21.3 ± 2.7 | 53 |
| m5_landmarks_svm | 83.8 ± 0.0 | – | – | – | n/a | – | – |
| m5_landmarks_mlp | 69.9 ± 0.6 | – | – | – | n/a | – | – |
| m4_mobilenetv2 | 67.8 ± 5.4 | 5.2 ± 0.7 | 6.8 ± 1.5 | [4.5, 8.7] | 1.5 ± 1.5 | 22.9 ± 1.7 | 88 |
| m4_resnet18 | 68.7 ± 1.8 | 8.3 ± 1.9 | 8.0 ± 0.7 | [5.7, 10.0] | -0.3 ± 1.2 | 22.0 ± 0.8 | 56 |
| m3_cnn | 24.1 ± 2.8 | 0.7 ± 0.1 | 1.7 ± 1.1 | [0.9, 2.4] | 1.0 ± 1.2 | 14.4 ± 2.6 | 52 |
