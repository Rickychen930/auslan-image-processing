Macro-F1 in %, mean ± s.d. over three draws. P3 = P2 where a method cannot transfer, and a dash means the method was not run there (M5 needs detected hands). 95% CI is the bootstrap interval of the draw-averaged macro-F1.

| Method | P1 ASL Test | P2 Roboflow | P3 Roboflow | 95% CI | ΔT(20) | Top-3 | ms/frame |
|---|---|---|---|---|---|---|---|
| m1_skin_svm | 5.1 ± 0.0 | 3.3 ± 0.4 | 3.3 ± 0.4 (=P2) | [2.1, 4.6] | n/a | 11.2 ± 1.4 | 44 |
| m2_hog_svm | 43.9 ± 0.0 | 7.1 ± 1.5 | 7.1 ± 1.5 (=P2) | [4.8, 9.2] | n/a | 24.1 ± 0.8 | 49 |
| m5_landmarks_svm | 83.7 ± 0.0 | – | – | – | n/a | – | – |
| m5_landmarks_mlp | 70.5 ± 0.4 | – | – | – | n/a | – | – |
| m4_mobilenetv2 | 69.9 ± 3.8 | 8.4 ± 1.2 | 10.0 ± 1.2 | [7.7, 11.8] | 1.6 ± 2.3 | 25.7 ± 3.5 | 53 |
| m4_resnet18 | 71.5 ± 3.4 | 8.3 ± 1.3 | 10.7 ± 2.5 | [8.5, 12.7] | 2.4 ± 3.8 | 27.6 ± 2.1 | 51 |
| m3_cnn | 22.6 ± 3.6 | 0.8 ± 0.5 | 1.8 ± 1.0 | [1.1, 2.6] | 1.0 ± 1.5 | 15.9 ± 3.8 | 45 |
