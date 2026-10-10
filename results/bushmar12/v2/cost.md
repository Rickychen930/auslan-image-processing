CPU timings; 'classify' includes decoding the cached crop and computing features.

| Method | train s | model MB | classify ms | detect+crop ms | frames/s |
|---|---|---|---|---|---|
| m1_skin_svm | 1 | 0.2 | 1.9 | 12.1 | 71.1 |
| m2_hog_svm | 2 | 4.8 | 3.0 | 12.1 | 66.0 |
| m5_landmarks_svm | 7 | 6.9 | 1.6 | 12.1 | 72.8 |
| m5_landmarks_mlp | 1 | 0.3 | 0.1 | 12.1 | 81.7 |
| m4_mobilenetv2 | 36 | 9.3 | 15.9 | 12.1 | 35.6 |
| m4_resnet18 | 38 | 44.8 | 4.3 | 12.1 | 60.9 |
| m3_cnn | 21 | 1.6 | 1.8 | 12.1 | 71.6 |
