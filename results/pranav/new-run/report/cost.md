CPU timings; 'classify' includes decoding the cached crop and computing features.

| Method | train s | model MB | classify ms | detect+crop ms | frames/s |
|---|---|---|---|---|---|
| m1_skin_svm | 1 | 0.2 | 7.9 | 44.7 | 19.0 |
| m2_hog_svm | 7 | 4.8 | 22.5 | 44.7 | 14.9 |
| m5_landmarks_svm | 29 | 6.8 | 6.5 | 44.7 | 19.5 |
| m5_landmarks_mlp | 2 | 0.3 | 0.9 | 44.7 | 22.0 |
| m4_mobilenetv2 | 29 | 9.3 | 12.8 | 44.7 | 17.4 |
| m4_resnet18 | 29 | 44.8 | 12.0 | 44.7 | 17.6 |
| m3_cnn | 29 | 1.6 | 7.2 | 44.7 | 19.3 |
