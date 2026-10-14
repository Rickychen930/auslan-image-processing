CPU timings; 'classify' includes decoding the cached crop and computing features.

| Method | train s | model MB | classify ms | detect+crop ms | frames/s |
|---|---|---|---|---|---|
| m1_skin_svm | 7 | 0.2 | 17.1 | 40.7 | 17.3 |
| m2_hog_svm | 14 | 4.8 | 26.2 | 40.7 | 15.0 |
| m5_landmarks_svm | 37 | 6.9 | 16.2 | 40.7 | 17.6 |
| m5_landmarks_mlp | 6 | 0.3 | 1.5 | 40.7 | 23.7 |
| m4_mobilenetv2 | 36 | 9.3 | 15.3 | 40.7 | 17.9 |
| m4_resnet18 | 36 | 44.8 | 19.4 | 40.7 | 16.7 |
| m3_cnn | 34 | 1.6 | 9.5 | 40.7 | 19.9 |
