CPU timings; 'classify' includes decoding the cached crop and computing features.

| Method | train s | model MB | classify ms | detect+crop ms | frames/s |
|---|---|---|---|---|---|
| m1_skin_svm | 1 | 0.2 | 2.8 | 48.1 | 19.6 |
| m2_hog_svm | 2 | 4.8 | 4.4 | 48.1 | 19.0 |
| m5_landmarks_svm | 12 | 6.9 | 2.5 | 48.1 | 19.7 |
| m5_landmarks_mlp | 2 | 0.3 | 0.2 | 48.1 | 20.7 |
| m4_mobilenetv2 | 39 | 9.3 | 40.3 | 48.1 | 11.3 |
| m4_resnet18 | 33 | 44.8 | 7.6 | 48.1 | 18.0 |
| m3_cnn | 40 | 1.6 | 3.5 | 48.1 | 19.4 |
