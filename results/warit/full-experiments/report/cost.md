CPU timings; 'classify' includes decoding the cached crop and computing features.

| Method | train s | model MB | classify ms | detect+crop ms | frames/s |
|---|---|---|---|---|---|
| m1_skin_svm | 1 | 0.2 | 6.6 | 37.7 | 22.6 |
| m2_hog_svm | 5 | 4.8 | 11.5 | 37.7 | 20.3 |
| m5_landmarks_svm | 32 | 6.8 | 10.7 | 37.7 | 20.7 |
| m5_landmarks_mlp | 5 | 0.3 | 1.2 | 37.7 | 25.7 |
| m4_mobilenetv2 | 523 | 9.3 | 15.4 | 37.7 | 18.8 |
| m4_resnet18 | 411 | 44.8 | 13.5 | 37.7 | 19.5 |
| m3_cnn | 307 | 1.6 | 6.8 | 37.7 | 22.5 |
