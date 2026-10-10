MediaPipe hand detection on the prepared images

| Dataset | images | no hand % | one hand % | two hands % | fallback crop % |
|---|---|---|---|---|---|
| asl_alphabet | 16980 | 21.5 | 77.8 | 0.7 | 21.5 |
| asl_alphabet_test | 704 | 3.7 | 96.3 | 0.0 | 3.7 |
| auslan_kaggle | 11028 | 100.0 | 0.0 | 0.0 | 100.0 |
| auslan_roboflow | 250 | 0.0 | 5.6 | 94.4 | 0.0 |

M1: images in which no hand-sized blob is found

| Test set | empty mask % |
|---|---|
| ASL test | 0.6 |
| Kaggle Auslan test | 0.0 |
| Roboflow (colour) | 0.0 |
