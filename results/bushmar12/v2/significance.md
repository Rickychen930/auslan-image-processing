Main k=20 runs on Roboflow. McNemar (exact) is Holm-adjusted across method pairs within each draw; the largest adjusted p over the draws is shown, so a difference counts only if it holds in all of them. The interval is the paired bootstrap of the macro-F1 difference (A minus B) averaged over draws; it reflects test-image uncertainty only.

| A | B | max adjusted p | ΔF1 95% CI (points) | verdict |
|---|---|---|---|---|
| m1_skin_svm | m2_hog_svm | 0.6072 | [-6.3, -1.3] | no clear difference |
| m1_skin_svm | m4_mobilenetv2 | 0.0723 | [-7.9, -3.4] | no clear difference |
| m1_skin_svm | m4_resnet18 | 0.0166 | [-9.6, -4.6] | m4_resnet18 better |
| m1_skin_svm | m3_cnn | 1.0000 | [+1.1, +3.6] | no clear difference |
| m2_hog_svm | m4_mobilenetv2 | 1.0000 | [-4.5, +0.4] | no clear difference |
| m2_hog_svm | m4_resnet18 | 1.0000 | [-6.2, -0.7] | no clear difference |
| m2_hog_svm | m3_cnn | 0.5306 | [+3.9, +8.3] | no clear difference |
| m4_mobilenetv2 | m4_resnet18 | 1.0000 | [-3.8, +1.1] | no clear difference |
| m4_mobilenetv2 | m3_cnn | 0.5864 | [+5.9, +10.1] | no clear difference |
| m4_resnet18 | m3_cnn | 0.1405 | [+7.2, +11.9] | no clear difference |

### Gain from ASL training, ΔT(20), 95% bootstrap interval

- m4_mobilenetv2: [+0.5, +5.0] points (pre-specified test)
- m4_resnet18: [-1.2, +4.1] points (exploratory)
- m3_cnn: [-0.1, +0.7] points (exploratory)
