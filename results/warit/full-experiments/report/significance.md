Main k=20 runs on Roboflow. McNemar (exact) is Holm-adjusted across method pairs within each draw; the largest adjusted p over the draws is shown, so a difference counts only if it holds in all of them. The interval is the paired bootstrap of the macro-F1 difference (A minus B) averaged over draws; it reflects test-image uncertainty only.

| A | B | max adjusted p | ΔF1 95% CI (points) | verdict |
|---|---|---|---|---|
| m1_skin_svm | m2_hog_svm | 0.7693 | [-6.3, -1.2] | no clear difference |
| m1_skin_svm | m4_mobilenetv2 | 0.1082 | [-8.7, -4.3] | no clear difference |
| m1_skin_svm | m4_resnet18 | 0.0539 | [-9.8, -4.8] | no clear difference |
| m1_skin_svm | m3_cnn | 1.0000 | [+0.0, +3.0] | no clear difference |
| m2_hog_svm | m4_mobilenetv2 | 1.0000 | [-5.4, -0.0] | no clear difference |
| m2_hog_svm | m4_resnet18 | 1.0000 | [-6.3, -0.8] | no clear difference |
| m2_hog_svm | m3_cnn | 0.8601 | [+3.0, +7.5] | no clear difference |
| m4_mobilenetv2 | m4_resnet18 | 1.0000 | [-2.8, +1.4] | no clear difference |
| m4_mobilenetv2 | m3_cnn | 0.4477 | [+6.0, +9.9] | no clear difference |
| m4_resnet18 | m3_cnn | 0.3292 | [+6.7, +10.7] | no clear difference |

### Gain from ASL training, ΔT(20), 95% bootstrap interval

- m4_mobilenetv2: [-0.7, +3.8] points (pre-specified test)
- m4_resnet18: [-0.1, +4.9] points (exploratory)
- m3_cnn: [+0.3, +1.8] points (exploratory)
