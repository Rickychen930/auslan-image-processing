Main k=20 runs on Roboflow. McNemar (exact) is Holm-adjusted across method pairs within each draw; the largest adjusted p over the draws is shown, so a difference counts only if it holds in all of them. The interval is the paired bootstrap of the macro-F1 difference (A minus B) averaged over draws; it reflects test-image uncertainty only.

| A | B | max adjusted p | ΔF1 95% CI (points) | verdict |
|---|---|---|---|---|
| m1_skin_svm | m2_hog_svm | 0.7579 | [-6.0, -1.1] | no clear difference |
| m1_skin_svm | m4_mobilenetv2 | 0.8163 | [-6.0, -1.1] | no clear difference |
| m1_skin_svm | m4_resnet18 | 0.2027 | [-7.0, -2.4] | no clear difference |
| m1_skin_svm | m3_cnn | 1.0000 | [+0.1, +2.7] | no clear difference |
| m2_hog_svm | m4_mobilenetv2 | 1.0000 | [-2.9, +2.9] | no clear difference |
| m2_hog_svm | m4_resnet18 | 1.0000 | [-3.5, +1.3] | no clear difference |
| m2_hog_svm | m3_cnn | 0.6921 | [+2.7, +7.2] | no clear difference |
| m4_mobilenetv2 | m4_resnet18 | 1.0000 | [-3.7, +1.2] | no clear difference |
| m4_mobilenetv2 | m3_cnn | 1.0000 | [+2.7, +7.2] | no clear difference |
| m4_resnet18 | m3_cnn | 0.5660 | [+4.0, +8.5] | no clear difference |

### Gain from ASL training, ΔT(20), 95% bootstrap interval

- m4_mobilenetv2: [-0.8, +3.7] points (pre-specified test)
- m4_resnet18: [-2.8, +2.2] points (exploratory)
- m3_cnn: [+0.1, +1.8] points (exploratory)
