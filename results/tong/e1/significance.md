Main k=20 runs on Roboflow. McNemar (exact) is Holm-adjusted across method pairs within each draw; the largest adjusted p over the draws is shown, so a difference counts only if it holds in all of them. The interval is the paired bootstrap of the macro-F1 difference (A minus B) averaged over draws; it reflects test-image uncertainty only.

| A | B | max adjusted p | ΔF1 95% CI (points) | verdict |
|---|---|---|---|---|
| m1_skin_svm | m2_hog_svm | 0.7397 | [-6.3, -1.2] | no clear difference |
| m1_skin_svm | m4_mobilenetv2 | 0.6973 | [-8.7, -4.0] | no clear difference |
| m1_skin_svm | m4_resnet18 | 0.3213 | [-9.6, -4.5] | no clear difference |
| m1_skin_svm | m3_cnn | 1.0000 | [+0.5, +3.3] | no clear difference |
| m2_hog_svm | m4_mobilenetv2 | 1.0000 | [-5.3, -0.3] | no clear difference |
| m2_hog_svm | m4_resnet18 | 1.0000 | [-6.1, -0.6] | no clear difference |
| m2_hog_svm | m3_cnn | 0.6973 | [+3.4, +7.8] | no clear difference |
| m4_mobilenetv2 | m4_resnet18 | 1.0000 | [-3.1, +1.8] | no clear difference |
| m4_mobilenetv2 | m3_cnn | 1.0000 | [+6.2, +10.1] | no clear difference |
| m4_resnet18 | m3_cnn | 1.0000 | [+6.6, +11.0] | no clear difference |

### Gain from ASL training, ΔT(20), 95% bootstrap interval

- m4_mobilenetv2: [-2.4, +2.3] points (pre-specified test)
- m4_resnet18: [-0.8, +4.5] points (exploratory)
- m3_cnn: [-0.0, +1.3] points (exploratory)
