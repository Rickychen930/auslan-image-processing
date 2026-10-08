# Related repositories

Surveyed on 2026-10-08. Repositories were read, never executed. Where a repo has no licence,
default copyright applies, so we took ideas only and wrote our own code. No repository does
cross-dataset or ASL-to-Auslan transfer in the P2/P3 style, and none is a maintained, permissively
licensed Auslan image classifier, so the protocols here are new; the components are not.

| Repository | Licence | Relevant to | What we took | What we avoided |
|------------|---------|-------------|--------------|-----------------|
| [shag527/Indian-Sign-Language-Recognition](https://github.com/shag527/Indian-Sign-Language-Recognition) | none | M1, M2 | HSV mask + morphology, several classifiers on one feature table | hand-tuned HSV trackbars, refitting the label encoder on test labels, SURF (needs non-free OpenCV) |
| [Abhi-Bhatnagar/AUSLAN_Converter](https://github.com/Abhi-Bhatnagar/AUSLAN_Converter) | none | M4, M5 | wrist-centred landmark vector scaled by wrist-to-middle-knuckle, zero-fill for a missing hand, no horizontal flip | early stopping on the test set, legacy `mp.solutions` API |
| [wang123c/Sign-Language-Recognition](https://github.com/wang123c/Sign-Language-Recognition) | GPL-3.0 | M3, M4 | three-stage split / cache / train layout, confusion-matrix reporting | image-level random split of a few-signer dataset (leakage); GPL code was not mixed in |
| [mxagar/asl_alphabet_image_classification](https://github.com/mxagar/asl_alphabet_image_classification) | none | M3, M4 | folder layout of the Kaggle ASL set | its 99.8% comes from a random split of near-duplicate frames |
| [mlradak/serbian-sign-language](https://github.com/mlradak/serbian-sign-language) | none | M5 | idea of caching landmarks instead of images | dynamic video data, no training code |
| [CMU-Perceptual-Computing-Lab/openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) | non-commercial | M5 | background on the 21-point hand layout | heavy C++ build; MediaPipe is used instead |
| [AlbertaBeef/asl_mediapipe_pointnet](https://github.com/AlbertaBeef/asl_mediapipe_pointnet) | Apache-2.0 | M5 | clean split between landmark extraction and the classifier | per-hand min-max scaling loses depth and size |
| [vic-chan67/real-time-bsl-fingerspelling-recognition](https://github.com/vic-chan67/real-time-bsl-fingerspelling-recognition) | MIT | M5 | two-hand feature extractor design | ordering hands by handedness label, which flips with mirrored images |
| [Andrei-AlexandruRusu/Hand-Gesture-HOG-SVM-KNN](https://github.com/Andrei-AlexandruRusu/Hand-Gesture-HOG-SVM-KNN) | none | M2 | HOG parameters (9 bins, 8 px cells, L2-Hys) | 34,000-dimensional features without scaling or PCA |
| [QuantumAlchemist03/BSL-Fingerspelling-Recognizer](https://github.com/QuantumAlchemist03/BSL-Fingerspelling-Recognizer) | all rights reserved | M4 | MobileNetV2 head layout as an idea | `1/255` rescaling for MobileNetV2 weights, flipped images on a handed alphabet |
| [HealthHackAu2018/auslan-party](https://github.com/HealthHackAu2018/auslan-party) and [auslan-alphabet-images](https://github.com/HealthHackAu2018/auslan-alphabet-images) | MIT | data | real Auslan annotation format, 26 letters | images are not shipped; frames from few videos |

## Practical consequences for this codebase

- Splits and de-duplication come first, because every high score found in these repos came from
  random splits of near-identical frames.
- ImageNet models use ImageNet normalisation (`torch_base.IMAGENET_MEAN/STD`), not `1/255`.
- No horizontal flips anywhere; hands are ordered by wrist x position (`HandCropper`).
- MediaPipe Tasks `HandLandmarker` is used, not the removed `mp.solutions` API.
- Detection rate is recorded per image (`n_hands`, `used_fallback`) instead of silently dropping
  images MediaPipe cannot read.
