# Roboflow crop fix: before and after (10 October 2026)

Evidence for the `fix/roboflow-signer-crop` change. Only the Roboflow crops were rebuilt
(`auslan prepare auslan_roboflow`); ASL and Kaggle Auslan crops are unchanged.

## What was wrong

MediaPipe finds 3 or 4 hands in 115 of the 250 Roboflow images (46%): two signers side by side,
or a close-up inset next to the full-body signer. With a 2-hand limit, the crop covered the first
two hands returned, often one from each person, plus the inset box and its letter caption.

## Crops

- [`crop_fix_same_letters.jpg`](crop_fix_same_letters.jpg): the letters A, B, C, L, M, Y. C and M
  were wrong before (inset + caption; second signer) and now show one signer.
- [`crop_fix_more_examples.jpg`](crop_fix_more_examples.jpg): six random crops that changed. All
  two-signer frames now show one person; the inset frame (G) is now a close-up of the hands.

Columns: Kaggle model input | Roboflow original | old crop | new crop | old and new model input
(crop + style transform T). 90 of 250 crops changed.

## Scores

Re-run of E1 for HOG + SVM and ResNet-18 under P2/P3 with the new crops. Training is identical
(Kaggle test scores match E3 exactly), so only the Roboflow crops differ. Roboflow macro-F1 in %,
with T; k = 20 is the mean of three draws, "all" is one draw.

| Method | Setup | Old crops | New crops | Change |
|---|---|---|---|---|
| HOG + SVM | P2, k = 20 | 6.6 | 7.1 | +0.5 |
| HOG + SVM | P2, all | 3.7 | 5.5 | +1.7 |
| ResNet-18 | P2, k = 20 | 8.0 | 9.2 | +1.2 |
| ResNet-18 | P2, all | 5.9 | 8.2 | +2.3 |
| ResNet-18 | P3, k = 20 | 10.0 | 10.7 | +0.7 |
| ResNet-18 | P3, all | 9.4 | 11.6 | +2.2 |

F1 rose in 10 of 12 runs (the two drops are −0.1 and −0.2); top-3 rose 1.6 to 4 points. The fix
is worth keeping, but Roboflow stays near chance (4.2%), so cropping was not the main cause of
the cross-dataset failure: the gap between the binarised Kaggle close-ups and the colour
Roboflow videos is.
