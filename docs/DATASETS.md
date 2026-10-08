# Datasets

All four datasets are public and used for coursework only. Check each licence before reuse; none
of the data is redistributed from this repository (`data/` is git-ignored).

| Name in code | Dataset | Where to get it | Size | Licence | Role |
|--------------|---------|-----------------|------|---------|------|
| `asl_alphabet` | ASL Alphabet, 87,000 images, 29 classes | [Kaggle `grassknoted/asl-alphabet`](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) | 1.1 GB | GPL-2.0 | ASL training (P1, P3 pretraining) |
| `asl_alphabet_test` | ASL Alphabet Test, 870 images on varied backgrounds | [Kaggle `danrasband/asl-alphabet-test`](https://www.kaggle.com/datasets/danrasband/asl-alphabet-test) | 25 MB | CC0 | ASL cross-dataset test (P1) |
| `auslan_kaggle` | Auslan Fingerspelling, about 71,000 binarised images, 36 classes | [Kaggle `rithwikchhugani/auslan-sign-language-fingerspelling-dataset`](https://www.kaggle.com/datasets/rithwikchhugani/auslan-sign-language-fingerspelling-dataset) | 6.0 GB | not stated | Auslan training (P2, P3) |
| `auslan_roboflow` | Roboflow Auslan, 352 photos with hand boxes | [Roboflow Universe `workspaces-xbqqe/auslan`](https://universe.roboflow.com/workspaces-xbqqe/auslan) | small | CC BY 4.0 (verify on the page) | independent Auslan test (P2, P3) |

No Hugging Face or other mirror of these four exists; Kaggle and Roboflow are the only sources.
The pipeline also needs two small MediaPipe models (Google, Apache-2.0), fetched by
`auslan download` from `storage.googleapis.com/mediapipe-models`.

## Downloading

```bash
uv run auslan download                                   # three Kaggle sets + MediaPipe models
export ROBOFLOW_API_KEY="<your key>"                     # free account, never commit it
uv run auslan download --roboflow                        # Roboflow, as COCO then sorted by letter
```

To fetch Kaggle sets by hand instead: `kaggle datasets download -d <slug> -p data/raw/<name> --unzip`
with the names from the table above (`data/raw/asl_alphabet`, `data/raw/asl_alphabet_test`,
`data/raw/auslan_kaggle`).

**Roboflow by hand.** Open the page, pick a version, choose *Download dataset* and the *COCO* format
(or *Folder structure* if the project offers it), and unzip. If you used COCO, run
`auslan download --roboflow` once to see the layout we expect: one folder per letter under
`data/raw/auslan_roboflow/<LETTER>/`. Any layout with letter-named folders works, because the
catalog scanner only looks at the name of each image's parent folder.

## What the data looks like

- **ASL Alphabet**: 200 x 200 colour photos of one signer, 3,000 frames per class cut from video, so
  neighbouring frames are almost identical. `del`, `nothing`, `space` and the motion letters J and Z
  are dropped, leaving 24 letters.
- **ASL Alphabet Test**: the same letters, 30 per class, on varied backgrounds. The Kaggle download
  unpacks every image twice (a nested copy); the exact-duplicate step removes the copy.
- **Auslan Kaggle**: 500 x 300 greyscale webcam frames of a person signing with both hands, already
  blurred and binarised into an edge-like drawing (`dataset_R_G_blur/<CLASS>/<n>.jpg`). Digits are
  dropped, keeping 26 letters. Signers are not identified.
- **Roboflow Auslan**: colour photos of two-handed letters from a different source. Its images pass
  through the same hand crop; the style transform T (greyscale, blur, adaptive threshold) makes them
  resemble the Kaggle drawings for the main test, and E6 also scores them without T.

## Preparation (`auslan prepare`)

1. scan class folders and keep the requested letters
2. drop exact duplicates (MD5) and near-duplicates (perceptual hash within a letter)
3. for Roboflow, also drop anything that nearly matches a Kaggle Auslan image
4. split 70/15/15 stratified by letter (training sets only)
5. detect hands with MediaPipe, crop to the padded union box, and cache 224 px crops and landmarks

Each stage is cached under `data/work/<name>/` (`hashed.csv`, `crops/`, `chunks/`, `catalog.csv`,
`landmarks.npy`). Delete a dataset's folder to redo it from scratch.

### Counts after preparation

| Dataset | Raw images (kept letters) | After de-duplication | Train / val / test | Per letter | No hand found |
|---------|--------------------------|----------------------|--------------------|------------|---------------|
| `asl_alphabet` | 72,000 | 16,980 | 11,886 / 2,547 / 2,547 | 583 to 865 | 21.5% |
| `asl_alphabet_test` | 1,440 (every image twice) | 704 | test only | 27 to 30 | 3.7% |
| `auslan_kaggle` | 51,263 | 11,913 | 8,339 / 1,787 / 1,787 | 356 to 691 (train: 249 to 484) | 100% |
| `auslan_roboflow` | not downloaded yet | | test only | | |

- The near-duplicate step (perceptual hash within 4 of 64 bits) is strict on purpose: consecutive
  video frames would otherwise land in both train and test. It is the reason "all data" is about
  320 images per letter and not the 1,400 mentioned in the Task 1 report. Change
  `near_duplicate_hamming` to trade size against leakage; the counts above are for the default.
- MediaPipe finds **no hand in any Kaggle Auslan image**. They are blurred, binarised 500 x 300
  frames of a whole upper body, and lowering the detection threshold to 0.03 still finds a hand
  in only 5 of 120 sampled images. These images therefore use the full-frame fallback, and M5
  (landmarks) is reported under P1 only, the outcome the report's risk register prepares for.
