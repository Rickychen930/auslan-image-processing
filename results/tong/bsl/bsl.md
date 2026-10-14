# BSL dataset

**Source:** https://www.kaggle.com/datasets/philipilono/bsl-static-alphabet

"BSL Static Alphabet" by Philip Ilono, version 2, published 2021-05-02. Photographs of the
British Sign Language fingerspelling alphabet, one folder per letter. 0.82 GB as downloaded.

We used it because BSL, Auslan and NZSL share the same two-handed manual alphabet, so it gives
the alphabet-matched source language that ASL never was.

## What we actually used

| | |
|---|---|
| Files in the download | 93,177 |
| **Images used after preparation** | **22,556** |
| Letters | 24 — the set omits H and J |
| Letters shared with our Auslan label space | 23 |
| Hands detected by MediaPipe | 80% |

The download is heavily duplicated — it ships a smaller 7-letter set a second time under
`BSL-data/`, and about **76% of the files are duplicates**. These were removed by the project's
existing de-duplication (MD5 plus perceptual hash) **before** the train/validation/test split, so
no near-identical image appears on both sides of the split. That is the same guard used for every
other dataset in this project.

The download also contains 236 test images sitting loose in a `test/` folder with no letter
subdirectories. Because they carry no label, the dataset scanner skipped them; every one of the
22,556 images came from a properly labelled folder.

## Licence

**Not stated.** Kaggle reports the licence as Unknown, so default copyright applies. No images
are redistributed in this repository — only derived numbers such as scores and confusion counts.
Anyone reproducing this work downloads the dataset themselves.

## Citation

> P. Ilono, "BSL Static Alphabet," Kaggle, version 2, 2021. [Online].
> Available: https://www.kaggle.com/datasets/philipilono/bsl-static-alphabet
> [Accessed: 11-Oct-2026]. Licence not stated by the uploader.

Prepared by `notebooks/bsl_comparison.ipynb`. Results in this folder.
