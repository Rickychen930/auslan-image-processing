# BSL dataset — source and licence

Provenance for the dataset added in this experiment, and how it compares with the datasets the
project already uses. Recorded so the report can cite it accurately.

---

## The dataset we used

| Field | Value |
|---|---|
| Kaggle ref | `philipilono/bsl-static-alphabet` |
| URL | https://www.kaggle.com/datasets/philipilono/bsl-static-alphabet |
| Title | BSL Static Alphabet |
| Owner | Philip Ilono (`philipilono`) |
| Version | 2 |
| Last updated | 2021-05-02 |
| Size | 821,894,406 bytes (0.82 GB) |
| Downloads / votes | 287 / 3 |
| **Licence** | **Unknown** — none declared by the uploader |

After our own de-duplication and preparation: **22,556 images, 24 letters**, from 93,177 raw
files. The archive ships a smaller 7-letter set a second time under `BSL-data/`, and roughly 76%
of the files are exact duplicates.

Accessed 11 October 2026 via the Kaggle API.

## What "Unknown" means here

Kaggle reports no licence for this dataset, so no permission has been granted explicitly and
default copyright applies. We are not lawyers and this is not a legal assessment; the honest
position for the report is:

- The data was used for **non-commercial academic coursework** only.
- **No images are redistributed** in this repository. Only derived numbers — scores, confusion
  counts and figures — are committed. Anyone reproducing the work downloads the dataset
  themselves from Kaggle under whatever terms apply to them.
- The dataset is **cited, not claimed**.

This is the same standard the project already applies to source code. `docs/RESEARCH.md` records
that where a surveyed repository has no licence, "default copyright applies, so we took ideas only
and wrote our own code."

## A provenance caveat worth stating

The Kaggle entry carries no description, no subtitle and no tags, and its usability rating is 0.0.
Several filenames read like `A (access ambassadors).jpg`, which suggests the images originate from
an organisation's published material rather than from the uploader. We could not establish who
created the photographs or under what terms they were published.

In other words, the licence is not merely unstated — the **upstream source is unverified**. That
should be said plainly in the report rather than left for a marker to discover.

## How this compares with the project's other datasets

| Dataset | Used for | Licence | Owner | Size |
|---|---|---|---|---|
| `grassknoted/asl-alphabet` | ASL source (P1, P3) | **GPL 2** | Akash | 1.10 GB |
| `danrasband/asl-alphabet-test` | ASL cross-dataset test | **CC0: Public Domain** | Dan Rasband | 0.03 GB |
| `rithwikchhugani/auslan-sign-language-fingerspelling-dataset` | Auslan training (P2, P3) | **Unknown** | Rithwik Chhugani | 6.01 GB |
| `philipilono/bsl-static-alphabet` | BSL source (this experiment) | **Unknown** | Philip Ilono | 0.82 GB |
| Roboflow `workspaces-xbqqe/auslan` | Independent Auslan test | see Roboflow Universe | — | — |

Two things follow from this table:

1. **An unlicensed dataset is not new to the project.** Our main Auslan training set has carried an
   Unknown licence from the start. Adding BSL does not introduce a category of risk the project did
   not already have — but it does double it, and neither is currently disclosed in the report.
2. **The ASL training set is GPL 2**, which is a software licence applied to data. If anything in
   the report implies the trained weights are freely redistributable, that claim needs checking.

## A cleaner alternative, and why we did not use it

`alifsathar/bsl-fingerspelling-dataset` is **Apache 2.0** licensed — a clear, permissive licence.
We checked it and could not use it: every one of its 18,664 unique images is already contained in
`philipilono/bsl-static-alphabet`, and it carries training data for **A–G only**, with 8–10 test
images for the remaining letters. It is a 7-class dataset in 24-class folders.

So the permissively licensed option exists but cannot support a 24-letter experiment. That
trade-off is worth one sentence in the report: we chose coverage over licence clarity, knowingly.

## Suggested citation

> P. Ilono, "BSL Static Alphabet," Kaggle, version 2, 2021. [Online].
> Available: https://www.kaggle.com/datasets/philipilono/bsl-static-alphabet
> [Accessed: 11-Oct-2026]. Licence not stated by the uploader.

## If a cleaner source is needed later

BANZ-FS (ICLR 2026) covers BSL, Auslan and NZSL with letter-level alignment, 35,000+ instances.
A third-party summary describes it as CC BY 4.0, **which we have not verified** — the OpenReview
PDF sits behind a browser check. If the licensing position here becomes a problem, that is the
first place to look.

- Paper: https://proceedings.iclr.cc/paper_files/paper/2026/hash/3d4c0a618d0acd7921493e4f30395c22-Abstract-Conference.html
- Forum: https://openreview.net/forum?id=GMR9BUsPbq
