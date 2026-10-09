# Architecture

This page shows how the code is organised and how data flows through it. All diagrams are
[Mermaid](https://mermaid.js.org/) and render on GitHub and in most Markdown viewers.

## 1. End-to-end flow

From a clean checkout to the tables of the Task 2 report. Each box is one command or module;
everything in the "disk" column is cached, so any step can be repeated or resumed.

```mermaid
flowchart TD
    A([Start]) --> B[auslan download<br/>Kaggle datasets, MediaPipe models,<br/>Roboflow export]
    B --> C[(data/raw)]
    C --> D[auslan prepare]

    subgraph PREP [prepare: once per dataset]
        direction TB
        D1[scan class folders<br/>keep requested letters] --> D2[hash images<br/>MD5 + perceptual hash]
        D2 --> D3[drop exact and near duplicates<br/>drop overlap with reference set]
        D3 --> D4{training set?}
        D4 -- yes --> D5[stratified 70/15/15 split]
        D4 -- no --> D6[whole set = test]
        D5 --> D7
        D6 --> D7[detect hands with MediaPipe<br/>crop to padded union box<br/>fallback: skin blobs or full frame]
    end
    D --> D1
    D7 --> E[(data/work<br/>catalog.csv, landmarks.npy, crops)]

    E --> F[auslan run EXPERIMENT]
    subgraph RUN [runner: one RunSpec at a time]
        direction TB
        F1{protocol} -- P1 --> F2[train ASL<br/>test ASL split + ASL Alphabet Test]
        F1 -- P2 --> F3[draw k images per letter<br/>train from scratch<br/>test Kaggle split + Roboflow]
        F1 -- P3 --> F4[start from ASL weights<br/>then as P2]
        F2 --> F5[scores, predictions, timing]
        F3 --> F5
        F4 --> F5
    end
    F --> F1
    F5 --> G[(artifacts<br/>results.jsonl, predictions)]
    G --> H[auslan report]
    H --> I([tables: Table 9, k curve, confusions,<br/>significance, success criteria])
```

## 2. Layers and dependencies

Arrows point from a module to what it depends on. Nothing in a lower layer imports from a higher
one, so each layer can be tested without the ones above it.

```mermaid
flowchart TD
    CLI[cli.py<br/>commands]
    EXP[experiments.py<br/>E1 to E7 as lists of RunSpec]
    RUN[runner.py<br/>P1 / P2 / P3]
    REP[report/<br/>tables, comparisons, failure rates]
    RES[results.py<br/>RunSpec, Result, ResultStore]
    MET[methods/<br/>Classifier + 5 families + registry]
    EVA[evaluation/<br/>metrics, stats, speed]
    DAT[data/<br/>catalog, dedup, splits, prepare,<br/>SampleSet, DataBundle, download]
    PRE[preprocessing/<br/>HandCropper, MediaPipe adapters]
    CFG[config.py<br/>Settings]

    CLI --> EXP & RUN & REP & DAT
    EXP --> RES
    RUN --> MET & EVA & DAT & RES
    REP --> RES & EVA & EXP
    MET --> EVA & DAT & PRE
    DAT --> PRE & CFG
    EVA --> DAT
    PRE --> CFG
```

| Layer | Responsibility | Knows nothing about |
|-------|----------------|---------------------|
| `config` | immutable `Settings`, YAML overrides | everything else |
| `preprocessing` | find hands, crop, fallback regions | datasets, methods |
| `data` | turn raw folders into prepared sample sets | methods, protocols |
| `methods` | `fit` / `predict_proba` for the five families | protocols, datasets |
| `evaluation` | metrics, significance tests, timing | which method produced a prediction |
| `runner`, `experiments` | what to train on and test on, in which order | how a method works inside |
| `report`, `cli` | present results, run commands | training details |

## 3. The method interface

Every method hides behind one small interface, so the runner treats all of them the same way.
`supports_transfer` tells the runner whether protocol P3 applies, `uses_landmarks` lets it skip
M5 where MediaPipe finds no hands, and `needs_colour` keeps M1 on the colour photo.

```mermaid
classDiagram
    class Classifier {
        <<interface>>
        +name
        +supports_transfer
        +uses_landmarks
        +needs_colour
        +fit(train, val, init)
        +predict_proba(samples)
    }
    class FeatureClassifier {
        features then scikit-learn estimator
        candidates tuned on validation
        supports_transfer = false
    }
    class TorchClassifier {
        step budget, cosine LR, early stopping
        transfer: copy all but the head
        supports_transfer = true
    }
    class ImageClassifier {
        augmentation without flips
    }
    Classifier <|.. FeatureClassifier
    Classifier <|.. TorchClassifier
    TorchClassifier <|-- ImageClassifier
    FeatureClassifier <|-- SkinShapeClassifier : M1
    FeatureClassifier <|-- HogClassifier : M2
    FeatureClassifier <|-- LandmarkClassifier : M5 SVM / RF
    ImageClassifier <|-- CnnClassifier : M3
    ImageClassifier <|-- TorchvisionClassifier : M4 MobileNetV2 / ResNet-18
    TorchClassifier <|-- LandmarkMlp : M5 MLP
```

Methods are looked up by name from a registry (`methods/base.py`), so adding one is a single
decorated function and nothing else changes:

```python
@register("my_method", seeded=True)
def _my_method(**options) -> MyClassifier:
    return MyClassifier("my_method", **options)
```

## 4. The three protocols

```mermaid
flowchart LR
    subgraph ASL [ASL Alphabet, 24 letters]
        A1[train 70%]
        A2[val 15%]
        A3[test 15%]
    end
    A4[ASL Alphabet Test]
    subgraph AUS [Kaggle Auslan, 24 letters]
        K1[train 70%<br/>k per letter drawn]
        K2[val 15%<br/>10 per letter used]
        K3[test 15%]
    end
    R[Roboflow Auslan<br/>independent test]

    A1 --> P1[P1 train]
    A2 --> P1
    P1 --> A3
    P1 --> A4

    K1 --> P2[P2 train from scratch]
    K2 --> P2
    P2 --> K3
    P2 --> R

    A1 -. style T .-> PRE[ASL pretraining]
    PRE -- weights without the output layer --> P3[P3 train]
    K1 --> P3
    K2 --> P3
    P3 --> K3
    P3 --> R
```

P2 and P3 differ only in the starting weights: the same drawn images, validation images, seed,
step budget and test sets. Hand-crafted methods cannot reuse ASL training, so for them P3 is P2
and the runner skips it.

## 5. One run in detail

```mermaid
sequenceDiagram
    participant CLI as auslan run
    participant PL as experiments.plan
    participant RU as Runner
    participant ST as ResultStore
    participant DB as DataBundle
    participant ME as Classifier

    CLI->>PL: experiment, methods, seeds, protocols
    PL-->>CLI: list of RunSpec (no duplicates)
    loop every RunSpec
        CLI->>RU: run(spec)
        RU->>ST: has(spec.key)?
        ST-->>RU: no
        RU->>ME: create(method, seed, options)
        RU->>DB: train / validation / test sets
        opt protocol P3
            RU->>RU: load or train ASL weights (cached)
        end
        RU->>ME: fit(train, val, init)
        RU->>ME: predict_proba(each test set)
        RU->>RU: macro-F1, accuracy, top-3, CPU time, model size
        RU->>ST: add(results, predictions)
    end
```

## 6. Files on disk

```
data/                       git-ignored
  raw/<dataset>/            downloaded, untouched
  models/                   MediaPipe hand and face models
  work/<dataset>/
    hashed.csv              paths with MD5 and perceptual hashes (cache of the slow step)
    chunks/*.npz            crop results per 2,000 images (resume point)
    crops/*.jpg             224 px hand crops
    landmarks.npy           (N, 2, 21, 3) hand landmarks, zero where a hand was not found
    settings.json           what the dataset was built with; mismatches are refused
    catalog.csv             one row per image: path, label, split, n_hands, used_fallback, ...
artifacts/                  git-ignored
  results.jsonl             one line per run and test set
  predictions/*.npz         labels and class probabilities of every run
  pretrained/*.joblib       ASL-trained weights that P3 starts from
  crop_ms.json              hand detection time per frame
  report/                   the tables written by `auslan report`
```

## 7. Design decisions worth knowing

- **Leakage first.** Duplicates are removed before splitting, test sets never influence tuning,
  and the overlap of a test set with its training set is removed from the test set.
- **Same pipeline for everyone.** Hands are cropped once; M1 to M4 read the cached crop and M5
  reads the cached landmarks, so differences between methods are not data-handling differences.
- **Failures are visible.** Per-image hand counts and fallback use are stored in the catalog and
  summarised by the report. M5 is skipped, with a warning, where training images contain no
  detected hands (the report's risk register anticipates this for the Kaggle drawings).
- **Small, replaceable parts.** Detectors, fallbacks and methods are passed in or looked up by
  name, so tests use tiny fakes and new variants need no changes to the runner.
