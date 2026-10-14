# BSL experiment — brief

Why we added a second source language, and what it changed.

---

## What we changed

Every pipeline is tested on the same thing: the independent **Roboflow** Auslan photographs.
Only the training route differs.

| | Source | Then fine-tuned on | Tested on | Macro-F1 % |
|---|---|---|---|---|
| **Before** — baseline | *(none)* | Kaggle Auslan — **line drawings** | Roboflow — photos | 8.44 ± 3.61 |
| **Before** — our original transfer | **ASL** — one-handed photos | Kaggle Auslan — **line drawings** | Roboflow — photos | 10.48 ± 2.88 |
| **After** — swap the source | **BSL** — two-handed photos | Kaggle Auslan — **line drawings** | Roboflow — photos | 15.42 ± 0.97 |
| **After** — skip the drawings | **BSL** — two-handed photos | *(nothing — straight to test)* | Roboflow — photos | **23.91 ± 1.93** |
| **Control** — skip them with ASL | **ASL** — one-handed photos | *(nothing — straight to test)* | Roboflow — photos | **3.93 ± 1.06** |

ResNet-18, mean ± s.d. over three seeds. Chance is about 4.2%.

---

## The result in one table

The control exists so the two changes can be told apart. It makes the whole thing legible:

|  | via Kaggle Auslan | straight to Roboflow |
|---|---|---|
| **ASL** source | 10.48 ± 2.88 | **3.93 ± 1.06** |
| **BSL** source | 15.42 ± 0.97 | **23.91 ± 1.93** |

**The two changes interact — the route that helps one source ruins the other.**

| Change | With an ASL source | With a BSL source |
|---|---|---|
| Remove Kaggle Auslan from the chain | 10.48 → 3.93 (**−6.6**) | 15.42 → 23.91 (**+8.5**) |

---

## Why BSL

ASL is a **one-handed** alphabet whose handshapes are unrelated to Auslan's. ASL "A" and Auslan
"A" share a name and nothing else, so there was never anything real for the transfer to carry.

BSL, Auslan and NZSL form the BANZSL family and share the **same two-handed alphabet**. NZSL
would have worked equally well in principle, but its one substantial dataset was never released.
BSL was the only matched source we could obtain — 22,556 usable images from Kaggle
(`philipilono/bsl-static-alphabet`) after de-duplication.

It is also **photographs**, where the Kaggle Auslan set is binarised line drawings.

---

## What it means

**ASL alone performs at chance.** 3.93 against a 4.35 baseline. Shown Auslan photographs, an
ASL-trained model does no better than guessing — which is structurally unsurprising once stated,
since it learned one-handed shapes for a two-handed alphabet.

**That explains the E1 null.** ΔT(20) ≈ 0 was not a weak transfer effect; it was *no* effect,
because nothing transferable existed. Everything the original pipeline achieved came from the
Kaggle Auslan fine-tune, not from the ASL pretraining.

**A matched alphabet makes the Auslan data optional.** BSL reaches 23.91 having never seen a
single Auslan training image. Passing it through the drawings *costs* it 8.5 points.

**So the drawings are not simply bad.** They are the only thing keeping an ASL source above
chance, and the main thing holding a BSL source back. Which answer you get depends on what you
started from.

**23.91% is still unusable.** Our success criterion S2 asked for macro-F1 ≥ 60% and top-3 ≥ 85%;
we have 23.91% and about 38%. This identifies *which component is broken*, not a working
recogniser.

**The remaining gap looks like framing.** BSL crops mostly contain one detected hand (55.6%),
while Roboflow crops almost always contain two (94.4%). For a two-handed alphabet the model is
trained and tested on structurally different inputs, even though both are photographs.

### Three faults, found one layer at a time

1. **Wrong alphabet** — ASL is unrelated to Auslan, and contributes nothing on its own
2. **Wrong medium** — line drawings cannot stand in for photographs
3. **Wrong framing** — one hand in training, two at test

Each was hidden by the one above it, which is why three experiments were needed to separate them.

---

## Caveats

- `m3_cnn` is at or below chance in every arm (1.1 – 3.5) and its arm ordering is noise. It is a
  null result, not a comparison.
- The "straight to Roboflow" arms use the 23 letters BSL and Auslan share; the "via Kaggle
  Auslan" arms use our usual 24. Chance differs by 0.1 points, but the tasks are not identical.
- Training on CUDA is not bit-reproducible: repeated runs of the same seed moved arm C by up to
  1.7 points. All figures are means over three seeds for that reason.
- The Roboflow test set is 242–250 images over 23–24 classes, so roughly ten per letter.

---

## Files

| File | Contents |
|---|---|
| `bsl_arms.csv` | Every run: arm, method, seed, macro-F1, accuracy, top-3 |
| `comparison_macro_f1.csv` | All five pipelines side by side |
| `comparison.png` | The chart as a figure |
| `README.md` | Hardware, torch build, dataset provenance |

The CSVs label the pipelines A–E, in the order of the first table: **A** baseline, **B** via ASL,
**C** via BSL, **D** BSL direct, **E** ASL direct. `D*` is a diagnostic — arm D scored with the
drawing-style transform wrongly applied, which collapses it from 23.9 to 1.0 and is the clearest
single measure of how much the medium matters.

Produced by `notebooks/bsl_comparison.ipynb`.
