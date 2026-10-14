# Tong — BSL source comparison

Does an alphabet-matched source transfer better than ASL?

Run from `notebooks/bsl_comparison.ipynb` on NVIDIA GeForce RTX 4050 Laptop GPU (2.14.1+cu130), methods ['m4_resnet18', 'm3_cnn'], seeds [0, 1, 2].

Source: Kaggle `philipilono/bsl-static-alphabet` — 22,556 images after de-duplication, 24 letters (this dataset omits H and J, not J and Z).
**Licence: Unknown (none declared).** Source, licence and what we actually used: [bsl.md](bsl.md).
BSL and our Auslan label space share 23 letters.

Arms A and B are the existing P2/P3 Roboflow results; C and D are new.
Chance is about 4.2%.
