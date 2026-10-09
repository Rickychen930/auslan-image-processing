# Setup

## Requirements

| Need | Details |
|------|---------|
| OS | macOS (Apple silicon uses the GPU through MPS), Linux, or Windows via WSL |
| Python | 3.12 (MediaPipe wheels do not exist for 3.13 and newer yet) |
| Tooling | [uv](https://docs.astral.sh/uv/) |
| Disk | about 20 GB free: 7.2 GB of downloads, plus prepared crops |
| Accounts | a free Kaggle account; a free Roboflow account for the independent test set |
| Time | `prepare` took about 4 minutes on an M2 Pro (all four datasets); E1 takes several hours on a laptop |

Python packages are pinned in `uv.lock`. `requirements.txt` and `requirements-dev.txt` are exports
of the same versions for people who prefer `pip`.

## Install

```bash
git clone <this repository> && cd "Image Processing"
uv sync                                       # creates .venv with Python 3.12
uv run pytest tests/test_splits.py            # quick check that the install works
```

With pip instead of uv:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt && pip install -e . --no-deps
```

## Credentials

**Kaggle** (three datasets). Under *Kaggle → Settings → API*, choose *Generate New Token* and
save it to `~/.kaggle/access_token` (`chmod 600`) or export it as `KAGGLE_API_TOKEN`. A legacy
`kaggle.json` in `~/.kaggle/`, or `KAGGLE_USERNAME` with `KAGGLE_KEY`, also works. The
`auslan-sign-language-fingerspelling-dataset` page may ask you to accept its terms once in the
browser.

**Roboflow** (the independent Auslan test set). Sign in at roboflow.com, open *Settings → API
Keys*, and export the key in your shell. Never commit it:

```bash
export ROBOFLOW_API_KEY="<your key>"
uv run auslan download --roboflow                          # newest version; or add --roboflow-version 2
```

## Run

```bash
uv run auslan download            # models and Kaggle data (resumable, skips what exists)
uv run auslan prepare             # cached crops in data/work/
uv run auslan run E1 --dry-run    # list the planned runs
uv run auslan run E1              # train and evaluate; artifacts/ holds results and predictions
uv run auslan report              # artifacts/report/
```

All settings live in `src/auslan/config.py`; override any of them with a YAML file
(`uv run auslan --config configs/default.yaml run E1`). Unknown keys are rejected.


## Hardware notes

- Training picks MPS (Apple GPU), otherwise the CPU. There is no CUDA path because the project
  targets a student laptop; adding one means extending `default_device()` in
  `methods/torch_base.py`.
- `workers` (default 4) is the number of processes used for hashing and hand detection. Lower it
  on machines with little memory.
- MobileNetV2 is slow on the CPU because of its depthwise convolutions; use MPS or ResNet-18
  there.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `kaggle ... 401` or `403` | check the token file permissions, or accept the dataset terms in a browser |
| `auslan_roboflow is not prepared yet` | download Roboflow first (see above) or run only experiments that avoid P2/P3 |
| `ROBOFLOW_API_KEY` error | export the variable in the same shell that runs `auslan` |
| Out of disk during `download` | free about 15 GB; the Auslan archive is 6 GB zipped and unpacks to 6 GB |
| MediaPipe model errors | rerun `auslan download`; it fetches `data/models/*.task` and `*.tflite` |
