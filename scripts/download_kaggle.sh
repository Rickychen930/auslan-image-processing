#!/usr/bin/env bash
# Download the three Kaggle datasets used by the project into data/raw/<name>.
set -euo pipefail
cd "$(dirname "$0")/.."
fetch() { # slug name
  local dest="data/raw/$2"
  if [ -d "$dest" ] && [ -n "$(ls -A "$dest" 2>/dev/null)" ]; then echo "skip $2 (exists)"; return; fi
  mkdir -p "$dest"
  uv run kaggle datasets download -d "$1" -p "$dest" --unzip
  rm -f "$dest"/*.zip
}
fetch danrasband/asl-alphabet-test asl_alphabet_test
fetch grassknoted/asl-alphabet asl_alphabet
fetch rithwikchhugani/auslan-sign-language-fingerspelling-dataset auslan_kaggle
echo DONE
