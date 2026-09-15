#!/usr/bin/env bash
# Clones the upstream Thirukkural dataset into the path flatten_dataset.py expects.
# data/raw/ is gitignored, so run this once after cloning the repo.
set -euo pipefail
cd "$(dirname "$0")/.."

DEST="data/raw/thirukkural-source/Thirukkural-Tamil-Dataset"
if [ -d "$DEST/.git" ]; then
  echo "Dataset already present at $DEST"
  exit 0
fi

git clone --depth 1 https://github.com/vijayanandrp/Thirukkural-Tamil-Dataset.git "$DEST"
echo "Done. Next: python3 scripts/flatten_dataset.py"
