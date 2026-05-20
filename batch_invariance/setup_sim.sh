#!/usr/bin/env bash
# Laptop NKI CPU simulator venv — see ../../article.md for macOS wheel rename.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3.12}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "Need Python 3.12 (e.g. brew install python@3.12)" >&2
  exit 1
fi

"$PY" -m venv nki-cpu-sim
source nki-cpu-sim/bin/activate
pip install --upgrade pip
pip install numpy ml-dtypes torch

WHEEL_DIR="nki-cpu-sim/wheels"
mkdir -p "$WHEEL_DIR"
curl -L -o "$WHEEL_DIR/nki-linux.whl" \
  "https://pip.repos.neuron.amazonaws.com/nki/nki-0.3.0%2B23928721754.g18aa1271-cp312-cp312-linux_x86_64.whl"
cp "$WHEEL_DIR/nki-linux.whl" "$WHEEL_DIR/nki-0.3.0-cp312-cp312-macosx_14_0_arm64.whl"
pip install "$WHEEL_DIR/nki-0.3.0-cp312-cp312-macosx_14_0_arm64.whl" --no-deps

echo "Installing nki-library (attention_cte) from GitHub..."
pip install "git+https://github.com/aws-neuron/nki-library.git" -q

echo ""
echo "Done.  source nki-cpu-sim/bin/activate && ./run.sh sim"
