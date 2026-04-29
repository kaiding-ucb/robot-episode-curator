#!/usr/bin/env bash
# One-shot setup: create Python venv, install backend deps, install frontend deps.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON="${PYTHON:-python3}"

cd "$ROOT_DIR"

if [[ ! -d ".venv" ]]; then
    echo "==> Creating virtualenv at .venv"
    "$PYTHON" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Installing Python deps (editable + dev + ai)"
pip install --upgrade pip
pip install -e ".[dev,ai]"

echo "==> Installing frontend deps"
cd frontend
if [[ -f package-lock.json ]]; then
    npm ci
else
    npm install
fi

echo
echo "Setup complete. Next steps:"
echo "  cp .env.example .env    # then fill in tokens"
echo "  make dev                # start backend + frontend"
