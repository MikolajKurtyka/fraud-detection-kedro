#!/usr/bin/env bash
# One-command project setup for Linux / macOS / Git Bash.
#   ./scripts/setup.sh              # environment + git hooks + credential check
#   ./scripts/setup.sh --download   # ...and download the Kaggle data
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> Checking uv"
if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found, installing..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "==> Installing dependencies (uv sync)"
uv sync

echo "==> Installing pre-commit hooks"
[ -d .git ] || git init -q
uv run pre-commit install

[ -f .env ] || { cp .env.example .env; echo "Created .env from .env.example"; }

echo "==> Checking Kaggle credentials and competition access"
uv run python scripts/download_data.py --check

if [ "${1:-}" = "--download" ]; then
  echo "==> Downloading data"
  uv run python scripts/download_data.py
fi

echo
echo "Setup complete. Next: uv run kedro run --pipelines data_ingestion"
