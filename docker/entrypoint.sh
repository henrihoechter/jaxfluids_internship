#!/usr/bin/env bash
set -euo pipefail

# Ensure Python exists (comes from system Python 3.11)
python3 --version >/dev/null 2>&1 || python --version

# Create project-local venv if missing
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

# Activate (PATH is already set, but source for safety)
# shellcheck source=/dev/null
source .venv/bin/activate

# Upgrade pip quietly
python -m pip install --upgrade pip >/dev/null

# Install dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
  pip install -r requirements.txt
fi

# Optional: install pre-commit hooks if config present
if [ -f ".pre-commit-config.yaml" ]; then
  pre-commit install || true
fi

# Hand off to whatever the devcontainer starts
exec "$@"
