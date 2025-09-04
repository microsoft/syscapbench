#!/bin/bash

set -e  # Exit immediately on error.

echo "==> Setting up VanSys-CLI environment..."

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
deactivate

echo "==> VanSys-CLI environment is set up successfully."
