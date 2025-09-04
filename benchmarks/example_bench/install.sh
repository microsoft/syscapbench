#!/bin/bash

set -e  # Exit immediately on error.

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pytest
pip install pytest-cov
deactivate

echo "==> ExampleBench environment is set up successfully."
