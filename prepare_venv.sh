#!/bin/bash
python3 -m venv .venv
source .venv/bin/activate
pip install hmmlearn==0.3 wandb==0.16 numpy==1.24.4 scikit-learn==1.3.2