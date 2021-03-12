## NPhil

This example demonstrates how to build a simple nonlinear feature-network filter model, see https://github.com/capoe/nphil.

```bash
# First, install nphil
pip install mkl mkl-include
pip install nphil

# Generate toy dataset (saved as example.extt)
python generate_toy_dataset.py

# Analyse
philter --extt_file example.extt

# Build & compare models (using BenchML)
pyton model_toy_dataset.py
```
