import numpy as np
import copy
import json
from .logger import log

try_smiles_key = [
    "smiles",
    "SMILES",
    "canonical_smiles",
    "CANONICAL_SMILES"
]

def get_smiles(config):
    for key in try_smiles_key:
        if key in config.info: break
    return config.info[key]

