```bash
#! /bin/bash
# These are the four steps that lead us from an input csv file to a deployable "benchml" model:
# 1. Prepare input
# 2. Benchmark
# 3. Train
# 4. Apply/deploy

# A short synposis is as follows
# $ binput --from_csv dataset.csv --output dataset.xyz
# $ bmeta --extxyz dataset.xyz --meta dataset.json
# $ bml --mode benchmark --meta dataset.json --models ecfp physchem gylm
# $ bml --mode train --meta dataset.json --models gylm
# $ bml --mode map --archfile models/gylm.arch --extxyz test.xyz
# $ binput --input test.xyz --to_csv test.csv

# 1. Prepare input
# ================

# a) Convert from csv to extended xyz format
binput --from_csv dataset.csv --output dataset.xyz
# b) Create a metadata file that specifies the dataset type, targets etc.
#    The metadata file is generated interactively via a series of prompts:
#    - To stick with the default for any of the prompts, just hit return.
#    - Let's say we would like to regress log-transformed values: To this end,
#      we need to enter 'log10' for 'Set conversion'
bmeta --extxyz dataset.xyz --meta dataset.json

# Feel free to look inside dataset.json. Occasionally we may want to manually modify this
# file, for example to adjust the hypersplitting procedure
# $ vi dataset.json

# 2. Benchmark
# ============

# Here we consider three model groups for which we run the benchmark:
# - ecfp-based models ("ecfp"), here in two different formulations, ("morgan_krr" and "morgan_ridge")
# - physicochemical models ("physchem"), here a combination of 2D physchem and user-defined descriptors
# - convolutional-descriptor models ("gylm"), here with a G-Ylm basis set
# Note that --configure let's us override model parameters. Here we use it to plug in additional descriptors
# from the csv file (here: ClogD, ChemAxon logP). These are forwarded to the "PhyschemUser" node
# The output of the benchmark is stored in models/benchmark.json
bml --mode benchmark --meta dataset.json --models ecfp physchem gylm \
    --configure '{ "PhyschemUser.fields": ["ClogD", "ChemAxon logP"] }' \
    --benchmark_json models/benchmark_dataset.json

# Let's look at the benchmark results
bml --mode analyse --benchmark_json models/benchmark_dataset.json

# 3. Train
# =======
# Based on these results, the gylm architecture performed best by a tiny margin.
# (Side remark: the observation that vastly different models perform so similarly indicates
# that they are limited by data noise (such as experimental error) and/or that the dataset is very challenging.
# Let's use gylm to train our final model
bml --mode train --meta dataset.json --models gylm --archfile models/{model}.arch # <- This will serialize the model to models/gylm.arch

# 4. Apply
# ========
bml --mode map --archfile models/gylm.arch --extxyz dataset.xyz # <- This will output the predictions as a json object
bml --mode map --archfile models/gylm.arch --extxyz dataset.xyz --store_as cl_pred # This will store the predictions in the ext-xyz metadata

# Write back to csv for analysis of the predictions
binput --input dataset.xyz --to_csv dataset_w_pred.csv
```
