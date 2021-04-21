```bash
cd data
binput --from_csv 5ht2b.csv --output 5ht2b.xyz  --smiles_from smiles
cd ..
bmeta -e data/*.xyz -m metadata.json
bml --mode benchmark --meta metadata.json --models "bmol_ecfp_.*_class" --bench benchmark.json
bml --mode analyse --bench benchmark.json
```
