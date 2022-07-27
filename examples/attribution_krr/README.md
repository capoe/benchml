## To run this example, simply:
```
python build_example.py
```

The script first runs a benchmark, then trains a model on the entire dataset, and finally applies the trained model to a query compound. Note that the attribution output can be saved as an extended-xyz file, see the "write\_xyz" parameter in the model definition.


## What if all I have is a CSV table?
The descriptor in this example relies on 3D coordinates being available as part of the data input. If your dataset is 2D, you will therefore need to generate 3D conformers and save the resulting dataset in extended-xyz format:
```
binput --from_csv example.csv --output example.xyz --gen3d --smiles_from smiles
```

The input CSV file can have any number of columns, one of which must contain the compound SMILES:
```
smiles,property1,property2,property3,...
CCC,1.2,0,2,...
CCCC,4.2,0,1,...
```
