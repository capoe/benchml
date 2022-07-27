## How to generate an extended-xyz file from a CSV table
```
binput --from_csv example.csv --output example.xyz --gen3d --smiles_from smiles
```

The CSV file can have any number of columns, one of which must contain the compound SMILES:
```
smiles,property1,property2,property3,...
CCC,1.2,0,2,...
CCCC,4.2,0,1,...
```
