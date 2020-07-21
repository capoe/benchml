## Prepare batch scripts

The bbatch tool assists with partitioning the benchmark onto batch jobs. In the "atomized" case, a single batch consists of one model run against one dataset.
The input arguments specify the data directory which is scanned for metadata files ("meta.json"), the models to be included in the benchmark, and a template batch script (here for the torque queueing system).

```bash
bbatch --walk data --collections "^bmol_.*" --models ".*" \
    --template batches/template.sh \
    --nodes "nodes=cpu:ppn=2" \
    --atomize

mkdir -p logs # Collects stdout
```
