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

## Test a single benchmark instance

In the "atomized" form, a batch consists of benchmarking a single model against a single dataset. If you would like to verify that the benchmark command does not result in any start-up or runtime errors for a particular model/dataset, use the following bml command:

```bash
bml --meta path/to/meta.json --models "^my_model_tag$" --benchmark_json results/test.json --use_ase
```

Note that the model tag is treated as a regular expression that is matched against all model tags in the benchml collection (these can be inspected via binfo --list_collections), hence the circumflex and dollar sign to guarantee matching against a single model entry only. 
