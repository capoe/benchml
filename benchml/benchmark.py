import json
import copy
import time
from .accumulator import Accumulator
from .splits import Split
from .logger import log

def parse(benchmark):
    if isinstance(benchmark, str):
        benchmark = json.load(open(benchmark))
    parsed = []
    for model, data in benchmark.items():
        meta = { kv[0]: kv[1] for pair in model.split(";") for kv in [pair.split("=")] }
        parsed.append({ "model": model, "meta": meta, "data": data })
    return parsed

def make_split_id(method, group, stream_train, stream_test):
    return "split={method:s};perf={perf:s};train:test={ntrain:d}:{ntest:d}".format(
        method=method, perf=group, ntrain=len(stream_train), ntest=len(stream_test))

def evaluate_model(dataset, model, 
        accu=None, 
        log=log, 
        verbose=False,
        detailed=True):
    if accu is None:
        accu = Accumulator(metrics=dataset["metrics"])
    record = {
        "dataset": str(dataset).split()[0],
        "model": model.tag,
        "splits": [],
        "hypers": {},
        "slices": {},
        "output": {},
        "performance": {},
    }
    # Open and precompute
    t_in = time.time()
    stream = model.open(dataset)
    model.precompute(stream, verbose=verbose)
    for split_args in dataset["splits"]:
        # Evaluate splits
        for stream_train, stream_test in stream.split(**split_args):
            train_id = make_split_id(dataset["splits"][0]["method"], "train", stream_train, stream_test)
            test_id = make_split_id(dataset["splits"][0]["method"], "test", stream_train, stream_test)
            log << "Evaluating %s|%s" % (train_id, test_id) << log.endl
            # Hyper fit
            if model.hyper is not None:
                model.hyperfit(
                    stream=stream_train, 
                    split_args=dataset["hypersplit"] if "hypersplit" in dataset.meta \
                        else {"method": "random", "n_splits": 5, "train_fraction": 0.75},
                    accu_args={"metric": dataset["metrics"][0]},
                    target="y",
                    target_ref="input.y",
                    log=log)
            else:
                model.fit(stream=stream_train)
            # Evaluate
            output_train = model.map(stream_train)
            output_test = model.map(stream_test)
            accu.append(test_id, output_test["y"], stream_test.resolve("input.y"))
            accu.append(train_id, output_train["y"], stream_train.resolve("input.y"))
            # Log hyper args and output
            if not test_id in record["splits"]:
                record["splits"].append(test_id)
                record["splits"].append(train_id)
                record["hypers"][train_id] = []
                record["hypers"][test_id] = []
                record["slices"][train_id] = []
                record["slices"][test_id] = []
                record["output"][train_id] = []
                record["output"][test_id] = []
            if detailed:
                model_args = copy.deepcopy(model.compileArgs())
                record["hypers"][train_id].append(model_args)
                record["hypers"][test_id].append("@"+train_id)
                record["slices"][train_id].append(stream_train.slice.tolist())
                record["slices"][test_id].append(stream_test.slice.tolist())
                record["output"][test_id].append({
                    "pred": output_test["y"].tolist(), 
                    "true": stream_test.resolve("input.y").tolist()})
                record["output"][train_id].append({
                    "pred": output_train["y"].tolist(), 
                    "true": stream_train.resolve("input.y").tolist()})
        model.close(stream, check=False)
    t_out = time.time()
    perf = accu.evaluateAll(
        metrics=dataset.meta["metrics"],
        bootstrap=100,
        log=log)
    record["performance"] = perf
    record["walltime"] = float(t_out - t_in)
    return record

def evaluate_ensemble(dataset, models, log, verbose=False, detailed=False):
    log << log.mg << "Dataset: %s" % dataset << log.endl
    benchdata = []
    for model in models:
        log << log.my << "Model:" << model.tag << log.endl
        record = evaluate_model(dataset, model, 
            log=log, verbose=verbose, detailed=detailed)
        benchdata.append(record)
    return benchdata

def evaluate(data, models, log, verbose=False, detailed=False):
    bench = []
    for dataset in data:
        benchdata = evaluate_ensemble(dataset, models, log, verbose, detailed)
        bench.extend(benchdata)
    return bench

