from .accumulator import Accumulator
from .splits import Split

def make_accu_id(model, dataset, mode, stream_train, stream_test):
    return "model={model};data={data};perf={mode};train:test={ntrain:d}:{ntest:d}".format(
        model=model.tag, 
        data=str(dataset).split()[0], 
        mode=mode,
        ntrain=len(stream_train),
        ntest=len(stream_test))

def evaluate_model(dataset, model, accu, log, verbose=False):
    # Open and precompute
    stream = model.open(dataset)
    model.precompute(stream)
    # Evaluate splits
    for stream_train, stream_test in stream.split(**dataset["splits"][0]):
        # Hyper fit
        if model.hyper is not None:
            model.hyperfit(
                stream=stream_train, 
                split_args={"method": "random", "n_splits": 5, "train_fraction": 0.75},
                accu_args={"metric": dataset["metrics"][0]},
                target="y",
                target_ref="input.y",
                log=log)
        else:
            model.fit(stream=stream_train)
        # Evaluate
        output_train = model.map(stream_train)
        output_test = model.map(stream_test)
        accu.append(make_accu_id(model, dataset, "test", stream_train, stream_test), 
            output_test["y"], stream_test.resolve("input.y"))
        accu.append(make_accu_id(model, dataset, "train", stream_train, stream_test), 
            output_train["y"], stream_train.resolve("input.y"))
    model.close(check=False)

def evaluate_ensemble(dataset, models, log, verbose=False):
    log << log.mg << "Dataset: %s" % dataset << log.endl
    accu = Accumulator(metrics=dataset["metrics"])
    for model in models:
        log << log.my << "Model:" << model.tag << log.endl
        evaluate_model(dataset, model, accu, log, verbose)
    performance = accu.evaluateAll(
        metrics=dataset.meta["metrics"], 
        bootstrap=100, log=log)
    return performance

def evaluate(data, models, log, verbose=False):
    bench = {}
    for dataset in data:
        bench.update(evaluate_ensemble(dataset, models, log, verbose))
    return bench

