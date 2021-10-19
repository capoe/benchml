import numpy as np
import scipy.stats
import sklearn.metrics

import benchml as bml

log = bml.log
log.setLevel("info")
bml.splits.synchronize(0)


def build_marchenko_conformal():
    return bml.pipeline.Module(
        tag="ExttMarchenkoConformalEnsemble",
        transforms=[
            bml.transforms.ExttInput(tag="input"),
            bml.transforms.CleanMatrix(
                tag="clean", args={"axis": 0, "std_threshold": 1e-10}, inputs={"X": "input.X"}
            ),
            bml.transforms.MarchenkoPasturFilter(
                tag="descriptor", args={}, inputs={"X": "input.X"}
            ),
            bml.transforms.LinearRegression(tag="linear", detached=True, args={}),
            bml.transforms.EnsembleRegressor(
                tag="ensemble",
                detached=True,
                args={
                    "size": 10,
                    "forward_inputs": {"X": "X", "y": "y"},  # = default
                    "input_type": "descriptor",
                },
                inputs={"base_transform": "linear"},
            ),
            bml.transforms.ConformalRegressor(
                tag="predictor",
                args={
                    "forward_inputs": {"X": "X", "y": "y"},  # = default
                    "input_type": "descriptor",  # = default
                },
                inputs={"X": "descriptor.X", "y": "input.Y", "base_transform": "ensemble"},
            ),
        ],
        broadcast={},
        outputs={"y": "predictor.y", "dy": "predictor.dy"},
    )


def build_simple_ensemble():
    return bml.pipeline.Module(
        tag="ExttLinearEnsemble",
        transforms=[
            bml.transforms.ExttInput(tag="input"),
            bml.transforms.LinearRegression(detached=True, args={}),
            bml.transforms.EnsembleRegressor(
                tag="predictor",
                args={
                    "size": 10,
                    "forward_inputs": {"X": "X", "y": "y"},  # = default
                    "input_type": "descriptor",
                },
                inputs={"X": "input.X", "y": "input.Y", "base_transform": "LinearRegression"},
            ),
        ],
        broadcast={},
        outputs={"y": "predictor.y", "dy": "predictor.dy"},
    )


def build_simple_linear():
    return bml.pipeline.Module(
        tag="ExttLinear",
        transforms=[
            bml.transforms.ExttInput(tag="input"),
            bml.transforms.LinearRegression(
                tag="predictor", args={}, inputs={"X": "input.X", "y": "input.Y"}
            ),
        ],
        broadcast={},
        outputs={"y": "predictor.y"},
    )


def build_models():
    return [build_marchenko_conformal(), build_simple_ensemble(), build_simple_linear()]


if __name__ == "__main__":
    dataset = bml.load_dataset("data.extt")
    for model in build_models():
        log << log.mg << "Model %s" % model.tag << log.endl
        stream = model.open(dataset)
        accu = bml.Accumulator(metrics=["mae", "r2", "rhop"])
        yt = []
        yp = []
        dyp = []
        for idx, (stream_train, stream_test) in enumerate(
            stream.split(method="random", n_splits=10, train_fraction=0.9)
        ):
            log << log.back << "  Split %3d" % idx << log.flush
            model.fit(stream_train)
            output_train = model.map(stream_train)
            output_test = model.map(stream_test)
            accu.append("train", output_train["y"], stream_train.resolve("input.Y"))
            accu.append("test", output_test["y"], stream_test.resolve("input.Y"))
        log << log.endl
        res = accu.evaluateAll(log=bml.log, bootstrap=0)
