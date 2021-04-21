import benchml as bml
import nphil
import scipy.stats
import sklearn.metrics
import numpy as np
log = bml.log
log.setLevel("info")
bml.splits.synchronize(0)

def build_simple_rf():
    return bml.pipeline.Module(
        tag="ExttRandomForest",
        transforms=[
            bml.transforms.ExttInput(tag="input"),
            bml.transforms.RandomForestRegressor(tag="predictor",
                inputs={"X": "input.X", "y": "input.Y"}),
        ],
        hyper=bml.transforms.GridHyper(
            bml.transforms.Hyper({"predictor.max_depth": [None]})),
        broadcast={},
        outputs={"y": "predictor.y"})


def build_simple_nphil():
    return bml.pipeline.Module(
        tag="ExttMarchenkoLinear",
        transforms=[
            bml.transforms.ExttInput(
                tag="input"),
            bml.transforms.CleanMatrix(
                tag="clean",
                args={
                    "axis": 0, 
                    "std_threshold": 1e-10},
                inputs={"X":"input.X"}),
            bml.transforms.MarchenkoPasturFilter(
                tag="descriptor",
                args={},
                inputs={"X":"clean.X"}),
            bml.transforms.LinearRegression(
                tag="predictor",
                args={},
                inputs={
                    "X": "descriptor.X",
                    "y": "input.Y"})
        ],
        outputs={"y":"predictor.y"})

def build_models():
    return [
        build_simple_rf(),
        build_simple_nphil()
    ]

if __name__ == "__main__":
    dataset = bml.load_dataset("ising.extt")
    for model in build_models():
        log << log.mg << "Model %s" % model.tag << log.endl
        stream = model.open(dataset)
        accu = bml.Accumulator(metrics=["mae", "r2", "rhop"])
        for idx, (stream_train, stream_test) in enumerate(stream.split(method="random", n_splits=10, train_fraction=0.9)):
            log << log.back << "  Split %3d" % idx << log.flush
            model.fit(stream_train)
            output_train = model.map(stream_train)
            output_test = model.map(stream_test)
            accu.append("train", output_train["y"], stream_train.resolve("input.Y"))
            accu.append("test", output_test["y"], stream_test.resolve("input.Y"))
        log << log.endl
        res = accu.evaluateAll(log=bml.log, bootstrap=100)
        
