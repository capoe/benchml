import numpy as np
import rdkit.Chem as chem
from rdkit.Chem import rdDistGeom

import benchml as bml
import benchml.transforms as btf
from benchml.accumulator import Accumulator
from benchml.hyper import GridHyper, Hyper

log = bml.log


def build_model():
    return btf.Module(
        tag="bmol_gylm_match_attr",
        transforms=[
            btf.ExtXyzInput(tag="input"),
            btf.GylmAtomic(
                tag="descriptor", 
                args={"heavy_only": True}, 
                inputs={"configs": "input.configs"}
            ),
            btf.KernelSmoothMatch(
                tag="kernel", 
                args={"self_kernel": True},
                inputs={"X": "descriptor.X"}
            ),
            btf.GaussianProcess(
                tag="predictor",
                args={"alpha": None, "power": 2},
                inputs={"K": "kernel.K", "K_diag": "kernel.K_diag", "y": "input.y"}
            ),
            btf.AttributeSmoothMatchKernelRidge(
                tag="attribute",
                args={"write_xyz": ""},
                inputs={
                    "configs": "input.configs",
                    "X": "kernel._X",
                    "X_probe": "descriptor.X",
                    "w": "predictor._w",
                    "y_mean": "predictor._y_mean",
                    "y_std": "predictor._y_std",
                },
            ),
        ],
        hyper=GridHyper(
            Hyper({"predictor.alpha": np.logspace(-9, +7, 17)}),
            Hyper({"predictor.power": [2.0]}),
        ),
        broadcast={"meta": "input.meta"},
        outputs={"y": "predictor.y", "dy": "predictor.dy", "Z": "attribute.Z"},
    )


def benchmark_model(model, datafile, split=None, hypersplit=None):
    # Prepare
    data = list(bml.data.DatasetIterator(meta_json=datafile)).pop(0)
    if split is None:
        split = dict(method="random", train_fraction=0.9, n_splits=10)
    if hypersplit is None:
        hypersplit = dict(method="random", n_splits=5, train_fraction=0.75)
    log << log.mg << "Benchmark" << log.endl
    accu = Accumulator(metrics=data["metrics"])

    # Precompute
    stream = model.open(data)
    model.precompute(stream, verbose=True)

    # Split and evaluate
    for split_idx, (train, test) in enumerate(stream.split(**split)):
        log << log.mb << f"Begin split #{split_idx}" << log.endl
        model.hyperfit(
            stream=train,
            split_args=hypersplit,
            accu_args=dict(metric=data["metrics"][0]),
            target="y",
            target_ref="input.y",
            log=log,
        )
        output_train = model.map(train)
        output_test = model.map(test)
        accu.append("test", output_test["y"], test.resolve("input.y"))
        accu.append("train", output_train["y"], train.resolve("input.y"))

    log << log.mg << "Performance" << log.endl
    accu.evaluateAll(metrics=data["metrics"], bootstrap=100, log=log)
    model.close(stream, check=False)
    return model


def train_model(model, datafile, hypersplit_args=None):
    data = list(bml.data.DatasetIterator(meta_json=datafile)).pop(0)
    model = build_model()
    if hypersplit_args is None:
        hypersplit_args = dict(method="random", n_splits=5, train_fraction=0.75)
    log << log.mg << "Fit" << log.endl
    with bml.stream(model, data) as stream:
        with bml.hupdate(model, {"attribute.pass": True}):
            model.hyperfit(
                stream=stream,
                split_args=hypersplit_args,
                accu_args=dict(metric=data["metrics"][0]),
                target="y",
                target_ref="input.y",
                log=log,
            )
    return model


def apply_model(model, smi, num_confs=1):
    # Generate conformer
    mol = chem.MolFromSmiles(smi)
    mol = chem.AddHs(mol)
    conf_gen = rdDistGeom.ETKDGv3()
    conf_gen.randomSeed = 0xF00D
    rdDistGeom.EmbedMultipleConfs(mol, num_confs, conf_gen)

    # Mol to config to data
    configs = []
    syms = [_.GetSymbol() for _ in mol.GetAtoms()]
    for conf in mol.GetConformers():
        config = bml.readwrite.ExtendedXyz(symbols=syms, pos=conf.GetPositions())
        configs.append(config)
    data = bml.data.Dataset(meta={}, configs=configs)
    out = model.map(data)
    return out["y"], out["dy"], out["Z"]


if __name__ == "__main__":
    bml.splits.synchronize(731)
    model = build_model()
    model = benchmark_model(model, "data/dataset_example.json")
    model = train_model(model, "data/dataset_example.json")
    bml.save("model_example.arch", model)

    model = bml.load("model_example.arch")
    smi = "CCn1cc(NC(=O)C2CCC2)cn1"
    value, error, attribution = apply_model(model, smi=smi)
    print("Prediction on", smi)
    print("  Value (total)     =", value[0], "+/-", error[0])
    print("  Attribution       =", attribution[0,0:3], "...")
    print("  Attribution (sum) =", attribution.sum())
