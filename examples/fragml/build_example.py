import numpy as np
import benchml as bml
import benchml.transforms as btf
import rdkit.Chem as chem
log = bml.log

from rdkit.Chem import rdDistGeom
from benchml.hyper import Hyper, GridHyper
from benchml.accumulator import Accumulator


def build_model():
    """Model architecture as described in https://arxiv.org/abs/2204.06348.
    (Note that the pipeline does not include any attribution rank-filtering 
    techniques.)
    """
    return btf.Module(
        tag="bmol_gylm_match_class_attr",
        transforms=[
            btf.ExtXyzInput(
                tag="input"
            ),
            btf.GylmAtomic(
                tag="descriptor", 
                args={"heavy_only": True}, 
                inputs={"configs": "input.configs"}
            ),
            btf.KernelSmoothMatch(
                tag="kernel", 
                inputs={"X": "descriptor.X"}
            ),
            btf.SupportVectorClassifier(
                tag="predictor",
                args={"C": None, "power": 2},
                inputs={"K": "kernel.K", "y": "input.y"},
            ),
            btf.AttributeKernelSmoothMatchSVM(
                tag="attribute",
                args={"write_xyz": "attribution.xyz"},
                inputs={
                    "configs": "input.configs",
                    "X": "kernel._X",
                    "X_probe": "descriptor.X",
                    "model": "predictor._model",
                },
            ),
        ],
        hyper=GridHyper(
            Hyper({"predictor.C": np.logspace(-9, +7, 17)}),
            Hyper({"predictor.power": [2.0]}),
        ),
        broadcast={"meta": "input.meta"},
        outputs={
            "z": "predictor.z", 
            "y": "predictor.y", 
            "Z": "attribute.Z"
        },
    )


def benchmark_model(
        model,
        datafile,
        split=None,
        hypersplit=None
):
    """A barebone benchmarking routine (AUC only) with a 
    nested grid-based hyperparameter search
    """
    data = list(bml.data.DatasetIterator(meta_json=datafile)).pop(0)

    if split is None:
        split = dict(method="random", train_fraction=0.9, n_splits=10)

    if hypersplit is None:
        hypersplit = dict(method="random", n_splits=5, train_fraction=0.75)

    log << log.mg << "Benchmark" << log.endl
    accu = Accumulator(metrics=data["metrics"])
    stream = model.open(data)
    model.precompute(stream, verbose=True)
    for split_idx, (train, test) in enumerate(stream.split(**split)):
        log << log.mb << f"Begin split #{split_idx}" << log.endl
        model.hyperfit(
            stream=train,
            split_args=hypersplit,
            accu_args=dict(metric=data["metrics"][0]),
            target="y",
            target_ref="input.y",
            log=log
        )
        output_train = model.map(train)
        output_test = model.map(test)
        accu.append("test", output_test["z"], test.resolve("input.y"))
        accu.append("train", output_train["z"], train.resolve("input.y"))

    log << log.mg << "Performance" << log.endl
    perf = accu.evaluateAll(metrics=data["metrics"], bootstrap=100, log=log)
    model.close(stream, check=False)



def train_model(
        model,
        datafile,
        hypersplit_args=None
):
    """Training routine with basic hyperparameter search
    """
    data = list(bml.data.DatasetIterator(meta_json=datafile)).pop(0)
    model = build_model()
    if hypersplit_args is None:
        hypersplit_args = dict(method="random", n_splits=5, train_fraction=0.75)
    log << log.mg << "Fit" << log.endl
    with bml.stream(model, data) as stream:
        # vvv To bypass attribution during training
        with bml.hupdate(model, {"attribute.pass": True}): 
            model.hyperfit(
                stream=stream,
                split_args=hypersplit_args,
                accu_args=dict(metric=data["metrics"][0]),
                target="y",
                target_ref="input.y",
                log=log
            )
    return model


def test_on_smiles(model, smi, num_confs=1):
    """Applies model to a molecule in a test setting
    """
    # Generate conformer
    mol = chem.MolFromSmiles(smi)
    mol = chem.AddHs(mol)
    conf_gen = rdDistGeom.ETKDGv3()
    conf_gen.randomSeed = 0xf00d
    cids = rdDistGeom.EmbedMultipleConfs(mol, num_confs, conf_gen)

    # Mol to data
    configs = []
    syms = [ _.GetSymbol() for _ in mol.GetAtoms() ]
    for conf in mol.GetConformers():
        config = bml.readwrite.ExtendedXyz(symbols=syms, pos=conf.GetPositions())
        configs.append(config)
    
    data = bml.data.Dataset(
        meta={}, configs=configs
    )

    with bml.stream(model, data) as stream:
        out = model.map(stream)

    return out["y"], out["z"], out["Z"]


if __name__ == "__main__":
    bml.splits.synchronize(731)
    model = build_model()
    model = benchmark_model(model, "data/dataset_example.json")
    model = train_model(model, "data/dataset_example.json")
    bml.save("model_example.arch", model)

    smi = "CCn1cc(NC(=O)C2CCC2)cn1"
    label, score, attribution = test_on_smiles(model, smi=smi)
    print("Prediction on", smi)
    print("  Label, score =", label, score)
    print("  Attribution  =", attribution)

