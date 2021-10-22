import json

import numpy as np

import benchml as bml
import benchml.transforms as btf
from benchml.hyper import GridHyper, Hyper

log = bml.log
log.setLevel("info")
bml.splits.synchronize(0)
regularization_range = np.logspace(-4, +0, 5)


def build_models():
    return [
        build_gylm_krr(),
    ]


def build_gylm_krr():
    return btf.Module(
        tag="gylm_krr",
        transforms=[
            btf.ExtXyzInput(tag="input"),
            btf.GylmAtomic(
                tag="descriptor_atomic",
                args={
                    "normalize": True,
                    "rcut": 7.5,
                    "heavy_only": True,
                    "rcut_width": 0.5,
                    "nmax": 9,
                    "lmax": 7,
                    "sigma": 0.75,
                    "part_sigma": 0.5,
                    "wconstant": False,
                    "wscale": 0.5,
                    "wcentre": 0.5,
                    "ldamp": 0.5,
                    "power": True,
                },
                inputs={"configs": "input.configs"},
            ),
            btf.ReduceTypedMatrix(
                tag="descriptor",
                precompute=True,
                args={
                    "reduce_op": "sum",
                    "normalize": True,
                    "reduce_by_type": False,
                    "types": None,
                    "epsilon": 1e-10,
                },
                inputs={"X": "descriptor_atomic.X", "T": None},
            ),
            btf.KernelDot(tag="kernel", inputs={"X": "descriptor.X"}),
            btf.DoDivideBySize(
                tag="input_norm",
                args={
                    "config_to_size": "lambda c: len(c.getHeavy()[0])",
                    "force": False,
                    "skip_if_not_force": False,
                },
                inputs={"configs": "input.configs", "meta": "input.meta", "y": "input.y"},
            ),
            btf.KernelRidge(
                tag="predictor",
                args={"alpha": None, "power": 4},
                inputs={"K": "kernel.K", "y": "input_norm.y"},
            ),
            btf.UndoDivideBySize(
                tag="output", inputs={"y": "predictor.y", "sizes": "input_norm.sizes"}
            ),
        ],
        hyper=GridHyper(
            Hyper(
                {
                    "predictor.alpha": regularization_range,
                }
            )
        ),
        broadcast={"meta": "input.meta"},
        outputs={"y": "output.y"},
    )


def build_hloops():
    return [
        GridHyper(
            Hyper({"descriptor_atomic.normalize": [False, True]}),
            Hyper({"descriptor_atomic.rcut": [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.5, 6.5, 7.5]}),
            Hyper({"descriptor_atomic.nmax": [6, 7, 8, 9, 10]}),
            Hyper({"descriptor_atomic.lmax": [4, 5, 6, 7]}),
            Hyper({"descriptor_atomic.sigma": [0.25, 0.5, 0.75]}),
            Hyper({"descriptor_atomic.part_sigma": [0.25, 0.5, 0.75]}),
            Hyper({"descriptor_atomic.wconstant": [False, True]}),
            Hyper({"descriptor_atomic.wscale": [0.5, 1.0, 1.5]}),
            Hyper({"descriptor_atomic.wcentre": [0.5, 1.0, 1.5]}),
            Hyper({"descriptor_atomic.ldamp": [0.5, 1.0, 2.0, 4.0]}),
            # Hyper({
            #     "descriptor_atomic.heavy_only": [False, True],
            #     "input_norm.config_to_size": [
            #         "lambda c: len(c)", "lambda c: len(c.getHeavy()[0]"]}),
            Hyper(
                {
                    "descriptor.reduce_op": ["mean", "sum", "sum"],
                    "descriptor.normalize": [False, False, True],
                }
            ),
            Hyper({"predictor.power": [1, 2, 3, 4]}),
        ),
    ]


def evaluate_single(
    dataset,
    model,
    hyper=None,
    split=None,
    hsplit=None,
):
    if split is None:
        split = dict(method="random", n_splits=5, train_fraction=0.9)
    if hsplit is None:
        hsplit = dict(method="random", n_splits=3, train_fraction=0.75)
    log << log.mg << "Model %s" % model.tag << log.endl
    if hyper is not None:
        model.hyperUpdate(hyper, verbose=True)
    stream = model.open(dataset, verbose=True)
    accu = bml.Accumulator(metrics=["mse", "mae", "r2", "rhop"])
    model.precompute(stream, verbose=True)
    for idx, (stream_train, stream_test) in enumerate(stream.split(**split)):
        log << log.back << "  Split %3d" % idx << log.flush
        if idx == 0:
            model.hyperfit(
                stream=stream_train,
                split_args=hsplit,
                accu_args={"metric": dataset["metrics"][0]},
                target="y",
                target_ref="input.y",
                log=log,
            )
        else:
            model.fit(stream_train, verbose=True)
        output_train = model.map(stream_train)
        output_test = model.map(stream_test)
        accu.append("train", output_train["y"], stream_train.resolve("input.y"))
        accu.append("test", output_test["y"], stream_test.resolve("input.y"))
    log << log.endl
    return accu.evaluateAll(log=bml.log, bootstrap=0)


if __name__ == "__main__":
    dataset = bml.load_dataset(
        "data/aqsol_1144.xyz",
        meta={
            "target": "log_sol_mol_l",
            "elements": ["C", "N", "O", "S", "H", "F", "Cl", "Br", "I", "P", "B", "Si"],
            "name": "?",
            "task": "regression",
            "scaling": "non-additive",
            "metrics": ["mse", "mae"],
        },
    )
    n_hyper_settings = 10
    for model, hloop in zip(build_models(), build_hloops()):
        dump = []
        for i in range(n_hyper_settings):
            log << log.mg << "Hyperloop" << i << log.endl
            hargs = hloop.random()
            metrics = evaluate_single(dataset, model, hargs)
            dump.append({"m": metrics, "h": hargs})
        json.dump(dump, open(model.tag + "_metrics_brief.json", "w"), indent=1, sort_keys=True)
