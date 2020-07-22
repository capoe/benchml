import numpy as np
from ..transforms import *

def compile_physchem(custom_fields=[], with_hyper=False, **kwargs):
    return [
        Module(
            tag="bmol_physchem",
            transforms=[
                ExtXyzInput(tag="input"),
                Physchem2D(tag="Physchem2D",
                    inputs={"configs": "input.configs"}),
                PhyschemUser(tag="PhyschemUser",
                    args={
                        "fields": custom_fields
                    },
                    inputs={"configs": "input.configs"}),
                Concatenate(tag="desc",
                    inputs={"X": [ "Physchem2D.X", "PhyschemUser.X" ]}),
                RandomForestRegressor(tag="pred",
                    inputs={"X": "desc.X", "y": "input.y"})
            ],
            hyper=GridHyper(
                Hyper({"pred.max_depth": [None]})),
            broadcast={"meta": "input.meta"},
            outputs={"y": "pred.y"}),
    ]

def make_soap_krr(tag):
    return Module(
        tag=tag,
        transforms=[
            ExtXyzInput(
                tag="input"),
            UniversalSoapGylmxx(
                tag="descriptor",
                inputs={
                    "configs": "input.configs"
                }),
            ReduceTypedMatrix(
                tag="reduce",
                inputs={
                    "X": "descriptor.X",
                    "T": "descriptor.T"
                }),
            WhitenMatrix(
                tag="whiten",
                inputs={
                    "X": "reduce.X"
                }),
            KernelDot(
                tag="kernel",
                inputs={
                    "X": "whiten.X"
                }),
            KernelRidge(
                tag="predictor",
                args={
                    "alpha": None
                },
                inputs={
                    "K": "kernel.K",
                    "y": "input.y"
                })
        ],
        hyper=GridHyper(
            Hyper({ "predictor.alpha": np.logspace(-7, +7, 15), })),
        broadcast={"meta": "input.meta"},
        outputs={ "y": "predictor.y" })

def compile_soap_krr(basic=False, **kwargs):
    if basic:
        hyper = GridHyper(
            Hyper({"descriptor.normalize": [ False ] }),
            Hyper({"descriptor.mode": [ "minimal" ] }),
            Hyper({"descriptor.crossover": [ True ] }),
            Hyper({"reduce.reduce_op": [ "sum" ]}),
            Hyper({"reduce.normalize": [ True ]}),
            Hyper({"reduce.reduce_by_type": [ False ]}),
            Hyper({"whiten.centre": [ False ]}),
            Hyper({"whiten.scale":  [ False ]}),
            Hyper({"predictor.power": [ 2 ] }))
    else:
        hyper = GridHyper(
            Hyper({"descriptor.normalize": [ True ] }),
            Hyper({"descriptor.mode": [ "minimal", "smart", "longrange" ] }),
            Hyper({"descriptor.crossover": [ False, True ] }),
            Hyper({"reduce.reduce_op": [ "mean" ]}),     # + "sum"
            Hyper({"reduce.normalize": [ True ]}),
            Hyper({"reduce.reduce_by_type": [ False ]}), # + True
            Hyper({"whiten.centre": [ False ]}),         # + True
            Hyper({"whiten.scale":  [ False ]}),         # + True
            Hyper({"predictor.power": [ 2 ] }))
    models = []
    for hidx, updates in enumerate(hyper):
        model = make_soap_krr(tag="bmol_soap_krr_%02d" % hidx)
        model.hyperUpdate(updates)
        models.append(model)
    return models

def make_soap_ridge(tag):
    return Module(
        tag=tag,
        transforms=[
            ExtXyzInput(
                tag="input"),
            UniversalSoapGylmxx(
                tag="descriptor",
                inputs={
                    "configs": "input.configs"
                }),
            ReduceTypedMatrix(
                tag="reduce",
                inputs={
                    "X": "descriptor.X",
                    "T": "descriptor.T"
                }),
            WhitenMatrix(
                tag="whiten",
                inputs={
                    "X": "reduce.X"
                }),
            Ridge(
                tag="predictor",
                inputs={"X": "reduce.X", "y": "input.y"}) ],
        hyper=GridHyper(
            Hyper({ "predictor.alpha": np.logspace(-7, +7, 15), })),
        broadcast={"meta": "input.meta"},
        outputs={ "y": "predictor.y" })

def compile_soap_rr(basic=False, **kwargs):
    if basic:
        hyper = GridHyper(
            Hyper({"descriptor.normalize": [ False ] }),
            Hyper({"descriptor.mode": [ "minimal" ] }),
            Hyper({"descriptor.crossover": [ True ] }),
            Hyper({"reduce.reduce_op": [ "sum" ]}),
            Hyper({"reduce.normalize": [ True ]}),
            Hyper({"reduce.reduce_by_type": [ False ]}),
            Hyper({"whiten.centre": [ False ]}),
            Hyper({"whiten.scale":  [ False ]}))
    else:
        hyper = GridHyper(
            Hyper({"descriptor.normalize": [ True ] }),
            Hyper({"descriptor.mode": [ "minimal", "smart", "longrange" ] }),
            Hyper({"descriptor.crossover": [ False, True ] }),
            Hyper({"reduce.reduce_op": [ "mean" ]}),     # + "sum"
            Hyper({"reduce.normalize": [ True ]}),
            Hyper({"reduce.reduce_by_type": [ False ]}), # + True
            Hyper({"whiten.centre": [ False ]}),         # + True
            Hyper({"whiten.scale":  [ False ]}))         # + True
    models = []
    for hidx, updates in enumerate(hyper):
        model = make_soap_krr(tag="bmol_soap_rr_%02d" % hidx)
        model.hyperUpdate(updates)
        models.append(model)
    return models

def compile_dscribe(**kwargs):
    return [
        Module(
            tag="bmol_"+DescriptorClass.__name__.lower()+"_ridge",
            transforms=[
                ExtXyzInput(tag="input"),
                DescriptorClass(
                    tag="descriptor",
                    inputs={"configs": "input.configs"}),
                ReduceMatrix(
                    tag="reduce",
                    inputs={"X": "descriptor.X"}),
                Ridge(
                    tag="predictor",
                    inputs={"X": "reduce.X", "y": "input.y"}) ],
            hyper=GridHyper(
                Hyper({ "predictor.alpha": np.logspace(-5,+5, 7), })),
            broadcast={"meta": "input.meta"},
            outputs={ "y": "predictor.y" }) \
        for DescriptorClass in [ DscribeCM, DscribeACSF, DscribeMBTR, DscribeLMBTR ]
    ]

def compile_ecfp_krr(**kwargs):
    return [
        Module(
            tag="bmol_ecfp_krr",
            transforms=[
                ExtXyzInput(tag="input"),
                MorganFP(
                    tag="desc",
                    args={"length": 4096, "radius": 2, "normalize": True},
                    inputs={"configs": "input.configs"}),
                KernelDot(
                    tag="kern",
                    inputs={"X": "desc.X"}),
                KernelRidge(
                    args={"alpha": 1e-5, "power": 2},
                    inputs={"K": "kern.K", "y": "input.y"})
            ],
            hyper=GridHyper(
                Hyper({ "KernelRidge.alpha": np.logspace(-6,+1, 8), }),
                Hyper({ "KernelRidge.power": [ 2. ] })),
            broadcast={ "meta": "input.meta" },
            outputs={ "y": "KernelRidge.y" }
        )
    ]

def compile_ecfp_rr(**kwargs):
    return [
        Module(
            tag="bmol_ecfp_rr",
            transforms=[
                ExtXyzInput(tag="input"),
                MorganFP(
                    args={"length": 2048, "radius": 2, "normalize": True},
                    inputs={"configs": "input.configs"}),
                Ridge(inputs={"X": "MorganFP.X", "y": "input.y"})
            ],
            hyper=BayesianHyper(
                Hyper({"Ridge.alpha": np.linspace(-2,2,5)}),
                convert={
                    "Ridge.alpha": "lambda p: 10**p"}),
            outputs={"y": "Ridge.y"}
        )
    ]

def compile_gylm_bay(**kwargs):
    return [
        Module(
            tag="bmol_gylm_bay_krr",
            transforms=[
                ExtXyzInput(tag="input"),
                GylmAverage(
                    tag="desc",
                    inputs={"configs": "input.configs"}),
                KernelDot(
                    inputs={"X": "desc.X"}),
                KernelRidge(
                    args={"alpha": 1e-5, "power": 2},
                    inputs={"K": "KernelDot.K", "y": "input.y"})
            ],
            hyper=BayesianHyper(
                Hyper({ "KernelRidge.alpha": np.linspace(-5,+1, 7), }),
                Hyper({ "KernelRidge.power": [ 1., 4. ] }),
                init_points=10,
                n_iter=20,
                convert={
                    "KernelRidge.alpha": "lambda p: 10**p"}),
            broadcast={ "meta": "input.meta" },
            outputs={ "y": "KernelRidge.y" }
        ),
    ]

def compile_gylm_grid(**kwargs):
    return [
        Module(
            tag="bmol_gylm_grid_krr",
            transforms=[
                ExtXyzInput(tag="input"),
                GylmAverage(
                    tag="desc",
                    inputs={"configs": "input.configs"}),
                KernelDot(
                    inputs={"X": "desc.X"}),
                KernelRidge(
                    args={"alpha": 1e-5, "power": 2},
                    inputs={"K": "KernelDot.K", "y": "input.y"})
            ],
            hyper=GridHyper(
                Hyper({ "KernelRidge.alpha": np.logspace(-5,+1, 7), }),
                Hyper({ "KernelRidge.power": [ 2. ] }),
                init_points=10,
                n_iter=30,
                convert={
                    "KernelRidge.alpha": "lambda p: 10**p"}),
            broadcast={ "meta": "input.meta" },
            outputs={ "y": "KernelRidge.y" }
        ),
    ]

def register_all():
    return {
        "bmol_physchem": compile_physchem,
        "bmol_ecfp_rr": compile_ecfp_rr,
        "bmol_ecfp_krr": compile_ecfp_krr,
        "bmol_dscribe": compile_dscribe,
        "bmol_soap_krr": compile_soap_krr,
        "bmol_soap_rr": compile_soap_rr,
        "bmol_gylm_bay": compile_gylm_bay,
        "bmol_gylm_grid": compile_gylm_grid,
    }
