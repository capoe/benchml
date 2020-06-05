import numpy as np
from ..transforms import *

def compile(groups):
    selected = [ model \
        for group in groups \
            for model in collections[group]() ]
    return selected

def compile_null():
    return []

def compile_physchem(custom_fields=[], with_hyper=False):
    hyper = None
    if with_hyper:
        hyper=BayesianHyper(
            Hyper({
                "pred.n_estimators": [10, 200],
                "pred.max_depth": [2, 16],
            }),
            convert={
                "pred.n_estimators": "lambda x: int(x)",
                "pred.max_depth": "lambda x: int(x)"
            },
            init_points=10,
            n_iter=30)
    return [
        Module(
            tag="physchem",
            transforms=[
                ExtXyzInput(tag="input"),
                Physchem2D(tag="desc1",
                    inputs={"configs": "input.configs"}),
                PhyschemUser(tag="desc2",
                    args={
                        "fields": custom_fields
                    },
                    inputs={"configs": "input.configs"}),
                Concatenate(tag="desc",
                    inputs={"X": [ "desc1.X", "desc2.X" ]}),
                RandomForestRegressor(tag="pred",
                    inputs={"X": "desc.X", "y": "input.y"})
            ],
            hyper=hyper,
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

def compile_soap():
    hyper = GridHyper(
        Hyper({"descriptor.normalize": [ True ] }),
        Hyper({"descriptor.mode": [ "minimal", "smart", "longrange" ] }),
        Hyper({"descriptor.crossover": [ False, True ] }),
        Hyper({"reduce.reduce_op": [ "sum", "mean" ]}),
        Hyper({"reduce.normalize": [ True ]}),
        Hyper({"reduce.reduce_by_type": [ False, True ]}),
        Hyper({"whiten.centre": [ False, True ]}),
        Hyper({"whiten.scale":  [ False, True ]}),
        Hyper({"predictor.power": [ 2 ] }))
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
    models = []
    for hidx, updates in enumerate(hyper):
        model = make_soap_krr(tag="soap_krr_%02d" % hidx)
        model.hyperUpdate(updates)
        models.append(model)
    return models

def compile_dscribe():
    return [
        Module(
            tag=DescriptorClass.__name__+"_ridge",
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
        for DescriptorClass in [ DscribeSineMatrix, DscribeCM, DscribeSineMatrix, DscribeACSF, DscribeMBTR, DscribeLMBTR ]
    ]

def compile_asap():
    return [
        Module(
            tag="asap_xyz",
            transforms=[
                ExtXyzInput(tag="input"),
                AsapXyz(
                    inputs={"configs": "input.configs"}
                )
            ],
            broadcast={
                "meta": "input.meta"
            })
    ]

def compile_morgan():
    return [
        # Macro example
        # >>> Module(
        # >>>     tag="morgan_krrx2",
        # >>>     transforms=[
        # >>>         ExtXyzInput(tag="input"),
        # >>>         MorganKernel(
        # >>>             tag="A",
        # >>>             args={"x.fp_length": 1024, "x.fp_radius": 2},
        # >>>             inputs={"x.configs": "input.configs"}),
        # >>>         MorganKernel(
        # >>>             tag="B",
        # >>>             args={"x.fp_length": 2048, "x.fp_radius": 4},
        # >>>             inputs={"x.configs": "input.configs"}),
        # >>>         Add(
        # >>>             args={"coeffs": [ 0.5, 0.5 ]},
        # >>>             inputs={"X": ["A/k.K", "B/k.K"]}),
        # >>>         KernelRidge(
        # >>>             args={"alpha": 0.1, "power": 2},
        # >>>             inputs={"K": "Add.y", "y": "input.y"})
        # >>>     ],
        # >>>     hyper=BayesianHyper(
        # >>>         Hyper({ "Add.coeffs":
        # >>>             list(map(lambda f: [ f, 1.-f ], np.linspace(0.25, 0.75, 3)))
        # >>>         }),
        # >>>         Hyper({ "KernelRidge.alpha":
        # >>>             np.linspace(-3,+1, 5),
        # >>>         }),
        # >>>         n_iter=40,
        # >>>         init_points=10,
        # >>>         convert={
        # >>>             "KernelRidge.alpha": lambda p: 10**p}),
        # >>>     broadcast={ "meta": "input.meta" },
        # >>>     outputs={ "y": "KernelRidge.y" },
        # >>> ),
        Module(
            tag="morgan_krr",
            transforms=[
                ExtXyzInput(tag="input"),
                MorganFP(
                    tag="desc",
                    inputs={"configs": "input.configs"}),
                KernelDot(
                    tag="kern",
                    inputs={"X": "desc.X"}),
                KernelRidge(
                    args={"alpha": 1e-5, "power": 2},
                    inputs={"K": "kern.K", "y": "input.y"})
            ],
            hyper=GridHyper(
                Hyper({ "KernelRidge.alpha": np.logspace(-3,+1, 5), }),
                Hyper({ "KernelRidge.power": [ 2. ] })),
            broadcast={ "meta": "input.meta" },
            outputs={ "y": "KernelRidge.y" }
        ),
        Module(
            tag="morgan_ridge",
            transforms=[
                ExtXyzInput(tag="input"),
                MorganFP(
                    args={"length": 2048},
                    inputs={"configs": "input.configs"}),
                Ridge(inputs={"X": "MorganFP.X", "y": "input.y"})
            ],
            hyper=BayesianHyper(
                Hyper({"Ridge.alpha": np.linspace(-2,2,5)}),
                convert={
                    "Ridge.alpha": lambda p: 10**p}),
            outputs={"y": "Ridge.y"}
        ),
    ]

def compile_gylm_match():
    return [
        Module(
            tag="gylm_smooth_match",
            transforms=[
                ExtXyzInput(tag="input"),
                GylmAtomic(
                    tag="desc",
                    inputs={"configs": "input.configs"}),
                KernelSmoothMatch(
                    inputs={"X": "desc.X"}),
                KernelRidge(
                    args={"alpha": 1e-5, "power": 2},
                    inputs={"K": "KernelSmoothMatch.K", "y": "input.y"})
            ],
            hyper=GridHyper(
                Hyper({ "KernelRidge.alpha": np.logspace(-5,+1, 7), }),
                Hyper({ "KernelRidge.power": [ 2. ] })),
            broadcast={ "meta": "input.meta" },
            outputs={ "y": "KernelRidge.y" }
        ),
    ]

def compile_gylm():
    return [
        Module(
            tag="gylm",
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
                n_iter=30,
                convert={
                    "KernelRidge.alpha": "lambda p: 10**p"}),
            broadcast={ "meta": "input.meta" },
            outputs={ "y": "KernelRidge.y" }
        ),
    ]

collections = {
    "asap": compile_asap,
    "dscribe": compile_dscribe,
    "gylm": compile_gylm,
    "gylm_match": compile_gylm_match,
    "morgan": compile_morgan,
    "null": compile_null,
    "physchem": compile_physchem,
    "soap": compile_soap
}
