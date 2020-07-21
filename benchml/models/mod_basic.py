import numpy as np
from ..transforms import *

def compile(groups, **kwargs):
    selected = [ model \
        for group in groups \
            for model in collections[group](**kwargs) ]
    return selected

def compile_null(**kwargs):
    return []

def compile_physchem(custom_fields=[], with_hyper=False, **kwargs):
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
    else:
        hyper=GridHyper(
            Hyper({"pred.max_depth": [None]}))
    return [
        Module(
            tag="physchem",
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

def compile_soap(basic=False, **kwargs):
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
        model = make_soap_krr(tag="soap_krr_%02d" % hidx)
        model.hyperUpdate(updates)
        models.append(model)
    return models

def compile_dscribe(**kwargs):
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
        for DescriptorClass in [ DscribeCM, DscribeACSF, DscribeMBTR, DscribeLMBTR ]
    ]

def compile_dscribe_periodic(**kwargs):
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
        for DescriptorClass in [ DscribeSineMatrix ]
    ]

def compile_asap(**kwargs):
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

def compile_morgan_krr(**kwargs):
    return [
        Module(
            tag="morgan_krr",
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

def compile_morgan(**kwargs):
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
            tag="morgan_krr_ext",
            transforms=[
                ExtXyzInput(tag="input"),
                MorganFP(
                    tag="desc",
                    args={"length": 4096, "radius": 2},
                    inputs={"configs": "input.configs"}),
                KernelDot(
                    tag="kern",
                    inputs={"X": "desc.X"}),
                KernelRidge(
                    args={"alpha": 1e-5, "power": 2},
                    inputs={"K": "kern.K", "y": "input.y"})
            ],
            hyper=GridHyper(
                Hyper({ "desc.radius": [ 1, 2, 3, 4] }),
                Hyper({ "KernelRidge.alpha": np.logspace(-5,+1, 7), }),
                Hyper({ "KernelRidge.power": [ 2. ] })),
            # >>> hyper=BayesianHyper(
            # >>>     Hyper({ "KernelRidge.alpha": np.linspace(-3,+1, 5), }),
            # >>>     Hyper({ "KernelRidge.power": [ 1., 4. ] }),
            # >>>     n_iter=40,
            # >>>     init_points=10,
            # >>>     convert={
            # >>>         "KernelRidge.alpha": "lambda p: 10**p"
            # >>>     }),
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
                    "Ridge.alpha": "lambda p: 10**p"}),
            outputs={"y": "Ridge.y"}
        ),
        Module(
            tag="morgan_gb",
            transforms=[
                ExtXyzInput(tag="input"),
                MorganFP(
                    args={"length": 2048},
                    inputs={"configs": "input.configs"}),
                GradientBoosting(inputs={"X": "MorganFP.X", "y": "input.y"})
            ],
            hyper=GridHyper(
                Hyper({"GradientBoosting.max_depth": [1,3,5]})
            ),
            outputs={"y": "GradientBoosting.y"}
        ),
    ]

def compile_gylm_match(**kwargs):
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

def compile_gylm(**kwargs):
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

def compile_gylm_grid(**kwargs):
    return [
        Module(
            tag="gylm_grid",
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
        "asap": compile_asap,
        "dscribe": compile_dscribe,
        "dscribe_periodic": compile_dscribe_periodic,
        "ecfp": compile_morgan,
        "gylm": compile_gylm,
        "gylm_match": compile_gylm_match,
        "gylm_grid": compile_gylm_grid,
        "morgan_krr": compile_morgan_krr,
        "null": compile_null,
        "physchem": compile_physchem,
        "soap": compile_soap
    }
