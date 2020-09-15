import numpy as np
from ..transforms import *

def compile_physchem(custom_fields=[], with_hyper=False, **kwargs):
    models = []
    for descriptor_set in ["basic", "core", "logp", "extended"]:
        models.extend([
            Module(
                tag="bmol_physchem_%s_rf" % descriptor_set,
                transforms=[
                    ExtXyzInput(tag="input"),
                    Physchem2D(tag="Physchem2D",
                        args={"select_predef": descriptor_set},
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
            Module(
                tag="bmol_physchem_%s_rr" % descriptor_set,
                transforms=[
                    ExtXyzInput(tag="input"),
                    Physchem2D(tag="Physchem2D",
                        args={"select_predef": descriptor_set},
                        inputs={"configs": "input.configs"}),
                    PhyschemUser(tag="PhyschemUser",
                        args={
                            "fields": custom_fields
                        },
                        inputs={"configs": "input.configs"}),
                    Concatenate(tag="desc",
                        inputs={"X": [ "Physchem2D.X", "PhyschemUser.X" ]}),
                    WhitenMatrix(tag="whiten",
                        inputs={"X": "desc.X"}),
                    Ridge(tag="pred",
                        args={"alpha": None},
                        inputs={"X": "desc.X", "y": "input.y"})
                ],
                hyper=GridHyper(
                    Hyper({ "pred.alpha": np.logspace(-5,+5, 7), })),
                broadcast={"meta": "input.meta"},
                outputs={"y": "pred.y"}),
        ])
    return models

def compile_acsf(adjust_to_species=["C", "N", "O"], *args, **kwargs):
    models = []
    for scalerange, scale in zip(
        [ 1.0, 1.2, 1.8 ],
        [ "minimal", "smart", "longrange" ]):
        models.extend([
            Module(
                tag="bmol_acsf_%s_rr" % scale,
                transforms=[
                    ExtXyzInput(tag="input"),
                    UniversalDscribeACSF(
                        tag="descriptor",
                        args={
                            "adjust_to_species": adjust_to_species,
                            "scalerange": scalerange},
                        inputs={"configs": "input.configs"}),
                    ReduceMatrix(
                        tag="reduce",
                        inputs={"X": "descriptor.X"}),
                    WhitenMatrix(
                        tag="whiten",
                        inputs={
                            "X": "reduce.X"
                        }),
                    Ridge(
                        tag="predictor",
                        inputs={"X": "reduce.X", "y": "input.y"}) ],
                hyper=GridHyper(
                    Hyper({ "predictor.alpha": np.logspace(-5,+5, 7), })),
                broadcast={"meta": "input.meta"},
                outputs={ "y": "predictor.y" }),
            Module(
                tag="bmol_acsf_%s_krr" % scale,
                transforms=[
                    ExtXyzInput(tag="input"),
                    UniversalDscribeACSF(
                        tag="descriptor",
                        args={
                            "adjust_to_species": adjust_to_species,
                            "scalerange": scalerange},
                        inputs={"configs": "input.configs"}),
                    ReduceMatrix(
                        tag="reduce",
                        inputs={"X": "descriptor.X"}),
                    KernelDot(
                        tag="kernel",
                        inputs={
                            "X": "reduce.X"
                        }),
                    KernelRidge(
                        tag="predictor",
                        args={
                            "alpha": None
                        },
                        inputs={
                            "K": "kernel.K",
                            "y": "input.y"
                        }) ],
                hyper=GridHyper(
                    Hyper({ "predictor.alpha": np.logspace(-7, +7, 15), })),
                broadcast={"meta": "input.meta"},
                outputs={ "y": "predictor.y" })
        ])
    return models

def compile_cm(*args, **kwargs):
    models = []
    for permutation in ["sorted_l2", "eigenspectrum"]:
        models.extend([
            Module(
                tag="bmol_cm_%s_rr" % permutation,
                transforms=[
                    ExtXyzInput(tag="input"),
                    DscribeCM(
                        tag="descriptor",
                        args={
                            "permutation": permutation
                        },
                        inputs={"configs": "input.configs"}),
                    ReduceMatrix(
                        tag="reduce",
                        inputs={"X": "descriptor.X"}),
                    WhitenMatrix(
                        tag="whiten",
                        inputs={
                            "X": "reduce.X"
                        }),
                    Ridge(
                        tag="predictor",
                        inputs={"X": "reduce.X", "y": "input.y"}) ],
                hyper=GridHyper(
                    Hyper({ "predictor.alpha": np.logspace(-5,+5, 7), })),
                broadcast={"meta": "input.meta"},
                outputs={ "y": "predictor.y" }),
            Module(
                tag="bmol_cm_%s_krr" % permutation,
                transforms=[
                    ExtXyzInput(tag="input"),
                    DscribeCM(
                        tag="descriptor",
                        args={
                            "permutation": permutation
                        },
                        inputs={"configs": "input.configs"}),
                    ReduceMatrix(
                        tag="reduce",
                        inputs={"X": "descriptor.X"}),
                    KernelDot(
                        tag="kernel",
                        inputs={
                            "X": "reduce.X"
                        }),
                    KernelRidge(
                        tag="predictor",
                        args={
                            "alpha": None
                        },
                        inputs={
                            "K": "kernel.K",
                            "y": "input.y"
                        }) ],
                hyper=GridHyper(
                    Hyper({ "predictor.alpha": np.logspace(-7, +7, 15), })),
                broadcast={"meta": "input.meta"},
                outputs={ "y": "predictor.y" })
        ])
    return models

def compile_soap(*args, **kwargs):
    krr_hyper = GridHyper(
        Hyper({"descriptor.normalize": [ False ] }),
        Hyper({"descriptor.mode": [ "minimal", "smart", "longrange" ] }),
        Hyper({"descriptor.crossover": [ False, True ] }),
        Hyper({"reduce.reduce_op": [ "sum" ]}),
        Hyper({"reduce.normalize": [ True ]}),
        Hyper({"reduce.reduce_by_type": [ False ]}),
        Hyper({"whiten.centre": [ False ]}),
        Hyper({"whiten.scale":  [ False ]}),
        Hyper({"predictor.power": [ 2 ] }))
    rr_hyper = GridHyper(
        Hyper({"descriptor.normalize": [ True ] }),
        Hyper({"descriptor.mode": [ "minimal", "smart", "longrange" ] }),
        Hyper({"descriptor.crossover": [ False, True ] }),
        Hyper({"reduce.reduce_op": [ "mean" ]}),    
        Hyper({"reduce.normalize": [ True ]}),
        Hyper({"reduce.reduce_by_type": [ False ]}),
        Hyper({"whiten.centre": [ True ]}),         
        Hyper({"whiten.scale":  [ True ]}))         
    models = []
    for hidx, updates in enumerate(krr_hyper):
        tag = "%s_%s" % (
            updates["descriptor.mode"], 
            "cross" if updates["descriptor.crossover"] else "nocross")
        model = make_soap_krr(tag="bmol_soap_%s_krr" % tag)
        model.hyperUpdate(updates)
        models.append(model)
    for hidx, updates in enumerate(rr_hyper):
        tag = "%s_%s" % (
            updates["descriptor.mode"], 
            "cross" if updates["descriptor.crossover"] else "nocross")
        model = make_soap_rr(tag="bmol_soap_%s_rr" % tag)
        model.hyperUpdate(updates)
        models.append(model)
    return models

def compile_soap_alt(*args, **kwargs):
    models = []
    rr_hyper = GridHyper(
        Hyper({"descriptor.normalize": [ False ] }),
        Hyper({"descriptor.mode": [ "minimal", "smart", "longrange" ] }),
        Hyper({"descriptor.crossover": [ False, True ] }),
        Hyper({"reduce.reduce_op": [ "mean" ]}),
        Hyper({"reduce.normalize": [ False ]}),
        Hyper({"reduce.reduce_by_type": [ False ]}),
        Hyper({"whiten.centre": [ True ]}),         
        Hyper({"whiten.scale":  [ True ]}))
    for hidx, updates in enumerate(rr_hyper):
        tag = "%s_%s" % (
            updates["descriptor.mode"], 
            "cross" if updates["descriptor.crossover"] else "nocross")
        model = make_soap_alt_rr(tag="bmol_soap_alt_%s_rr" % tag)
        model.hyperUpdate(updates)
        models.append(model)
    return models

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

def make_soap_alt_rr(tag):
    return Module(
        tag=tag,
        transforms=[
            ExtXyzInput(
                tag="input"),
            UniversalSoapDscribe(
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

def make_soap_rr(tag):
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
        for DescriptorClass in [ DscribeMBTR, DscribeLMBTR ]
    ]

def compile_mbtr(**kwargs):
    models = []
    for _ in ["default"]:
        models.extend([
            Module(
                tag="bmol_mbtr_rr",
                transforms=[
                    ExtXyzInput(tag="input"),
                    DscribeMBTR(
                        tag="descriptor",
                        args={
                        },
                        inputs={"configs": "input.configs"}),
                    ReduceMatrix(
                        tag="reduce",
                        inputs={"X": "descriptor.X"}),
                    WhitenMatrix(
                        tag="whiten",
                        inputs={
                            "X": "reduce.X"
                        }),
                    Ridge(
                        tag="predictor",
                        inputs={"X": "reduce.X", "y": "input.y"}) ],
                hyper=GridHyper(
                    Hyper({ "predictor.alpha": np.logspace(-5,+5, 7), })),
                broadcast={"meta": "input.meta"},
                outputs={ "y": "predictor.y" }),
            Module(
                tag="bmol_mbtr_krr",
                transforms=[
                    ExtXyzInput(tag="input"),
                    DscribeMBTR(
                        tag="descriptor",
                        args={
                        },
                        inputs={"configs": "input.configs"}),
                    ReduceMatrix(
                        tag="reduce",
                        inputs={"X": "descriptor.X"}),
                    KernelDot(
                        tag="kernel",
                        inputs={
                            "X": "reduce.X"
                        }),
                    KernelRidge(
                        tag="predictor",
                        args={
                            "alpha": None
                        },
                        inputs={
                            "K": "kernel.K",
                            "y": "input.y"
                        }) ],
                hyper=GridHyper(
                    Hyper({ "predictor.alpha": np.logspace(-7, +7, 15), })),
                broadcast={"meta": "input.meta"},
                outputs={ "y": "predictor.y" })
        ])
    return models
    

def compile_ecfp(**kwargs):
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
                    tag="kernel",
                    inputs={"X": "desc.X"}),
                KernelRidge(
                    args={"alpha": 1e-5, "power": 2},
                    inputs={"K": "kernel.K", "y": "input.y"})
            ],
            hyper=GridHyper(
                Hyper({ "KernelRidge.alpha": np.logspace(-6,+1, 8), }),
                Hyper({ "KernelRidge.power": [ 2. ] })),
            broadcast={ "meta": "input.meta" },
            outputs={ "y": "KernelRidge.y" }
        ),
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
                Hyper({"Ridge.alpha": np.linspace(-5,5,7)}),
                convert={
                    "Ridge.alpha": "lambda p: 10**p"}),
            outputs={"y": "Ridge.y"}
        )
    ]

def compile_gylm(**kwargs):
    return [
        Module(
            tag="bmol_gylm_grid_krr",
            transforms=[
                ExtXyzInput(tag="input"),
                GylmAverage(
                    tag="desc",
                    inputs={"configs": "input.configs"}),
                KernelDot(
                    tag="kernel",
                    inputs={"X": "desc.X"}),
                KernelRidge(
                    args={"alpha": 1e-5, "power": 2},
                    inputs={"K": "kernel.K", "y": "input.y"})
            ],
            hyper=GridHyper(
                Hyper({ "KernelRidge.alpha": np.logspace(-5,+1, 7), }),
                Hyper({ "KernelRidge.power": [ 2. ] })),
            broadcast={ "meta": "input.meta" },
            outputs={ "y": "KernelRidge.y" }
        ),
        Module(
            tag="bmol_gylm_bayes_krr",
            transforms=[
                ExtXyzInput(tag="input"),
                GylmAverage(
                    tag="desc",
                    inputs={"configs": "input.configs"}),
                KernelDot(
                    tag="kernel",
                    inputs={"X": "desc.X"}),
                KernelRidge(
                    args={"alpha": 1e-5, "power": 2},
                    inputs={"K": "kernel.K", "y": "input.y"})
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

def register_all():
    return {
        "bmol_physchem": compile_physchem,
        "bmol_ecfp": compile_ecfp,
        "bmol_cm": compile_cm,
        "bmol_acsf": compile_acsf,
        "bmol_mbtr": compile_mbtr,
        "bmol_soap": compile_soap,
        "bmol_soap_alt": compile_soap_alt,
        "bmol_gylm": compile_gylm,
    }
