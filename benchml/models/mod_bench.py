import numpy as np
from ..transforms import *

whiten_hyper = [ False ] # NOTE: False = no whitening in ridge models
regularization_range = np.logspace(-9, +7, 17)

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
                    Concatenate(tag="descriptor",
                        inputs={"X": [ "Physchem2D.X", "PhyschemUser.X" ]}),
                    DoDivideBySize(
                        tag="input_norm",
                        args={
                            "config_to_size": "lambda c: len(c)",
                            "skip_if_not_force": True,
                            "force": None
                        },
                        inputs={
                            "configs": "input.configs",
                            "meta": "input.meta",
                            "y": "input.y"
                        }),
                    RandomForestRegressor(tag="predictor",
                        inputs={"X": "descriptor.X", "y": "input_norm.y"}),
                    UndoDivideBySize(
                        tag="output",
                        inputs={"y": "predictor.y", "sizes": "input_norm.sizes"}) 
                ],
                hyper=GridHyper(
                    Hyper({ "input_norm.force": [False, True] }),
                    Hyper({"predictor.max_depth": [None]})),
                broadcast={"meta": "input.meta"},
                outputs={"y": "output.y"}),
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
                    Concatenate(tag="descriptor",
                        inputs={"X": [ "Physchem2D.X", "PhyschemUser.X" ]}),
                    WhitenMatrix(tag="whiten",
                        inputs={"X": "descriptor.X"}),
                    DoDivideBySize(
                        tag="input_norm",
                        args={
                            "config_to_size": "lambda c: len(c)",
                            "skip_if_not_force": True,
                            "force": None
                        },
                        inputs={
                            "configs": "input.configs",
                            "meta": "input.meta",
                            "y": "input.y"
                        }),
                    Ridge(tag="predictor",
                        args={"alpha": None},
                        inputs={"X": "whiten.X", "y": "input_norm.y"}),
                    UndoDivideBySize(
                        tag="output",
                        inputs={"y": "predictor.y", "sizes": "input_norm.sizes"}) 
                ],
                hyper=GridHyper(
                    Hyper({ "input_norm.force": [False, True] }),
                    Hyper({ "predictor.alpha": regularization_range, })),
                broadcast={"meta": "input.meta"},
                outputs={"y": "output.y"}),
        ])
    return models

def compile_ecfp(**kwargs):
    models = []
    for radius in [2, 3]:
        models.extend([
            Module(
                tag="bmol_ecfp%d_krr" % (2*radius),
                transforms=[
                    ExtXyzInput(tag="input"),
                    MorganFP(
                        tag="descriptor",
                        args={"length": 4096, "radius": radius, "normalize": True},
                        inputs={"configs": "input.configs"}),
                    KernelDot(
                        tag="kernel",
                        inputs={"X": "descriptor.X"}),
                    DoDivideBySize(
                        tag="input_norm",
                        args={
                            "config_to_size": "lambda c: len(c)",
                            "skip_if_not_force": False,
                        },
                        inputs={
                            "configs": "input.configs",
                            "meta": "input.meta",
                            "y": "input.y"
                        }),
                    KernelRidge(
                        tag="predictor",
                        args={
                            "alpha": None,
                            "power": 2
                        },
                        inputs={
                            "K": "kernel.K",
                            "y": "input_norm.y"
                        }),
                    UndoDivideBySize(
                        tag="output",
                        inputs={"y": "predictor.y", "sizes": "input_norm.sizes"}) 
                ],
                hyper=GridHyper(
                    Hyper({ "predictor.alpha": regularization_range, }),
                    Hyper({ "predictor.power": [ 2. ] })),
                broadcast={ "meta": "input.meta" },
                outputs={ "y": "output.y" }
            ),
            Module(
                tag="bmol_ecfp%d_rr" % (2*radius),
                transforms=[
                    ExtXyzInput(tag="input"),
                    MorganFP(
                        tag="descriptor",
                        args={"length": 2048, "radius": radius, "normalize": True},
                        inputs={"configs": "input.configs"}),
                    WhitenMatrix(
                        tag="whiten",
                        inputs={
                            "X": "descriptor.X"
                        }),
                    DoDivideBySize(
                        tag="input_norm",
                        args={
                            "config_to_size": "lambda c: len(c)",
                            "skip_if_not_force": False,
                        },
                        inputs={
                            "configs": "input.configs",
                            "meta": "input.meta",
                            "y": "input.y"
                        }),
                    Ridge(
                        tag="predictor",
                        inputs={"X": "whiten.X", "y": "input_norm.y"}),
                    UndoDivideBySize(
                        tag="output",
                        inputs={"y": "predictor.y", "sizes": "input_norm.sizes"}) 
                ],
                hyper=GridHyper(
                    Hyper({ "predictor.alpha": regularization_range, }),
                    Hyper({ "whiten.centre": [ False, True], "whiten.scale": [False, True] })
                ),
                outputs={"y": "output.y"}
            )
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
                        tag="descriptor_atomic",
                        args={
                            "permutation": permutation
                        },
                        inputs={"configs": "input.configs"}),
                    ReduceMatrix(
                        tag="descriptor",
                        args = {
                            "reduce": "np.mean(x, axis=0)",
                            "norm": False,
                            "epsilon": 1e-10 },
                        inputs={"X": "descriptor_atomic.X"}),
                    WhitenMatrix(
                        tag="whiten",
                        inputs={
                            "X": "descriptor.X"
                        }),
                    DoDivideBySize(
                        tag="input_norm",
                        args={
                            "config_to_size": "lambda c: len(c)",
                            "skip_if_not_force": False,
                            "force": None
                        },
                        inputs={
                            "configs": "input.configs",
                            "meta": "input.meta",
                            "y": "input.y"
                        }),
                    Ridge(
                        tag="predictor",
                        inputs={"X": "whiten.X", "y": "input_norm.y"}),
                    UndoDivideBySize(
                        tag="output",
                        inputs={"y": "predictor.y", "sizes": "input_norm.sizes"}) 
                ],
                hyper=GridHyper(
                    Hyper({ "input_norm.force": [False, True] }),
                    Hyper({ "predictor.alpha": regularization_range, })),
                broadcast={"meta": "input.meta"},
                outputs={ "y": "output.y" }),
            Module(
                tag="bmol_cm_%s_krr" % permutation,
                transforms=[
                    ExtXyzInput(tag="input"),
                    DscribeCM(
                        tag="descriptor_atomic",
                        args={
                            "permutation": permutation
                        },
                        inputs={"configs": "input.configs"}),
                    ReduceMatrix(
                        tag="descriptor",
                        args = {
                            "reduce": "np.mean(x, axis=0)",
                            "norm": False,
                            "epsilon": 1e-10 },
                        inputs={"X": "descriptor_atomic.X"}),
                    KernelDot(
                        tag="kernel",
                        inputs={
                            "X": "descriptor.X"
                        }),
                    DoDivideBySize(
                        tag="input_norm",
                        args={
                            "config_to_size": "lambda c: len(c)",
                            "skip_if_not_force": False,
                            "force": None
                        },
                        inputs={
                            "configs": "input.configs",
                            "meta": "input.meta",
                            "y": "input.y"
                        }),
                    KernelRidge(
                        tag="predictor",
                        args={
                            "alpha": None,
                        },
                        inputs={
                            "K": "kernel.K",
                            "y": "input_norm.y"
                        }),
                    UndoDivideBySize(
                        tag="output",
                        inputs={"y": "predictor.y", "sizes": "input_norm.sizes"}) 
                ], 
                hyper=GridHyper(
                    Hyper({ "input_norm.force": [False, True] }),
                    Hyper({ "predictor.alpha": regularization_range, })),
                broadcast={"meta": "input.meta"},
                outputs={ "y": "output.y" })
        ])
    return models

def compile_acsf(adjust_to_species=["C", "N", "O"], *args, **kwargs):
    models = []
    for scalerange, sharpness, scale in zip(
        [ 0.85, 1.2, 1.8 ],
        [ 1.0,  1.0, 1.2 ],
        [ "minimal", "smart", "longrange" ]):
        for extensive in [False, True]:
            models.extend([
                Module(
                    tag="bmol_acsf_%s_%s_rr" % (scale, "ext" if extensive else "int"),
                    transforms=[
                        ExtXyzInput(tag="input"),
                        UniversalDscribeACSF(
                            tag="descriptor_atomic",
                            args={
                                "adjust_to_species": None, # TODO
                                "scalerange": scalerange,
                                "sharpness": sharpness},
                            inputs={"configs": "input.configs"}),
                        ReduceMatrix(
                            tag="descriptor",
                            args = {
                                "reduce": "np.sum(x, axis=0)" if extensive else "np.mean(x, axis=0)",
                                "norm": False,
                                "epsilon": 1e-10 },
                            inputs={"X": "descriptor_atomic.X"}),
                        WhitenMatrix(
                            tag="whiten",
                            inputs={
                                "X": "descriptor.X"
                            }),
                        DoDivideBySize(
                            tag="input_norm",
                            args={
                                "config_to_size": "lambda c: len(c)",
                                "skip_if_not_force": True if extensive else False,
                            },
                            inputs={
                                "configs": "input.configs",
                                "meta": "input.meta",
                                "y": "input.y"
                            }),
                        Ridge(
                            tag="predictor",
                            inputs={"X": "whiten.X", "y": "input_norm.y"}),
                        UndoDivideBySize(
                            tag="output",
                            inputs={"y": "predictor.y", "sizes": "input_norm.sizes"}) 
                    ],
                    hyper=GridHyper(
                        Hyper({ "whiten.centre": whiten_hyper,  
                                "whiten.scale":  whiten_hyper}),
                        Hyper({ "predictor.alpha": regularization_range, })),
                    broadcast={"meta": "input.meta"},
                    outputs={ "y": "output.y" }),
                Module(
                    tag="bmol_acsf_%s_%s_krr" % (scale, "ext" if extensive else "int"),
                    transforms=[
                        ExtXyzInput(tag="input"),
                        UniversalDscribeACSF(
                            tag="descriptor_atomic",
                            args={
                                "adjust_to_species": None, # TODO
                                "scalerange": scalerange},
                            inputs={"configs": "input.configs"}),
                        ReduceMatrix(
                            tag="descriptor",
                            args = {
                                "reduce": "np.sum(x, axis=0)" if extensive else "np.mean(x, axis=0)",
                                "norm": False,
                                "epsilon": 1e-10 },
                            inputs={"X": "descriptor_atomic.X"}),
                        KernelDot(
                            tag="kernel",
                            inputs={
                                "X": "descriptor.X"
                            }),
                        DoDivideBySize(
                            tag="input_norm",
                            args={
                                "config_to_size": "lambda c: len(c)",
                                "skip_if_not_force": True if extensive else False,
                            },
                            inputs={
                                "configs": "input.configs",
                                "meta": "input.meta",
                                "y": "input.y"
                            }),
                        KernelRidge(
                            tag="predictor",
                            args={
                                "alpha": None
                            },
                            inputs={
                                "K": "kernel.K",
                                "y": "input_norm.y"
                            }),
                        UndoDivideBySize(
                            tag="output",
                            inputs={"y": "predictor.y", "sizes": "input_norm.sizes"}) 
                    ],
                    hyper=GridHyper(
                        Hyper({ "predictor.alpha": regularization_range, })),
                    broadcast={"meta": "input.meta"},
                    outputs={ "y": "output.y" })
            ])
    return models

def compile_mbtr(**kwargs):
    models = []
    for _ in ["default"]:
        for extensive in [False, True]:
            models.extend([
                Module(
                    tag="bmol_mbtr_%s_rr" % ("ext" if extensive else "int"),
                    transforms=[
                        ExtXyzInput(tag="input"),
                        DscribeMBTR(
                            tag="descriptor_atomic",
                            args={
                            },
                            inputs={"configs": "input.configs"}),
                        ReduceMatrix(
                            tag="descriptor",
                            args = {
                                "reduce": "np.sum(x, axis=0)" if extensive else "np.mean(x, axis=0)",
                                "norm": False,
                                "epsilon": 1e-10 },
                            inputs={"X": "descriptor_atomic.X"}),
                        WhitenMatrix(
                            tag="whiten",
                            inputs={
                                "X": "descriptor.X"
                            }),
                        DoDivideBySize(
                            tag="input_norm",
                            args={
                                "config_to_size": "lambda c: len(c)",
                                "skip_if_not_force": True if extensive else False,
                            },
                            inputs={
                                "configs": "input.configs",
                                "meta": "input.meta",
                                "y": "input.y"
                            }),
                        Ridge(
                            tag="predictor",
                            inputs={"X": "whiten.X", "y": "input_norm.y"}),
                        UndoDivideBySize(
                            tag="output",
                            inputs={"y": "predictor.y", "sizes": "input_norm.sizes"}) 
                    ],
                    hyper=GridHyper(
                        Hyper({ "whiten.centre": whiten_hyper,   
                                "whiten.scale":  whiten_hyper}), 
                        Hyper({ "predictor.alpha": regularization_range, })),
                    broadcast={"meta": "input.meta"},
                    outputs={ "y": "output.y" }),
                Module(
                    tag="bmol_mbtr_%s_krr" % ("ext" if extensive else "int"),
                    transforms=[
                        ExtXyzInput(tag="input"),
                        DscribeMBTR(
                            tag="descriptor_atomic",
                            args={
                            },
                            inputs={"configs": "input.configs"}),
                        ReduceMatrix(
                            tag="descriptor",
                            args = {
                                "reduce": "np.sum(x, axis=0)" if extensive else "np.mean(x, axis=0)",
                                "norm": False,
                                "epsilon": 1e-10 },
                            inputs={"X": "descriptor_atomic.X"}),
                        KernelDot(
                            tag="kernel",
                            inputs={
                                "X": "descriptor.X"
                            }),
                        DoDivideBySize(
                            tag="input_norm",
                            args={
                                "config_to_size": "lambda c: len(c)",
                                "skip_if_not_force": True if extensive else False,
                            },
                            inputs={
                                "configs": "input.configs",
                                "meta": "input.meta",
                                "y": "input.y"
                            }),
                        KernelRidge(
                            tag="predictor",
                            args={
                                "alpha": None
                            },
                            inputs={
                                "K": "kernel.K",
                                "y": "input_norm.y"
                            }),
                        UndoDivideBySize(
                            tag="output",
                            inputs={"y": "predictor.y", "sizes": "input_norm.sizes"}) 
                    ],
                    hyper=GridHyper(
                        Hyper({ "predictor.alpha": regularization_range, })),
                    broadcast={"meta": "input.meta"},
                    outputs={ "y": "output.y" })
            ])
    return models

def compile_soap(*args, **kwargs):
    krr_int_settings = GridHyper(
        Hyper({"descriptor_atomic.normalize": [ False ] }),
        Hyper({"descriptor_atomic.mode": [ "minimal", "smart", "longrange" ] }),
        Hyper({"descriptor_atomic.crossover": [ False, True ] }),
        Hyper({"descriptor.reduce_op": [ "mean" ]}),
        Hyper({"descriptor.normalize": [ False ]}),
        Hyper({"descriptor.reduce_by_type": [ False ]}),
        Hyper({"whiten.centre": [ False ]}),
        Hyper({"whiten.scale":  [ False ]}),
        Hyper({"predictor.power": [ 2 ] }))
    krr_int_hyper = GridHyper(
        Hyper({"predictor.power": [ 1, 2, 3 ] }))
    krr_ext_settings = GridHyper(
        Hyper({"descriptor_atomic.normalize": [ False ] }),
        Hyper({"descriptor_atomic.mode": [ "minimal", "smart", "longrange" ] }),
        Hyper({"descriptor_atomic.crossover": [ False, True ] }),
        Hyper({"descriptor.reduce_op": [ "sum" ]}),
        Hyper({"descriptor.normalize": [ False ]}),
        Hyper({"descriptor.reduce_by_type": [ False ]}),
        Hyper({"whiten.centre": [ False ]}),
        Hyper({"whiten.scale":  [ False ]}),
        Hyper({"predictor.power": [ 1 ] }))
    krr_ext_hyper = GridHyper(
        Hyper({"predictor.power": [ 1, 2, 3 ] }))
    rr_int_settings = GridHyper(
        Hyper({"descriptor_atomic.normalize": [ False ] }),
        Hyper({"descriptor_atomic.mode": [ "minimal", "smart", "longrange" ] }),
        Hyper({"descriptor_atomic.crossover": [ False, True ] }),
        Hyper({"descriptor.reduce_op": [ "mean" ]}),    
        Hyper({"descriptor.normalize": [ False ]}),
        Hyper({"descriptor.reduce_by_type": [ False ]}),
        Hyper({"whiten.centre": [ True ]}),         
        Hyper({"whiten.scale":  [ True ]}))         
    rr_int_hyper = GridHyper(
        Hyper({
            "whiten.centre": whiten_hyper,         
            "whiten.scale":  whiten_hyper}))         
    rr_ext_settings = GridHyper(
        Hyper({"descriptor_atomic.normalize": [ False ] }),
        Hyper({"descriptor_atomic.mode": [ "minimal", "smart", "longrange" ] }),
        Hyper({"descriptor_atomic.crossover": [ False, True ] }),
        Hyper({"descriptor.reduce_op": [ "sum" ]}),    
        Hyper({"descriptor.normalize": [ False ]}),
        Hyper({"descriptor.reduce_by_type": [ False ]}),
        Hyper({"whiten.centre": [ False ]}),         
        Hyper({"whiten.scale":  [ False ]}))         
    rr_ext_hyper = GridHyper(
        Hyper({
            "whiten.centre": whiten_hyper,         
            "whiten.scale":  whiten_hyper}))         
    models = []
    for hidx, updates in enumerate(krr_int_settings):
        tag = "%s_%s" % (
            updates["descriptor_atomic.mode"], 
            "cross" if updates["descriptor_atomic.crossover"] else "nocross")
        model = make_soap_krr(tag="bmol_soap_%s_int_krr" % tag, extensive=False)
        model.hyperUpdate(updates)
        model.hyper.add(krr_int_hyper)
        models.append(model)
    for hidx, updates in enumerate(krr_ext_settings):
        tag = "%s_%s" % (
            updates["descriptor_atomic.mode"], 
            "cross" if updates["descriptor_atomic.crossover"] else "nocross")
        model = make_soap_krr(tag="bmol_soap_%s_ext_krr" % tag, extensive=True)
        model.hyperUpdate(updates)
        model.hyper.add(krr_ext_hyper)
        models.append(model)
    for hidx, updates in enumerate(rr_int_settings):
        tag = "%s_%s" % (
            updates["descriptor_atomic.mode"], 
            "cross" if updates["descriptor_atomic.crossover"] else "nocross")
        model = make_soap_rr(tag="bmol_soap_%s_int_rr" % tag, extensive=False)
        model.hyperUpdate(updates)
        model.hyper.add(rr_int_hyper)
        models.append(model)
    for hidx, updates in enumerate(rr_ext_settings):
        tag = "%s_%s" % (
            updates["descriptor_atomic.mode"], 
            "cross" if updates["descriptor_atomic.crossover"] else "nocross")
        model = make_soap_rr(tag="bmol_soap_%s_ext_rr" % tag, extensive=True)
        model.hyperUpdate(updates)
        model.hyper.add(rr_ext_hyper)
        models.append(model)
    return models

def make_soap_krr(tag, extensive):
    return Module(
        tag=tag,
        transforms=[
            ExtXyzInput(
                tag="input"),
            UniversalSoapGylmxx(
                tag="descriptor_atomic",
                inputs={
                    "configs": "input.configs"
                }),
            ReduceTypedMatrix(
                tag="descriptor",
                args = {
                    "reduce_op": "np.sum(x, axis=0)",
                    "normalize": False,
                    "reduce_by_type": False,
                    "types": None,
                    "epsilon": 1e-10 },
                inputs={
                    "X": "descriptor_atomic.X",
                    "T": "descriptor_atomic.T"
                }),
            WhitenMatrix(
                tag="whiten",
                inputs={
                    "X": "descriptor.X"
                }),
            KernelDot(
                tag="kernel",
                inputs={
                    "X": "whiten.X"
                }),
            DoDivideBySize(
                tag="input_norm",
                args={
                    "config_to_size": "lambda c: len(c)",
                    "skip_if_not_force": True if extensive else False
                },
                inputs={
                    "configs": "input.configs",
                    "meta": "input.meta",
                    "y": "input.y"
                }),
            KernelRidge(
                tag="predictor",
                args={
                    "alpha": None
                },
                inputs={
                    "K": "kernel.K",
                    "y": "input_norm.y"
                }),
            UndoDivideBySize(
                tag="output",
                inputs={"y": "predictor.y", "sizes": "input_norm.sizes"}) 
        ],
        hyper=GridHyper(
            Hyper({ "predictor.alpha": regularization_range, })),
        broadcast={"meta": "input.meta"},
        outputs={ "y": "output.y" })

def make_soap_rr(tag, extensive):
    return Module(
        tag=tag,
        transforms=[
            ExtXyzInput(
                tag="input"),
            UniversalSoapGylmxx(
                tag="descriptor_atomic",
                inputs={
                    "configs": "input.configs"
                }),
            ReduceTypedMatrix(
                tag="descriptor",
                args = {
                    "reduce_op": "np.sum(x, axis=0)",
                    "normalize": False,
                    "reduce_by_type": False,
                    "types": None,
                    "epsilon": 1e-10 },
                inputs={
                    "X": "descriptor_atomic.X",
                    "T": "descriptor_atomic.T"
                }),
            WhitenMatrix(
                tag="whiten",
                inputs={
                    "X": "descriptor.X"
                }),
            DoDivideBySize(
                tag="input_norm",
                args={
                    "config_to_size": "lambda c: len(c)",
                    "skip_if_not_force": True if extensive else False
                },
                inputs={
                    "configs": "input.configs",
                    "meta": "input.meta",
                    "y": "input.y"
                }),
            Ridge(
                tag="predictor",
                inputs={"X": "whiten.X", "y": "input_norm.y"}),
            UndoDivideBySize(
                tag="output",
                inputs={"y": "predictor.y", "sizes": "input_norm.sizes"}) 
        ],
        hyper=GridHyper(
            Hyper({ "predictor.alpha": regularization_range, })),
        broadcast={"meta": "input.meta"},
        outputs={ "y": "output.y" })

def compile_gylm(*args, **kwargs):
    krr_int_settings = GridHyper(
        Hyper({"descriptor_atomic.normalize": [ False ] }),
        Hyper({"descriptor.reduce_op": [ "mean" ]}),
        Hyper({"descriptor.normalize": [ False ]}),
        Hyper({"descriptor.reduce_by_type": [ False ]}),
        Hyper({"predictor.power": [ 2 ] }))
    krr_int_hyper = GridHyper(
        Hyper({"predictor.power": [ 1, 2, 3 ] }))
    krr_ext_settings = GridHyper(
        Hyper({"descriptor_atomic.normalize": [ False ] }),
        Hyper({"descriptor.reduce_op": [ "sum" ]}),
        Hyper({"descriptor.normalize": [ False ]}),
        Hyper({"descriptor.reduce_by_type": [ False ]}),
        Hyper({"predictor.power": [ 1 ] }))
    krr_ext_hyper = GridHyper(
        Hyper({"predictor.power": [ 1, 2, 3 ] }))
    rr_int_settings = GridHyper(
        Hyper({"descriptor_atomic.normalize": [ False ] }),
        Hyper({"descriptor.reduce_op": [ "mean" ]}),    
        Hyper({"descriptor.normalize": [ False ]}),
        Hyper({"descriptor.reduce_by_type": [ False ]}),
        Hyper({"whiten.centre": [ True ]}),         
        Hyper({"whiten.scale":  [ True ]}))         
    rr_int_hyper = GridHyper(
        Hyper({
            "whiten.centre": whiten_hyper,         
            "whiten.scale":  whiten_hyper}))         
    rr_ext_settings = GridHyper(
        Hyper({"descriptor_atomic.normalize": [ False ] }),
        Hyper({"descriptor.reduce_op": [ "sum" ]}),    
        Hyper({"descriptor.normalize": [ False ]}),
        Hyper({"descriptor.reduce_by_type": [ False ]}),
        Hyper({"whiten.centre": [ False ]}),         
        Hyper({"whiten.scale":  [ False ]}))         
    rr_ext_hyper = GridHyper(
        Hyper({
            "whiten.centre": whiten_hyper,         
            "whiten.scale":  whiten_hyper}))         
    models = []
    for hidx, updates in enumerate(krr_int_settings):
        for minimal in [True, False]:
            tag = "%s" % ("minimal" if minimal else "standard")
            model = make_gylm_krr("bmol_gylm_%s_int_krr" % tag, minimal, extensive=False)
            model.hyperUpdate(updates)
            model.hyper.add(krr_int_hyper)
            models.append(model)
    for hidx, updates in enumerate(krr_ext_settings):
        for minimal in [True, False]:
            tag = "%s" % ("minimal" if minimal else "standard")
            model = make_gylm_krr("bmol_gylm_%s_ext_krr" % tag, minimal, extensive=True)
            model.hyperUpdate(updates)
            model.hyper.add(krr_ext_hyper)
            models.append(model)
    for hidx, updates in enumerate(rr_int_settings):
        for minimal in [True, False]:
            tag = "%s" % ("minimal" if minimal else "standard")
            model = make_gylm_rr("bmol_gylm_%s_int_rr" % tag, minimal, extensive=False)
            model.hyperUpdate(updates)
            model.hyper.add(rr_int_hyper)
            models.append(model)
    for hidx, updates in enumerate(rr_ext_settings):
        for minimal in [True, False]:
            tag = "%s" % ("minimal" if minimal else "standard")
            model = make_gylm_rr("bmol_gylm_%s_ext_rr" % tag, minimal, extensive=True)
            model.hyperUpdate(updates)
            model.hyper.add(rr_ext_hyper)
            models.append(model)
    return models

def make_gylm_rr(tag, minimal, extensive):
    return Module(
        tag=tag,
        transforms=[
            ExtXyzInput(
                tag="input"),
            GylmAtomic(
                tag="descriptor_atomic",
                args={
                    "normalize": False,
                    "rcut": 3.0 if minimal else 5.0,
                    "rcut_width": 0.5,
                    "nmax": 6 if minimal else 9,
                    "lmax": 4 if minimal else 6,
                    "sigma": 0.75,
                    "part_sigma": 0.5,
                    "wconstant": False,
                    "wscale": 0.5,
                    "wcentre": 0.5,
                    "ldamp": 0.5,
                    "power": True,
                },
                inputs={
                    "configs": "input.configs"
                }),
            ReduceTypedMatrix(
                tag="descriptor",
                args = {
                    "reduce_op": "sum",
                    "normalize": False,
                    "reduce_by_type": False,
                    "types": None,
                    "epsilon": 1e-10 },
                inputs={
                    "X": "descriptor_atomic.X",
                    "T": None
                }),
            WhitenMatrix(
                tag="whiten",
                inputs={
                    "X": "descriptor.X"
                }),
            DoDivideBySize(
                tag="input_norm",
                args={
                    "config_to_size": "lambda c: len(c)",
                    "skip_if_not_force": True if extensive else False
                },
                inputs={
                    "configs": "input.configs",
                    "meta": "input.meta",
                    "y": "input.y"
                }),
            Ridge(
                tag="predictor",
                inputs={"X": "whiten.X", "y": "input_norm.y"}),
            UndoDivideBySize(
                tag="output",
                inputs={"y": "predictor.y", "sizes": "input_norm.sizes"}) 
        ],
        hyper=GridHyper(
            Hyper({ "predictor.alpha": regularization_range, })),
        broadcast={"meta": "input.meta"},
        outputs={ "y": "output.y" }
    )

def make_gylm_krr(tag, minimal, extensive):
    return Module(
        tag=tag,
        transforms=[
            ExtXyzInput(tag="input"),
            GylmAtomic(
                tag="descriptor_atomic",
                args={
                    "normalize": False,
                    "rcut": 3.0 if minimal else 5.0,
                    "rcut_width": 0.5,
                    "nmax": 6 if minimal else 9,
                    "lmax": 4 if minimal else 6,
                    "sigma": 0.75,
                    "part_sigma": 0.5,
                    "wconstant": False,
                    "wscale": 0.5,
                    "wcentre": 0.5,
                    "ldamp": 0.5,
                    "power": True,
                },
                inputs={
                    "configs": "input.configs"
                }),
            ReduceTypedMatrix(
                tag="descriptor",
                args = {
                    "reduce_op": "sum",
                    "normalize": False,
                    "reduce_by_type": False,
                    "types": None,
                    "epsilon": 1e-10 },
                inputs={
                    "X": "descriptor_atomic.X",
                    "T": None
                }),
            KernelDot(
                tag="kernel",
                inputs={"X": "descriptor.X"}),
            DoDivideBySize(
                tag="input_norm",
                args={
                    "config_to_size": "lambda c: len(c)",
                    "skip_if_not_force": True if extensive else False
                },
                inputs={
                    "configs": "input.configs",
                    "meta": "input.meta",
                    "y": "input.y"
                }),
            KernelRidge(
                tag="predictor",
                args={"alpha": None, "power": 2},
                inputs={"K": "kernel.K", "y": "input_norm.y"}),
            UndoDivideBySize(
                tag="output",
                inputs={"y": "predictor.y", "sizes": "input_norm.sizes"}) 
        ],
        hyper=GridHyper(
            Hyper({ "predictor.alpha": regularization_range, })
        ),
        broadcast={ "meta": "input.meta" },
        outputs={ "y": "output.y" }
    )

def compile_pdf():
    models = []
    for minimal in [False, True]:
        models.extend(make_pdf_rr(minimal))
        models.extend(make_pdf_krr(minimal))
    return models

def make_pdf_krr(minimal):
    return [
        Module(
            tag="bmol_pdf_soap_%s_krr" % ("minimal" if minimal else "standard"),
            transforms=[
                ExtXyzInput(tag="input"),
                SoapGylmxx(
                    tag="descriptor_atomic",
                    args={
                        "rcut": 3.0 if minimal else 5.0,
                        "nmax": 6 if minimal else 9,
                        "lmax": 4 if minimal else 6,
                        "sigma": 1.0,
                        "types": None,
                        "crossover": True,
                        "periodic": None,
                        "power": False,
                        "normalize": False 
                    },
                    inputs={
                        "configs": "input.configs"
                    }),
                GylmReduceConvolve(
                    tag="descriptor",
                    args={
                        "nmax": "@descriptor_atomic.nmax",
                        "lmax": "@descriptor_atomic.lmax",
                        "types": "@descriptor_atomic.types",
                        "normalize": True # NOTE Important
                    },
                    inputs={
                        "Q": "descriptor_atomic.X"
                    }),
                KernelDot(
                    tag="kernel",
                    inputs={"X": "descriptor.X"}),
                DoDivideBySize(
                    tag="input_norm",
                    args={
                        "config_to_size": "lambda c: len(c)",
                        "skip_if_not_force": False
                    },
                    inputs={
                        "configs": "input.configs",
                        "meta": "input.meta",
                        "y": "input.y"
                    }),
                KernelRidge(
                    tag="predictor",
                    args={"alpha": None, "power": 2},
                    inputs={"K": "kernel.K", "y": "input_norm.y"}),
                UndoDivideBySize(
                    tag="output",
                    inputs={"y": "predictor.y", "sizes": "input_norm.sizes"}) 
            ],
            hyper=GridHyper(
                Hyper({ "predictor.alpha": regularization_range, })
            ),
            broadcast={ "meta": "input.meta" },
            outputs={ "y": "output.y" }
        ),
        Module(
            tag="bmol_pdf_gylm_%s_krr" % ("minimal" if minimal else "standard"),
            transforms=[
                ExtXyzInput(tag="input"),
                GylmAtomic(
                    tag="descriptor_atomic",
                    args={
                        "rcut": 3.0 if minimal else 5.0,
                        "rcut_width": 0.5,
                        "nmax": 6 if minimal else 9,
                        "lmax": 4 if minimal else 6,
                        "sigma": 0.75,
                        "part_sigma": 0.5,
                        "wconstant": False,
                        "wscale": 0.5,
                        "wcentre": 0.5,
                        "ldamp": 0.5,
                        "power": False,
                        "normalize": False,
                    },
                    inputs={
                        "configs": "input.configs"
                    }),
                GylmReduceConvolve(
                    tag="descriptor",
                    args={
                        "nmax": "@descriptor_atomic.nmax",
                        "lmax": "@descriptor_atomic.lmax",
                        "types": "@descriptor_atomic.types",
                        "normalize": True # NOTE Important
                    },
                    inputs={
                        "Q": "descriptor_atomic.X"
                    }),
                KernelDot(
                    tag="kernel",
                    inputs={"X": "descriptor.X"}),
                DoDivideBySize(
                    tag="input_norm",
                    args={
                        "config_to_size": "lambda c: len(c)",
                        "skip_if_not_force": False
                    },
                    inputs={
                        "configs": "input.configs",
                        "meta": "input.meta",
                        "y": "input.y"
                    }),
                KernelRidge(
                    tag="predictor",
                    args={"alpha": None, "power": 2},
                    inputs={"K": "kernel.K", "y": "input_norm.y"}),
                UndoDivideBySize(
                    tag="output",
                    inputs={"y": "predictor.y", "sizes": "input_norm.sizes"}) 
            ],
            hyper=GridHyper(
                Hyper({ "predictor.alpha": regularization_range, })
            ),
            broadcast={ "meta": "input.meta" },
            outputs={ "y": "output.y" }
        )
    ]

def make_pdf_rr(minimal):
    return [
        Module(
            tag="bmol_pdf_soap_%s_rr" % ("minimal" if minimal else "standard"),
            transforms=[
                ExtXyzInput(tag="input"),
                SoapGylmxx(
                    tag="descriptor_atomic",
                    args={
                        "rcut": 3.0 if minimal else 5.0,
                        "nmax": 6 if minimal else 9,
                        "lmax": 4 if minimal else 6,
                        "sigma": 1.0,
                        "types": None,
                        "crossover": True,
                        "periodic": None,
                        "power": False,
                        "normalize": False 
                    },
                    inputs={
                        "configs": "input.configs"
                    }),
                GylmReduceConvolve(
                    tag="descriptor",
                    args={
                        "nmax": "@descriptor_atomic.nmax",
                        "lmax": "@descriptor_atomic.lmax",
                        "types": "@descriptor_atomic.types",
                        "normalize": True # NOTE Important
                    },
                    inputs={
                        "Q": "descriptor_atomic.X"
                    }),
                WhitenMatrix(
                    tag="whiten",
                    inputs={
                        "X": "descriptor.X"
                    }),
                DoDivideBySize(
                    tag="input_norm",
                    args={
                        "config_to_size": "lambda c: len(c)",
                        "skip_if_not_force": False
                    },
                    inputs={
                        "configs": "input.configs",
                        "meta": "input.meta",
                        "y": "input.y"
                    }),
                Ridge(
                    tag="predictor",
                    inputs={"X": "whiten.X", "y": "input_norm.y"}),
                UndoDivideBySize(
                    tag="output",
                    inputs={"y": "predictor.y", "sizes": "input_norm.sizes"}) 
            ],
            hyper=GridHyper(
                Hyper({ "whiten.centre": whiten_hyper,
                        "whiten.scale":  whiten_hyper}),
                Hyper({ "predictor.alpha": regularization_range, })),
            broadcast={"meta": "input.meta"},
            outputs={ "y": "output.y" }
        ),
        Module(
            tag="bmol_pdf_gylm_%s_rr" % ("minimal" if minimal else "standard"),
            transforms=[
                ExtXyzInput(tag="input"),
                GylmAtomic(
                    tag="descriptor_atomic",
                    args={
                        "rcut": 3.0 if minimal else 5.0,
                        "rcut_width": 0.5,
                        "nmax": 6 if minimal else 9,
                        "lmax": 4 if minimal else 6,
                        "sigma": 0.75,
                        "part_sigma": 0.5,
                        "wconstant": False,
                        "wscale": 0.5,
                        "wcentre": 0.5,
                        "ldamp": 0.5,
                        "power": False,
                        "normalize": False,
                    },
                    inputs={
                        "configs": "input.configs"
                    }),
                GylmReduceConvolve(
                    tag="descriptor",
                    args={
                        "nmax": "@descriptor_atomic.nmax",
                        "lmax": "@descriptor_atomic.lmax",
                        "types": "@descriptor_atomic.types",
                        "normalize": True # NOTE Important
                    },
                    inputs={
                        "Q": "descriptor_atomic.X"
                    }),
                WhitenMatrix(
                    tag="whiten",
                    inputs={
                        "X": "descriptor.X"
                    }),
                DoDivideBySize(
                    tag="input_norm",
                    args={
                        "config_to_size": "lambda c: len(c)",
                        "skip_if_not_force": False
                    },
                    inputs={
                        "configs": "input.configs",
                        "meta": "input.meta",
                        "y": "input.y"
                    }),
                Ridge(
                    tag="predictor",
                    inputs={"X": "whiten.X", "y": "input_norm.y"}),
                UndoDivideBySize(
                    tag="output",
                    inputs={"y": "predictor.y", "sizes": "input_norm.sizes"}) 
            ],
            hyper=GridHyper(
                Hyper({ "whiten.centre": whiten_hyper,
                        "whiten.scale":  whiten_hyper}),
                Hyper({ "predictor.alpha": regularization_range, })),
            broadcast={"meta": "input.meta"},
            outputs={ "y": "output.y" }
        )
    ]

def register_all():
    return {
        "bmol_physchem": compile_physchem,
        "bmol_ecfp": compile_ecfp,
        "bmol_cm": compile_cm,
        "bmol_acsf": compile_acsf,
        "bmol_mbtr": compile_mbtr,
        "bmol_soap": compile_soap,
        "bmol_gylm": compile_gylm,
        "bmol_pdf": compile_pdf,
    }

