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
                    Concatenate(tag="descriptor",
                        inputs={"X": [ "Physchem2D.X", "PhyschemUser.X" ]}),
                    RandomForestRegressor(tag="pred",
                        inputs={"X": "descriptor.X", "y": "input.y"})
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
                    Concatenate(tag="descriptor",
                        inputs={"X": [ "Physchem2D.X", "PhyschemUser.X" ]}),
                    WhitenMatrix(tag="whiten",
                        inputs={"X": "descriptor.X"}),
                    Ridge(tag="predictor",
                        args={"alpha": None},
                        inputs={"X": "whiten.X", "y": "input.y"})
                ],
                hyper=GridHyper(
                    Hyper({ "predictor.alpha": np.logspace(-5,+5, 7), })),
                broadcast={"meta": "input.meta"},
                outputs={"y": "predictor.y"}),
        ])
    return models

def compile_ecfp(**kwargs):
    return [
        Module(
            tag="bmol_ecfp_krr",
            transforms=[
                ExtXyzInput(tag="input"),
                MorganFP(
                    tag="descriptor",
                    args={"length": 4096, "radius": 2, "normalize": True},
                    inputs={"configs": "input.configs"}),
                KernelDot(
                    tag="kernel",
                    inputs={"X": "descriptor.X"}),
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
                    tag="descriptor",
                    args={"length": 2048, "radius": 2, "normalize": True},
                    inputs={"configs": "input.configs"}),
                Ridge(inputs={"X": "descriptor.X", "y": "input.y"})
            ],
            hyper=BayesianHyper(
                Hyper({"Ridge.alpha": np.linspace(-5,5,7)}),
                convert={
                    "Ridge.alpha": "lambda p: 10**p"}),
            outputs={"y": "Ridge.y"}
        )
    ]

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
                            "reduce": "np.sum(x, axis=0)",
                            "norm": False,
                            "epsilon": 1e-10 },
                        inputs={"X": "descriptor_atomic.X"}),
                    WhitenMatrix(
                        tag="whiten",
                        inputs={
                            "X": "descriptor.X"
                        }),
                    Ridge(
                        tag="predictor",
                        inputs={"X": "whiten.X", "y": "input.y"}) ],
                hyper=GridHyper(
                    Hyper({ "predictor.alpha": np.logspace(-9,+7, 17), })),
                broadcast={"meta": "input.meta"},
                outputs={ "y": "predictor.y" }),
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
                            "reduce": "np.sum(x, axis=0)",
                            "norm": True,
                            "epsilon": 1e-10 },
                        inputs={"X": "descriptor_atomic.X"}),
                    KernelDot(
                        tag="kernel",
                        inputs={
                            "X": "descriptor.X"
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
                    Hyper({ "predictor.alpha": np.logspace(-9, +7, 17), })),
                broadcast={"meta": "input.meta"},
                outputs={ "y": "predictor.y" })
        ])
    return models

def compile_acsf(adjust_to_species=["C", "N", "O"], *args, **kwargs):
    models = []
    for scalerange, scale in zip(
        [ 1.0, 1.2, 1.8 ],
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
                                "adjust_to_species": adjust_to_species,
                                "scalerange": scalerange},
                            inputs={"configs": "input.configs"}),
                        ReduceMatrix(
                            tag="descriptor",
                            args = {
                                "reduce": "np.sum(x, axis=0)",
                                "norm": False if extensive else True,
                                "epsilon": 1e-10 },
                            inputs={"X": "descriptor_atomic.X"}),
                        WhitenMatrix(
                            tag="whiten",
                            inputs={
                                "X": "descriptor.X"
                            }),
                        Ridge(
                            tag="predictor",
                            inputs={"X": "whiten.X", "y": "input.y"}) ],
                    hyper=GridHyper(
                        Hyper({ "whiten.centre": [False, True],
                                "whiten.scale":  [False, True]}),
                        Hyper({ "predictor.alpha": np.logspace(-9,+7, 17), })),
                    broadcast={"meta": "input.meta"},
                    outputs={ "y": "predictor.y" }),
                Module(
                    tag="bmol_acsf_%s_%s_krr" % (scale, "ext" if extensive else "int"),
                    transforms=[
                        ExtXyzInput(tag="input"),
                        UniversalDscribeACSF(
                            tag="descriptor_atomic",
                            args={
                                "adjust_to_species": adjust_to_species,
                                "scalerange": scalerange},
                            inputs={"configs": "input.configs"}),
                        ReduceMatrix(
                            tag="descriptor",
                            args = {
                                "reduce": "np.sum(x, axis=0)",
                                "norm": False if extensive else True,
                                "epsilon": 1e-10 },
                            inputs={"X": "descriptor_atomic.X"}),
                        KernelDot(
                            tag="kernel",
                            inputs={
                                "X": "descriptor.X"
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
                        Hyper({ "predictor.alpha": np.logspace(-9, +7, 17), })),
                    broadcast={"meta": "input.meta"},
                    outputs={ "y": "predictor.y" })
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
                                "reduce": "np.sum(x, axis=0)",
                                "norm": False if extensive else True,
                                "epsilon": 1e-10 },
                            inputs={"X": "descriptor_atomic.X"}),
                        WhitenMatrix(
                            tag="whiten",
                            inputs={
                                "X": "descriptor.X"
                            }),
                        Ridge(
                            tag="predictor",
                            inputs={"X": "whiten.X", "y": "input.y"}) ],
                    hyper=GridHyper(
                        Hyper({ "whiten.centre": [False, True],
                                "whiten.scale":  [False, True]}),
                        Hyper({ "predictor.alpha": np.logspace(-9,+7, 17), })),
                    broadcast={"meta": "input.meta"},
                    outputs={ "y": "predictor.y" }),
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
                                "reduce": "np.sum(x, axis=0)",
                                "norm": False if extensive else True,
                                "epsilon": 1e-10 },
                            inputs={"X": "descriptor_atomic.X"}),
                        KernelDot(
                            tag="kernel",
                            inputs={
                                "X": "descriptor.X"
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
                        Hyper({ "predictor.alpha": np.logspace(-9, +7, 17), })),
                    broadcast={"meta": "input.meta"},
                    outputs={ "y": "predictor.y" })
            ])
    return models

def compile_soap(*args, **kwargs):
    krr_int_settings = GridHyper(
        Hyper({"descriptor_atomic.normalize": [ True ] }),
        Hyper({"descriptor_atomic.mode": [ "minimal", "smart", "longrange" ] }),
        Hyper({"descriptor_atomic.crossover": [ False, True ] }),
        Hyper({"descriptor.reduce_op": [ "sum" ]}),
        Hyper({"descriptor.normalize": [ True ]}),
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
            "whiten.centre": [ False, True ],         
            "whiten.scale":  [ False, True ]}))         
    rr_int_settings = GridHyper(
        Hyper({"descriptor_atomic.normalize": [ False ] }),
        Hyper({"descriptor_atomic.mode": [ "minimal", "smart", "longrange" ] }),
        Hyper({"descriptor_atomic.crossover": [ False, True ] }),
        Hyper({"descriptor.reduce_op": [ "sum" ]}),    
        Hyper({"descriptor.normalize": [ True ]}),
        Hyper({"descriptor.reduce_by_type": [ False ]}),
        Hyper({"whiten.centre": [ True ]}),         
        Hyper({"whiten.scale":  [ True ]}))         
    rr_int_hyper = GridHyper(
        Hyper({
            "whiten.centre": [ False, True ],         
            "whiten.scale":  [ False, True ]}))         
    models = []
    for hidx, updates in enumerate(krr_int_settings):
        tag = "%s_%s" % (
            updates["descriptor_atomic.mode"], 
            "cross" if updates["descriptor_atomic.crossover"] else "nocross")
        model = make_soap_krr(tag="bmol_soap_%s_int_krr" % tag)
        model.hyperUpdate(updates)
        model.hyper.add(krr_int_hyper)
        models.append(model)
    for hidx, updates in enumerate(krr_ext_settings):
        tag = "%s_%s" % (
            updates["descriptor_atomic.mode"], 
            "cross" if updates["descriptor_atomic.crossover"] else "nocross")
        model = make_soap_krr(tag="bmol_soap_%s_ext_krr" % tag)
        model.hyperUpdate(updates)
        model.hyper.add(krr_ext_hyper)
        models.append(model)
    for hidx, updates in enumerate(rr_int_settings):
        tag = "%s_%s" % (
            updates["descriptor_atomic.mode"], 
            "cross" if updates["descriptor_atomic.crossover"] else "nocross")
        model = make_soap_rr(tag="bmol_soap_%s_int_rr" % tag)
        model.hyperUpdate(updates)
        model.hyper.add(rr_int_hyper)
        models.append(model)
    for hidx, updates in enumerate(rr_ext_settings):
        tag = "%s_%s" % (
            updates["descriptor_atomic.mode"], 
            "cross" if updates["descriptor_atomic.crossover"] else "nocross")
        model = make_soap_rr(tag="bmol_soap_%s_ext_rr" % tag)
        model.hyperUpdate(updates)
        model.hyper.add(rr_ext_hyper)
        models.append(model)
    return models

def make_soap_krr(tag):
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

def make_soap_rr(tag):
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
            Ridge(
                tag="predictor",
                inputs={"X": "whiten.X", "y": "input.y"}) ],
        hyper=GridHyper(
            Hyper({ "predictor.alpha": np.logspace(-9, +7, 17), })),
        broadcast={"meta": "input.meta"},
        outputs={ "y": "predictor.y" })

# def compile_dscribe(**kwargs):
#     return [
#         Module(
#             tag="bmol_"+DescriptorClass.__name__.lower()+"_ridge",
#             transforms=[
#                 ExtXyzInput(tag="input"),
#                 DescriptorClass(
#                     tag="descriptor_atomic",
#                     inputs={"configs": "input.configs"}),
#                 ReduceMatrix(
#                     tag="descriptor",
#                     args = {
#                         "reduce": "np.sum(x, axis=0)",
#                         "norm": False,
#                         "epsilon": 1e-10 },
#                     inputs={"X": "descriptor_atomic.X"}),
#                 WhitenMatrix(
#                     tag="whiten",
#                     inputs={
#                         "X": "descriptor.X"
#                     }),
#                 Ridge(
#                     tag="predictor",
#                     inputs={"X": "whiten.X", "y": "input.y"}) ],
#             hyper=GridHyper(
#                 Hyper({ "predictor.alpha": np.logspace(-5,+5, 7), })),
#             broadcast={"meta": "input.meta"},
#             outputs={ "y": "predictor.y" }) \
#         for DescriptorClass in [ DscribeMBTR, DscribeLMBTR ]
#     ]

def compile_gylm(*args, **kwargs):
    krr_int_settings = GridHyper(
        Hyper({"descriptor_atomic.normalize": [ True ] }),
        Hyper({"descriptor.reduce_op": [ "sum" ]}),
        Hyper({"descriptor.normalize": [ True ]}),
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
    rr_ext_settings = GridHyper(
        Hyper({"descriptor_atomic.normalize": [ False ] }),
        Hyper({"descriptor.reduce_op": [ "sum" ]}),    
        Hyper({"descriptor.normalize": [ False ]}),
        Hyper({"descriptor.reduce_by_type": [ False ]}),
        Hyper({"whiten.centre": [ False ]}),         
        Hyper({"whiten.scale":  [ False ]}))         
    rr_ext_hyper = GridHyper(
        Hyper({
            "whiten.centre": [ False ],         
            "whiten.scale":  [ False ]}))         
    rr_int_settings = GridHyper(
        Hyper({"descriptor_atomic.normalize": [ False ] }),
        Hyper({"descriptor.reduce_op": [ "sum" ]}),    
        Hyper({"descriptor.normalize": [ True ]}),
        Hyper({"descriptor.reduce_by_type": [ False ]}),
        Hyper({"whiten.centre": [ True ]}),         
        Hyper({"whiten.scale":  [ True ]}))         
    rr_int_hyper = GridHyper(
        Hyper({
            "whiten.centre": [ False ],         
            "whiten.scale":  [ False ]}))         
    models = []
    for hidx, updates in enumerate(krr_int_settings):
        for minimal in [True, False]:
            tag = "%s" % ("minimal" if minimal else "standard")
            model = make_gylm_krr("bmol_gylm_%s_int_krr" % tag, minimal)
            model.hyperUpdate(updates)
            model.hyper.add(krr_int_hyper)
            models.append(model)
    for hidx, updates in enumerate(krr_ext_settings):
        for minimal in [True, False]:
            tag = "%s" % ("minimal" if minimal else "standard")
            model = make_gylm_krr("bmol_gylm_%s_ext_krr" % tag, minimal)
            model.hyperUpdate(updates)
            model.hyper.add(krr_ext_hyper)
            models.append(model)
    for hidx, updates in enumerate(rr_int_settings):
        for minimal in [True, False]:
            tag = "%s" % ("minimal" if minimal else "standard")
            model = make_gylm_rr("bmol_gylm_%s_int_rr" % tag, minimal)
            model.hyperUpdate(updates)
            model.hyper.add(rr_int_hyper)
            models.append(model)
    for hidx, updates in enumerate(rr_ext_settings):
        for minimal in [True, False]:
            tag = "%s" % ("minimal" if minimal else "standard")
            model = make_gylm_rr("bmol_gylm_%s_ext_rr" % tag, minimal)
            model.hyperUpdate(updates)
            model.hyper.add(rr_ext_hyper)
            models.append(model)
    return models

def make_gylm_rr(tag, minimal):
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
            Ridge(
                tag="predictor",
                inputs={"X": "whiten.X", "y": "input.y"}) ],
        hyper=GridHyper(
            Hyper({ "predictor.alpha": np.logspace(-9, +7, 17), })),
        broadcast={"meta": "input.meta"},
        outputs={ "y": "predictor.y" }
    )

def make_gylm_krr(tag, minimal):
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
            KernelRidge(
                tag="predictor",
                args={"alpha": 1e-5, "power": 2},
                inputs={"K": "kernel.K", "y": "input.y"})
        ],
        hyper=GridHyper(
            Hyper({ "predictor.alpha": np.logspace(-9,+7, 17), })
        ),
        broadcast={ "meta": "input.meta" },
        outputs={ "y": "predictor.y" }
    )

# def compile_gylm(**kwargs):
#     return [
#         Module(
#             tag="bmol_gylm_minimal_rr",
#             transforms=[
#                 ExtXyzInput(
#                     tag="input"),
#                 GylmAtomic(
#                     tag="descriptor_atomic",
#                     args={
#                         "normalize": False,
#                         "rcut": 3.0,
#                         "rcut_width": 0.5,
#                         "nmax": 6,
#                         "lmax": 4,
#                         "sigma": 0.75,
#                         "part_sigma": 0.5,
#                         "wconstant": False,
#                         "wscale": 1.0,
#                         "wcentre": 1.0,
#                         "ldamp": 4,
#                         "power": True,
#                     },
#                     inputs={
#                         "configs": "input.configs"
#                     }),
#                 ReduceTypedMatrix(
#                     tag="descriptor",
#                     args = {
#                         "reduce_op": "sum",
#                         "normalize": False,
#                         "reduce_by_type": False,
#                         "types": None,
#                         "epsilon": 1e-10 },
#                     inputs={
#                         "X": "descriptor_atomic.X",
#                         "T": None
#                     }),
#                 WhitenMatrix(
#                     tag="whiten",
#                     inputs={
#                         "X": "descriptor.X"
#                     }),
#                 Ridge(
#                     tag="predictor",
#                     inputs={"X": "whiten.X", "y": "input.y"}) ],
#             hyper=GridHyper(
#                 Hyper({ "predictor.alpha": np.logspace(-7, +7, 15), })),
#             broadcast={"meta": "input.meta"},
#             outputs={ "y": "predictor.y" }
#         ),
#         Module(
#             tag="bmol_gylm_grid_krr",
#             transforms=[
#                 ExtXyzInput(tag="input"),
#                 GylmAverage(
#                     tag="descriptor",
#                     inputs={"configs": "input.configs"}),
#                 KernelDot(
#                     tag="kernel",
#                     inputs={"X": "descriptor.X"}),
#                 KernelRidge(
#                     args={"alpha": 1e-5, "power": 2},
#                     inputs={"K": "kernel.K", "y": "input.y"})
#             ],
#             hyper=GridHyper(
#                 Hyper({ "KernelRidge.alpha": np.logspace(-5,+1, 7), }),
#                 Hyper({ "KernelRidge.power": [ 2. ] })),
#             broadcast={ "meta": "input.meta" },
#             outputs={ "y": "KernelRidge.y" }
#         ),
#         Module(
#             tag="bmol_gylm_bayes_krr",
#             transforms=[
#                 ExtXyzInput(tag="input"),
#                 GylmAverage(
#                     tag="descriptor",
#                     inputs={"configs": "input.configs"}),
#                 KernelDot(
#                     tag="kernel",
#                     inputs={"X": "descriptor.X"}),
#                 KernelRidge(
#                     args={"alpha": 1e-5, "power": 2},
#                     inputs={"K": "kernel.K", "y": "input.y"})
#             ],
#             hyper=BayesianHyper(
#                 Hyper({ "KernelRidge.alpha": np.linspace(-5,+1, 7), }),
#                 Hyper({ "KernelRidge.power": [ 1., 4. ] }),
#                 init_points=10,
#                 n_iter=20,
#                 convert={
#                     "KernelRidge.alpha": "lambda p: 10**p"}),
#             broadcast={ "meta": "input.meta" },
#             outputs={ "y": "KernelRidge.y" }
#         ),
#     ]

def register_all():
    return {
        "bmol_physchem": compile_physchem,
        "bmol_ecfp": compile_ecfp,
        "bmol_cm": compile_cm,
        "bmol_acsf": compile_acsf,
        "bmol_mbtr": compile_mbtr,
        "bmol_soap": compile_soap,
        "bmol_gylm": compile_gylm,
    }

