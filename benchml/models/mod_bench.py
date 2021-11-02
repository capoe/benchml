import numpy as np

import benchml.transforms as btf
from benchml.hyper import GridHyper, Hyper
from benchml.models.common import (
    get_bench_pdf_gylm_krr_kwargs,
    get_bench_pdf_gylm_rr_kwargs,
    get_bench_pdf_soap_krr_kwargs,
    get_bench_pdf_soap_rr_kwargs,
    get_compile_gylm,
    make_soap_krr,
    make_soap_rr,
)

whiten_hyper = [False]  # NOTE: False = no whitening in ridge models
regularization_range = np.logspace(-9, +7, 17)


def compile_physchem(custom_fields=None, with_hyper=False, **kwargs):
    if custom_fields is None:
        custom_fields = []
    models = []
    for descriptor_set in ["basic", "core", "logp", "extended"]:
        models.extend(
            [
                btf.Module(
                    tag="bmol_physchem_%s_rfr" % descriptor_set,
                    transforms=[
                        btf.ExtXyzInput(tag="input"),
                        btf.Physchem2D(
                            tag="Physchem2D",
                            args={"select_predef": descriptor_set},
                            inputs={"configs": "input.configs"},
                        ),
                        btf.PhyschemUser(
                            tag="PhyschemUser",
                            args={"fields": custom_fields},
                            inputs={"configs": "input.configs"},
                        ),
                        btf.Concatenate(
                            tag="descriptor", inputs={"X": ["Physchem2D.X", "PhyschemUser.X"]}
                        ),
                        btf.DoDivideBySize(
                            tag="input_norm",
                            args={
                                "config_to_size": "lambda c: len(c)",
                                "skip_if_not_force": True,
                                "force": None,
                            },
                            inputs={
                                "configs": "input.configs",
                                "meta": "input.meta",
                                "y": "input.y",
                            },
                        ),
                        btf.RandomForestRegressor(
                            tag="predictor", inputs={"X": "descriptor.X", "y": "input_norm.y"}
                        ),
                        btf.UndoDivideBySize(
                            tag="output", inputs={"y": "predictor.y", "sizes": "input_norm.sizes"}
                        ),
                    ],
                    hyper=GridHyper(
                        Hyper({"input_norm.force": [False, True]}),
                        Hyper({"predictor.max_depth": [None]}),
                    ),
                    broadcast={"meta": "input.meta"},
                    outputs={"y": "output.y"},
                ),
                btf.Module(
                    tag="bmol_physchem_%s_rr" % descriptor_set,
                    transforms=[
                        btf.ExtXyzInput(tag="input"),
                        btf.Physchem2D(
                            tag="Physchem2D",
                            args={"select_predef": descriptor_set},
                            inputs={"configs": "input.configs"},
                        ),
                        btf.PhyschemUser(
                            tag="PhyschemUser",
                            args={"fields": custom_fields},
                            inputs={"configs": "input.configs"},
                        ),
                        btf.Concatenate(
                            tag="descriptor", inputs={"X": ["Physchem2D.X", "PhyschemUser.X"]}
                        ),
                        btf.WhitenMatrix(tag="whiten", inputs={"X": "descriptor.X"}),
                        btf.DoDivideBySize(
                            tag="input_norm",
                            args={
                                "config_to_size": "lambda c: len(c)",
                                "skip_if_not_force": True,
                                "force": None,
                            },
                            inputs={
                                "configs": "input.configs",
                                "meta": "input.meta",
                                "y": "input.y",
                            },
                        ),
                        btf.Ridge(
                            tag="predictor",
                            args={"alpha": None},
                            inputs={"X": "whiten.X", "y": "input_norm.y"},
                        ),
                        btf.UndoDivideBySize(
                            tag="output", inputs={"y": "predictor.y", "sizes": "input_norm.sizes"}
                        ),
                    ],
                    hyper=GridHyper(
                        Hyper({"input_norm.force": [False, True]}),
                        Hyper(
                            {
                                "predictor.alpha": regularization_range,
                            }
                        ),
                    ),
                    broadcast={"meta": "input.meta"},
                    outputs={"y": "output.y"},
                ),
            ]
        )
    return models


def compile_ecfp(**kwargs):
    models = []
    for radius in [2, 3]:
        models.extend(
            [
                btf.Module(
                    tag="bmol_ecfp%d_krr" % (2 * radius),
                    transforms=[
                        btf.ExtXyzInput(tag="input"),
                        btf.MorganFP(
                            tag="descriptor",
                            args={"length": 4096, "radius": radius, "normalize": True},
                            inputs={"configs": "input.configs"},
                        ),
                        btf.KernelDot(tag="kernel", inputs={"X": "descriptor.X"}),
                        btf.DoDivideBySize(
                            tag="input_norm",
                            args={
                                "config_to_size": "lambda c: len(c)",
                                "skip_if_not_force": False,
                            },
                            inputs={
                                "configs": "input.configs",
                                "meta": "input.meta",
                                "y": "input.y",
                            },
                        ),
                        btf.KernelRidge(
                            tag="predictor",
                            args={"alpha": None, "power": 2},
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
                        ),
                        Hyper({"predictor.power": [2.0]}),
                    ),
                    broadcast={"meta": "input.meta"},
                    outputs={"y": "output.y"},
                ),
                btf.Module(
                    tag="bmol_ecfp%d_rr" % (2 * radius),
                    transforms=[
                        btf.ExtXyzInput(tag="input"),
                        btf.MorganFP(
                            tag="descriptor",
                            args={"length": 2048, "radius": radius, "normalize": True},
                            inputs={"configs": "input.configs"},
                        ),
                        btf.WhitenMatrix(tag="whiten", inputs={"X": "descriptor.X"}),
                        btf.DoDivideBySize(
                            tag="input_norm",
                            args={
                                "config_to_size": "lambda c: len(c)",
                                "skip_if_not_force": False,
                            },
                            inputs={
                                "configs": "input.configs",
                                "meta": "input.meta",
                                "y": "input.y",
                            },
                        ),
                        btf.Ridge(tag="predictor", inputs={"X": "whiten.X", "y": "input_norm.y"}),
                        btf.UndoDivideBySize(
                            tag="output", inputs={"y": "predictor.y", "sizes": "input_norm.sizes"}
                        ),
                    ],
                    hyper=GridHyper(
                        Hyper(
                            {
                                "predictor.alpha": regularization_range,
                            }
                        ),
                        Hyper({"whiten.centre": [False, True], "whiten.scale": [False, True]}),
                    ),
                    outputs={"y": "output.y"},
                ),
            ]
        )
    return models


def compile_cm(*args, **kwargs):
    models = []
    for permutation in ["sorted_l2", "eigenspectrum"]:
        models.extend(
            [
                btf.Module(
                    tag="bmol_cm_%s_rr" % permutation,
                    transforms=[
                        btf.ExtXyzInput(tag="input"),
                        btf.DscribeCM(
                            tag="descriptor_atomic",
                            args={"permutation": permutation},
                            inputs={"configs": "input.configs"},
                        ),
                        btf.ReduceMatrix(
                            tag="descriptor",
                            args={"reduce": "np.mean(x, axis=0)", "norm": False, "epsilon": 1e-10},
                            inputs={"X": "descriptor_atomic.X"},
                        ),
                        btf.WhitenMatrix(tag="whiten", inputs={"X": "descriptor.X"}),
                        btf.DoDivideBySize(
                            tag="input_norm",
                            args={
                                "config_to_size": "lambda c: len(c)",
                                "skip_if_not_force": False,
                                "force": None,
                            },
                            inputs={
                                "configs": "input.configs",
                                "meta": "input.meta",
                                "y": "input.y",
                            },
                        ),
                        btf.Ridge(tag="predictor", inputs={"X": "whiten.X", "y": "input_norm.y"}),
                        btf.UndoDivideBySize(
                            tag="output", inputs={"y": "predictor.y", "sizes": "input_norm.sizes"}
                        ),
                    ],
                    hyper=GridHyper(
                        Hyper({"input_norm.force": [False, True]}),
                        Hyper(
                            {
                                "predictor.alpha": regularization_range,
                            }
                        ),
                    ),
                    broadcast={"meta": "input.meta"},
                    outputs={"y": "output.y"},
                ),
                btf.Module(
                    tag="bmol_cm_%s_krr" % permutation,
                    transforms=[
                        btf.ExtXyzInput(tag="input"),
                        btf.DscribeCM(
                            tag="descriptor_atomic",
                            args={"permutation": permutation},
                            inputs={"configs": "input.configs"},
                        ),
                        btf.ReduceMatrix(
                            tag="descriptor",
                            args={"reduce": "np.mean(x, axis=0)", "norm": False, "epsilon": 1e-10},
                            inputs={"X": "descriptor_atomic.X"},
                        ),
                        btf.KernelDot(tag="kernel", inputs={"X": "descriptor.X"}),
                        btf.DoDivideBySize(
                            tag="input_norm",
                            args={
                                "config_to_size": "lambda c: len(c)",
                                "skip_if_not_force": False,
                                "force": None,
                            },
                            inputs={
                                "configs": "input.configs",
                                "meta": "input.meta",
                                "y": "input.y",
                            },
                        ),
                        btf.KernelRidge(
                            tag="predictor",
                            args={
                                "alpha": None,
                            },
                            inputs={"K": "kernel.K", "y": "input_norm.y"},
                        ),
                        btf.UndoDivideBySize(
                            tag="output", inputs={"y": "predictor.y", "sizes": "input_norm.sizes"}
                        ),
                    ],
                    hyper=GridHyper(
                        Hyper({"input_norm.force": [False, True]}),
                        Hyper(
                            {
                                "predictor.alpha": regularization_range,
                            }
                        ),
                    ),
                    broadcast={"meta": "input.meta"},
                    outputs={"y": "output.y"},
                ),
            ]
        )
    return models


def compile_acsf(adjust_to_species=None, *args, **kwargs):
    if adjust_to_species is None:
        adjust_to_species = ["C", "N", "O"]
    models = []
    for scalerange, sharpness, scale in zip(
        [0.85, 1.2, 1.8], [1.0, 1.0, 1.2], ["minimal", "smart", "longrange"]
    ):
        for extensive in [False, True]:
            models.extend(
                [
                    btf.Module(
                        tag="bmol_acsf_%s_%s_rr" % (scale, "ext" if extensive else "int"),
                        transforms=[
                            btf.ExtXyzInput(tag="input"),
                            btf.UniversalDscribeACSF(
                                tag="descriptor_atomic",
                                args={
                                    "adjust_to_species": None,  # TODO
                                    "scalerange": scalerange,
                                    "sharpness": sharpness,
                                },
                                inputs={"configs": "input.configs"},
                            ),
                            btf.ReduceMatrix(
                                tag="descriptor",
                                args={
                                    "reduce": "np.sum(x, axis=0)"
                                    if extensive
                                    else "np.mean(x, axis=0)",
                                    "norm": False,
                                    "epsilon": 1e-10,
                                },
                                inputs={"X": "descriptor_atomic.X"},
                            ),
                            btf.WhitenMatrix(tag="whiten", inputs={"X": "descriptor.X"}),
                            btf.DoDivideBySize(
                                tag="input_norm",
                                args={
                                    "config_to_size": "lambda c: len(c)",
                                    "skip_if_not_force": True if extensive else False,
                                },
                                inputs={
                                    "configs": "input.configs",
                                    "meta": "input.meta",
                                    "y": "input.y",
                                },
                            ),
                            btf.Ridge(
                                tag="predictor", inputs={"X": "whiten.X", "y": "input_norm.y"}
                            ),
                            btf.UndoDivideBySize(
                                tag="output",
                                inputs={"y": "predictor.y", "sizes": "input_norm.sizes"},
                            ),
                        ],
                        hyper=GridHyper(
                            Hyper({"whiten.centre": whiten_hyper, "whiten.scale": whiten_hyper}),
                            Hyper(
                                {
                                    "predictor.alpha": regularization_range,
                                }
                            ),
                        ),
                        broadcast={"meta": "input.meta"},
                        outputs={"y": "output.y"},
                    ),
                    btf.Module(
                        tag="bmol_acsf_%s_%s_krr" % (scale, "ext" if extensive else "int"),
                        transforms=[
                            btf.ExtXyzInput(tag="input"),
                            btf.UniversalDscribeACSF(
                                tag="descriptor_atomic",
                                args={"adjust_to_species": None, "scalerange": scalerange},  # TODO
                                inputs={"configs": "input.configs"},
                            ),
                            btf.ReduceMatrix(
                                tag="descriptor",
                                args={
                                    "reduce": "np.sum(x, axis=0)"
                                    if extensive
                                    else "np.mean(x, axis=0)",
                                    "norm": False,
                                    "epsilon": 1e-10,
                                },
                                inputs={"X": "descriptor_atomic.X"},
                            ),
                            btf.KernelDot(tag="kernel", inputs={"X": "descriptor.X"}),
                            btf.DoDivideBySize(
                                tag="input_norm",
                                args={
                                    "config_to_size": "lambda c: len(c)",
                                    "skip_if_not_force": True if extensive else False,
                                },
                                inputs={
                                    "configs": "input.configs",
                                    "meta": "input.meta",
                                    "y": "input.y",
                                },
                            ),
                            btf.KernelRidge(
                                tag="predictor",
                                args={"alpha": None},
                                inputs={"K": "kernel.K", "y": "input_norm.y"},
                            ),
                            btf.UndoDivideBySize(
                                tag="output",
                                inputs={"y": "predictor.y", "sizes": "input_norm.sizes"},
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
                    ),
                ]
            )
    return models


def compile_mbtr(**kwargs):
    models = []
    for _ in ["default"]:
        for extensive in [False, True]:
            models.extend(
                [
                    btf.Module(
                        tag="bmol_mbtr_%s_rr" % ("ext" if extensive else "int"),
                        transforms=[
                            btf.ExtXyzInput(tag="input"),
                            btf.DscribeMBTR(
                                tag="descriptor_atomic",
                                args={},
                                inputs={"configs": "input.configs"},
                            ),
                            btf.ReduceMatrix(
                                tag="descriptor",
                                args={
                                    "reduce": "np.sum(x, axis=0)"
                                    if extensive
                                    else "np.mean(x, axis=0)",
                                    "norm": False,
                                    "epsilon": 1e-10,
                                },
                                inputs={"X": "descriptor_atomic.X"},
                            ),
                            btf.WhitenMatrix(tag="whiten", inputs={"X": "descriptor.X"}),
                            btf.DoDivideBySize(
                                tag="input_norm",
                                args={
                                    "config_to_size": "lambda c: len(c)",
                                    "skip_if_not_force": True if extensive else False,
                                },
                                inputs={
                                    "configs": "input.configs",
                                    "meta": "input.meta",
                                    "y": "input.y",
                                },
                            ),
                            btf.Ridge(
                                tag="predictor", inputs={"X": "whiten.X", "y": "input_norm.y"}
                            ),
                            btf.UndoDivideBySize(
                                tag="output",
                                inputs={"y": "predictor.y", "sizes": "input_norm.sizes"},
                            ),
                        ],
                        hyper=GridHyper(
                            Hyper({"whiten.centre": whiten_hyper, "whiten.scale": whiten_hyper}),
                            Hyper(
                                {
                                    "predictor.alpha": regularization_range,
                                }
                            ),
                        ),
                        broadcast={"meta": "input.meta"},
                        outputs={"y": "output.y"},
                    ),
                    btf.Module(
                        tag="bmol_mbtr_%s_krr" % ("ext" if extensive else "int"),
                        transforms=[
                            btf.ExtXyzInput(tag="input"),
                            btf.DscribeMBTR(
                                tag="descriptor_atomic",
                                args={},
                                inputs={"configs": "input.configs"},
                            ),
                            btf.ReduceMatrix(
                                tag="descriptor",
                                args={
                                    "reduce": "np.sum(x, axis=0)"
                                    if extensive
                                    else "np.mean(x, axis=0)",
                                    "norm": False,
                                    "epsilon": 1e-10,
                                },
                                inputs={"X": "descriptor_atomic.X"},
                            ),
                            btf.KernelDot(tag="kernel", inputs={"X": "descriptor.X"}),
                            btf.DoDivideBySize(
                                tag="input_norm",
                                args={
                                    "config_to_size": "lambda c: len(c)",
                                    "skip_if_not_force": True if extensive else False,
                                },
                                inputs={
                                    "configs": "input.configs",
                                    "meta": "input.meta",
                                    "y": "input.y",
                                },
                            ),
                            btf.KernelRidge(
                                tag="predictor",
                                args={"alpha": None},
                                inputs={"K": "kernel.K", "y": "input_norm.y"},
                            ),
                            btf.UndoDivideBySize(
                                tag="output",
                                inputs={"y": "predictor.y", "sizes": "input_norm.sizes"},
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
                    ),
                ]
            )
    return models


def compile_soap(*args, **kwargs):
    krr_int_settings = GridHyper(
        Hyper({"descriptor_atomic.normalize": [False]}),
        Hyper({"descriptor_atomic.mode": ["minimal", "smart", "longrange"]}),
        Hyper({"descriptor_atomic.crossover": [False, True]}),
        Hyper({"descriptor.reduce_op": ["mean"]}),
        Hyper({"descriptor.normalize": [False]}),
        Hyper({"descriptor.reduce_by_type": [False]}),
        Hyper({"whiten.centre": [False]}),
        Hyper({"whiten.scale": [False]}),
        Hyper({"predictor.power": [2]}),
    )
    krr_int_hyper = GridHyper(Hyper({"predictor.power": [1, 2, 3]}))
    krr_ext_settings = GridHyper(
        Hyper({"descriptor_atomic.normalize": [False]}),
        Hyper({"descriptor_atomic.mode": ["minimal", "smart", "longrange"]}),
        Hyper({"descriptor_atomic.crossover": [False, True]}),
        Hyper({"descriptor.reduce_op": ["sum"]}),
        Hyper({"descriptor.normalize": [False]}),
        Hyper({"descriptor.reduce_by_type": [False]}),
        Hyper({"whiten.centre": [False]}),
        Hyper({"whiten.scale": [False]}),
        Hyper({"predictor.power": [1]}),
    )
    krr_ext_hyper = GridHyper(Hyper({"predictor.power": [1, 2, 3]}))
    rr_int_settings = GridHyper(
        Hyper({"descriptor_atomic.normalize": [False]}),
        Hyper({"descriptor_atomic.mode": ["minimal", "smart", "longrange"]}),
        Hyper({"descriptor_atomic.crossover": [False, True]}),
        Hyper({"descriptor.reduce_op": ["mean"]}),
        Hyper({"descriptor.normalize": [False]}),
        Hyper({"descriptor.reduce_by_type": [False]}),
        Hyper({"whiten.centre": [True]}),
        Hyper({"whiten.scale": [True]}),
    )
    rr_int_hyper = GridHyper(Hyper({"whiten.centre": whiten_hyper, "whiten.scale": whiten_hyper}))
    rr_ext_settings = GridHyper(
        Hyper({"descriptor_atomic.normalize": [False]}),
        Hyper({"descriptor_atomic.mode": ["minimal", "smart", "longrange"]}),
        Hyper({"descriptor_atomic.crossover": [False, True]}),
        Hyper({"descriptor.reduce_op": ["sum"]}),
        Hyper({"descriptor.normalize": [False]}),
        Hyper({"descriptor.reduce_by_type": [False]}),
        Hyper({"whiten.centre": [False]}),
        Hyper({"whiten.scale": [False]}),
    )
    rr_ext_hyper = GridHyper(Hyper({"whiten.centre": whiten_hyper, "whiten.scale": whiten_hyper}))
    models = []
    for hidx, updates in enumerate(krr_int_settings):
        tag = "%s_%s" % (
            updates["descriptor_atomic.mode"],
            "cross" if updates["descriptor_atomic.crossover"] else "nocross",
        )
        model = make_soap_krr(tag="bmol_soap_%s_int_krr" % tag, extensive=False)
        model.hyperUpdate(updates)
        model.hyper.add(krr_int_hyper)
        models.append(model)
    for hidx, updates in enumerate(krr_ext_settings):
        tag = "%s_%s" % (
            updates["descriptor_atomic.mode"],
            "cross" if updates["descriptor_atomic.crossover"] else "nocross",
        )
        model = make_soap_krr(tag="bmol_soap_%s_ext_krr" % tag, extensive=True)
        model.hyperUpdate(updates)
        model.hyper.add(krr_ext_hyper)
        models.append(model)
    for hidx, updates in enumerate(rr_int_settings):
        tag = "%s_%s" % (
            updates["descriptor_atomic.mode"],
            "cross" if updates["descriptor_atomic.crossover"] else "nocross",
        )
        model = make_soap_rr(tag="bmol_soap_%s_int_rr" % tag, extensive=False)
        model.hyperUpdate(updates)
        model.hyper.add(rr_int_hyper)
        models.append(model)
    for hidx, updates in enumerate(rr_ext_settings):
        tag = "%s_%s" % (
            updates["descriptor_atomic.mode"],
            "cross" if updates["descriptor_atomic.crossover"] else "nocross",
        )
        model = make_soap_rr(tag="bmol_soap_%s_ext_rr" % tag, extensive=True)
        model.hyperUpdate(updates)
        model.hyper.add(rr_ext_hyper)
        models.append(model)
    return models


compile_gylm = get_compile_gylm("bmol", whiten_hyper)


def compile_pdf():
    models = []
    for minimal in [False, True]:
        models.extend(make_pdf_rr(minimal))
        models.extend(make_pdf_krr(minimal))
    return models


def make_pdf_krr(minimal):
    return [
        btf.Module(
            tag="bmol_pdf_soap_%s_krr" % ("minimal" if minimal else "standard"),
            **get_bench_pdf_soap_krr_kwargs(minimal, regularization_range),
        ),
        btf.Module(
            tag="bmol_pdf_gylm_%s_krr" % ("minimal" if minimal else "standard"),
            **get_bench_pdf_gylm_krr_kwargs(minimal, regularization_range),
        ),
    ]


def make_pdf_rr(minimal):
    return [
        btf.Module(
            tag="bmol_pdf_soap_%s_rr" % ("minimal" if minimal else "standard"),
            **get_bench_pdf_soap_rr_kwargs(minimal, whiten_hyper, regularization_range),
        ),
        btf.Module(
            tag="bmol_pdf_gylm_%s_rr" % ("minimal" if minimal else "standard"),
            **get_bench_pdf_gylm_rr_kwargs(minimal, whiten_hyper, regularization_range),
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
        "bmol_gylm": compile_gylm,
        "bmol_pdf": compile_pdf,
    }
