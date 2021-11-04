import numpy as np

import benchml.transforms as btf
from benchml.hyper import GridHyper, Hyper

logd_hybrid_topo_gp_kwargs = dict(
    transforms=[
        btf.ExtXyzInput(tag="input"),
        btf.CxCalcTransform(
            tag="cx", args={"reshape_as_matrix": True}, inputs={"configs": "input.configs"}
        ),
        btf.KernelGaussian(tag="kern_gaussian", args={"self_kernel": True}, inputs={"X": "cx.X"}),
        btf.MorganFP(
            tag="desc",
            args={"length": 4096, "radius": 2, "normalize": True},
            inputs={"configs": "input.configs"},
        ),
        btf.KernelDot(tag="kern", args={"self_kernel": True}, inputs={"X": "desc.X"}),
        btf.Add(
            tag="add_k",
            args={"coeffs": [0.5, 0.5]},
            inputs={"X": ["kern_gaussian.K", "kern.K"]},
        ),
        btf.Add(
            tag="add_k_self",
            args={"coeffs": [0.5, 0.5]},
            inputs={"X": ["kern_gaussian.K_self", "kern.K_self"]},
        ),
        btf.GaussianProcess(
            args={"alpha": 1e-5, "power": 2},
            inputs={"K": "add_k.y", "K_self": "add_k_self.y", "y": "input.y"},
        ),
    ],
    hyper=(
        GridHyper(
            Hyper({"desc.radius": [2]}),
            Hyper(
                {
                    "GaussianProcess.alpha": np.logspace(-5, +1, 7),
                }
            ),
            Hyper({"kern_gaussian.scale": [1.0, 2.0]}),
            Hyper({"add_k.coeffs": [[0.25, 0.75]], "add_k_self.coeffs": [[0.25, 0.75]]}),
            Hyper({"GaussianProcess.power": [2.0]}),
        ),
    ),
    broadcast={"meta": "input.meta"},
    outputs={
        "y": "GaussianProcess.y",
        "dy": "GaussianProcess.dy",
        "dy_rank": "GaussianProcess.dy_rank",
        "dy_zscore": "GaussianProcess.dy_zscore",
    },
)

gylm_regularization_range = np.logspace(-9, +7, 17)


def make_gylm_rr(tag, minimal, extensive):
    return btf.Module(
        tag=tag,
        transforms=[
            btf.ExtXyzInput(tag="input"),
            btf.GylmAtomic(
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
                inputs={"configs": "input.configs"},
            ),
            btf.ReduceTypedMatrix(
                tag="descriptor",
                args={
                    "reduce_op": "sum",
                    "normalize": False,
                    "reduce_by_type": False,
                    "types": None,
                    "epsilon": 1e-10,
                },
                inputs={"X": "descriptor_atomic.X", "T": None},
            ),
            btf.WhitenMatrix(tag="whiten", inputs={"X": "descriptor.X"}),
            btf.DoDivideBySize(
                tag="input_norm",
                args={
                    "config_to_size": "lambda c: len(c)",
                    "skip_if_not_force": True if extensive else False,
                },
                inputs={"configs": "input.configs", "meta": "input.meta", "y": "input.y"},
            ),
            btf.Ridge(tag="predictor", inputs={"X": "whiten.X", "y": "input_norm.y"}),
            btf.UndoDivideBySize(
                tag="output", inputs={"y": "predictor.y", "sizes": "input_norm.sizes"}
            ),
        ],
        hyper=GridHyper(
            Hyper(
                {
                    "predictor.alpha": gylm_regularization_range,
                }
            )
        ),
        broadcast={"meta": "input.meta"},
        outputs={"y": "output.y"},
    )


def make_gylm_krr(tag, minimal, extensive):
    return btf.Module(
        tag=tag,
        transforms=[
            btf.ExtXyzInput(tag="input"),
            btf.GylmAtomic(
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
                inputs={"configs": "input.configs"},
            ),
            btf.ReduceTypedMatrix(
                tag="descriptor",
                args={
                    "reduce_op": "sum",
                    "normalize": False,
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
                    "config_to_size": "lambda c: len(c)",
                    "skip_if_not_force": True if extensive else False,
                },
                inputs={"configs": "input.configs", "meta": "input.meta", "y": "input.y"},
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
                    "predictor.alpha": gylm_regularization_range,
                }
            )
        ),
        broadcast={"meta": "input.meta"},
        outputs={"y": "output.y"},
    )


def get_bench_pdf_gylm_rr_kwargs(minimal, whiten_hyper, regularization_range):
    return dict(
        transforms=[
            btf.ExtXyzInput(tag="input"),
            btf.GylmAtomic(
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
                inputs={"configs": "input.configs"},
            ),
            btf.GylmReduceConvolve(
                tag="descriptor",
                args={
                    "nmax": "@descriptor_atomic.nmax",
                    "lmax": "@descriptor_atomic.lmax",
                    "types": "@descriptor_atomic.types",
                    "normalize": True,  # NOTE Important
                },
                inputs={"Q": "descriptor_atomic.X"},
            ),
            btf.WhitenMatrix(tag="whiten", inputs={"X": "descriptor.X"}),
            btf.DoDivideBySize(
                tag="input_norm",
                args={"config_to_size": "lambda c: len(c)", "skip_if_not_force": False},
                inputs={"configs": "input.configs", "meta": "input.meta", "y": "input.y"},
            ),
            btf.Ridge(tag="predictor", inputs={"X": "whiten.X", "y": "input_norm.y"}),
            btf.UndoDivideBySize(
                tag="output", inputs={"y": "predictor.y", "sizes": "input_norm.sizes"}
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
    )


def get_bench_pdf_gylm_krr_kwargs(minimal, regularization_range):
    return dict(
        transforms=[
            btf.ExtXyzInput(tag="input"),
            btf.GylmAtomic(
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
                inputs={"configs": "input.configs"},
            ),
            btf.GylmReduceConvolve(
                tag="descriptor",
                args={
                    "nmax": "@descriptor_atomic.nmax",
                    "lmax": "@descriptor_atomic.lmax",
                    "types": "@descriptor_atomic.types",
                    "normalize": True,  # NOTE Important
                },
                inputs={"Q": "descriptor_atomic.X"},
            ),
            btf.KernelDot(tag="kernel", inputs={"X": "descriptor.X"}),
            btf.DoDivideBySize(
                tag="input_norm",
                args={"config_to_size": "lambda c: len(c)", "skip_if_not_force": False},
                inputs={"configs": "input.configs", "meta": "input.meta", "y": "input.y"},
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
            )
        ),
        broadcast={"meta": "input.meta"},
        outputs={"y": "output.y"},
    )


def get_bench_pdf_soap_rr_kwargs(minimal, whiten_hyper, regularization_range):
    return dict(
        transforms=[
            btf.ExtXyzInput(tag="input"),
            btf.SoapGylmxx(
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
                    "normalize": False,
                },
                inputs={"configs": "input.configs"},
            ),
            btf.GylmReduceConvolve(
                tag="descriptor",
                args={
                    "nmax": "@descriptor_atomic.nmax",
                    "lmax": "@descriptor_atomic.lmax",
                    "types": "@descriptor_atomic.types",
                    "normalize": True,  # NOTE Important
                },
                inputs={"Q": "descriptor_atomic.X"},
            ),
            btf.WhitenMatrix(tag="whiten", inputs={"X": "descriptor.X"}),
            btf.DoDivideBySize(
                tag="input_norm",
                args={"config_to_size": "lambda c: len(c)", "skip_if_not_force": False},
                inputs={"configs": "input.configs", "meta": "input.meta", "y": "input.y"},
            ),
            btf.Ridge(tag="predictor", inputs={"X": "whiten.X", "y": "input_norm.y"}),
            btf.UndoDivideBySize(
                tag="output", inputs={"y": "predictor.y", "sizes": "input_norm.sizes"}
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
    )


def get_bench_pdf_soap_krr_kwargs(minimal, regularization_range):
    return dict(
        transforms=[
            btf.ExtXyzInput(tag="input"),
            btf.SoapGylmxx(
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
                    "normalize": False,
                },
                inputs={"configs": "input.configs"},
            ),
            btf.GylmReduceConvolve(
                tag="descriptor",
                args={
                    "nmax": "@descriptor_atomic.nmax",
                    "lmax": "@descriptor_atomic.lmax",
                    "types": "@descriptor_atomic.types",
                    "normalize": True,  # NOTE Important
                },
                inputs={"Q": "descriptor_atomic.X"},
            ),
            btf.KernelDot(tag="kernel", inputs={"X": "descriptor.X"}),
            btf.DoDivideBySize(
                tag="input_norm",
                args={"config_to_size": "lambda c: len(c)", "skip_if_not_force": False},
                inputs={"configs": "input.configs", "meta": "input.meta", "y": "input.y"},
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
            )
        ),
        broadcast={"meta": "input.meta"},
        outputs={"y": "output.y"},
    )


def make_soap_krr(tag, extensive):
    return btf.Module(
        tag=tag,
        transforms=[
            btf.ExtXyzInput(tag="input"),
            btf.UniversalSoapGylmxx(tag="descriptor_atomic", inputs={"configs": "input.configs"}),
            btf.ReduceTypedMatrix(
                tag="descriptor",
                args={
                    "reduce_op": "np.sum(x, axis=0)",
                    "normalize": False,
                    "reduce_by_type": False,
                    "types": None,
                    "epsilon": 1e-10,
                },
                inputs={"X": "descriptor_atomic.X", "T": "descriptor_atomic.T"},
            ),
            btf.WhitenMatrix(tag="whiten", inputs={"X": "descriptor.X"}),
            btf.KernelDot(tag="kernel", inputs={"X": "whiten.X"}),
            btf.DoDivideBySize(
                tag="input_norm",
                args={
                    "config_to_size": "lambda c: len(c)",
                    "skip_if_not_force": True if extensive else False,
                },
                inputs={"configs": "input.configs", "meta": "input.meta", "y": "input.y"},
            ),
            btf.KernelRidge(
                tag="predictor", args={"alpha": None}, inputs={"K": "kernel.K", "y": "input_norm.y"}
            ),
            btf.UndoDivideBySize(
                tag="output", inputs={"y": "predictor.y", "sizes": "input_norm.sizes"}
            ),
        ],
        hyper=GridHyper(
            Hyper(
                {
                    "predictor.alpha": gylm_regularization_range,
                }
            )
        ),
        broadcast={"meta": "input.meta"},
        outputs={"y": "output.y"},
    )


def make_soap_rr(tag, extensive):
    return btf.Module(
        tag=tag,
        transforms=[
            btf.ExtXyzInput(tag="input"),
            btf.UniversalSoapGylmxx(tag="descriptor_atomic", inputs={"configs": "input.configs"}),
            btf.ReduceTypedMatrix(
                tag="descriptor",
                args={
                    "reduce_op": "np.sum(x, axis=0)",
                    "normalize": False,
                    "reduce_by_type": False,
                    "types": None,
                    "epsilon": 1e-10,
                },
                inputs={"X": "descriptor_atomic.X", "T": "descriptor_atomic.T"},
            ),
            btf.WhitenMatrix(tag="whiten", inputs={"X": "descriptor.X"}),
            btf.DoDivideBySize(
                tag="input_norm",
                args={
                    "config_to_size": "lambda c: len(c)",
                    "skip_if_not_force": True if extensive else False,
                },
                inputs={"configs": "input.configs", "meta": "input.meta", "y": "input.y"},
            ),
            btf.Ridge(tag="predictor", inputs={"X": "whiten.X", "y": "input_norm.y"}),
            btf.UndoDivideBySize(
                tag="output", inputs={"y": "predictor.y", "sizes": "input_norm.sizes"}
            ),
        ],
        hyper=GridHyper(
            Hyper(
                {
                    "predictor.alpha": gylm_regularization_range,
                }
            )
        ),
        broadcast={"meta": "input.meta"},
        outputs={"y": "output.y"},
    )


def get_compile_gylm(mod_name, whiten_hyper):
    def customisable_compile_gylm(*args, **kwargs):
        krr_int_settings = GridHyper(
            Hyper({"descriptor_atomic.normalize": [False]}),
            Hyper({"descriptor.reduce_op": ["mean"]}),
            Hyper({"descriptor.normalize": [False]}),
            Hyper({"descriptor.reduce_by_type": [False]}),
            Hyper({"predictor.power": [2]}),
        )
        krr_int_hyper = GridHyper(Hyper({"predictor.power": [1, 2, 3]}))
        krr_ext_settings = GridHyper(
            Hyper({"descriptor_atomic.normalize": [False]}),
            Hyper({"descriptor.reduce_op": ["sum"]}),
            Hyper({"descriptor.normalize": [False]}),
            Hyper({"descriptor.reduce_by_type": [False]}),
            Hyper({"predictor.power": [1]}),
        )
        krr_ext_hyper = GridHyper(Hyper({"predictor.power": [1, 2, 3]}))
        rr_int_settings = GridHyper(
            Hyper({"descriptor_atomic.normalize": [False]}),
            Hyper({"descriptor.reduce_op": ["mean"]}),
            Hyper({"descriptor.normalize": [False]}),
            Hyper({"descriptor.reduce_by_type": [False]}),
            Hyper({"whiten.centre": [True]}),
            Hyper({"whiten.scale": [True]}),
        )
        rr_int_hyper = GridHyper(
            Hyper({"whiten.centre": whiten_hyper, "whiten.scale": whiten_hyper})
        )
        rr_ext_settings = GridHyper(
            Hyper({"descriptor_atomic.normalize": [False]}),
            Hyper({"descriptor.reduce_op": ["sum"]}),
            Hyper({"descriptor.normalize": [False]}),
            Hyper({"descriptor.reduce_by_type": [False]}),
            Hyper({"whiten.centre": [False]}),
            Hyper({"whiten.scale": [False]}),
        )
        rr_ext_hyper = GridHyper(
            Hyper({"whiten.centre": whiten_hyper, "whiten.scale": whiten_hyper})
        )
        models = []
        for hidx, updates in enumerate(krr_int_settings):
            for minimal in [True, False]:
                tag = "%s" % ("minimal" if minimal else "standard")
                model = make_gylm_krr(f"{mod_name}_gylm_{tag}_int_krr", minimal, extensive=False)
                model.hyperUpdate(updates)
                model.hyper.add(krr_int_hyper)
                models.append(model)
        for hidx, updates in enumerate(krr_ext_settings):
            for minimal in [True, False]:
                tag = "%s" % ("minimal" if minimal else "standard")
                model = make_gylm_krr(f"{mod_name}_gylm_{tag}_ext_krr", minimal, extensive=True)
                model.hyperUpdate(updates)
                model.hyper.add(krr_ext_hyper)
                models.append(model)
        for hidx, updates in enumerate(rr_int_settings):
            for minimal in [True, False]:
                tag = "%s" % ("minimal" if minimal else "standard")
                model = make_gylm_rr(f"{mod_name}_gylm_{tag}_int_rr", minimal, extensive=False)
                model.hyperUpdate(updates)
                model.hyper.add(rr_int_hyper)
                models.append(model)
        for hidx, updates in enumerate(rr_ext_settings):
            for minimal in [True, False]:
                tag = "%s" % ("minimal" if minimal else "standard")
                model = make_gylm_rr(f"{mod_name}_gylm_{tag}_ext_rr", minimal, extensive=True)
                model.hyperUpdate(updates)
                model.hyper.add(rr_ext_hyper)
                models.append(model)
        return models

    return customisable_compile_gylm


def get_acsf_krr_kwargs(scalerange, extensive, regularization_range):
    return dict(
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
                    "reduce": "np.sum(x, axis=0)" if extensive else "np.mean(x, axis=0)",
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
    )


def get_mbtr_rr_kwargs(extensive, whiten_hyper, regularization_range):
    return dict(
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
                    "reduce": "np.sum(x, axis=0)" if extensive else "np.mean(x, axis=0)",
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
            btf.Ridge(tag="predictor", inputs={"X": "whiten.X", "y": "input_norm.y"}),
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
    )


def get_acsf_rr_kwargs(scalerange, sharpness, extensive, whiten_hyper, regularization_range):
    return dict(
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
                    "reduce": "np.sum(x, axis=0)" if extensive else "np.mean(x, axis=0)",
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
            btf.Ridge(tag="predictor", inputs={"X": "whiten.X", "y": "input_norm.y"}),
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
    )


def get_mbtr_krr_kwargs(extensive, regularization_range):
    return dict(
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
                    "reduce": "np.sum(x, axis=0)" if extensive else "np.mean(x, axis=0)",
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
    )
