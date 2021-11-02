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


def get_bench_pdf_soap_krr_kwargs(minimal, whiten_hyper, regularization_range):
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
