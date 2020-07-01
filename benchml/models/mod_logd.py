import numpy as np
from ..transforms import *

def compile_logd(custom_fields=[], **kwargs):
    return [
        Module(
            tag="logp_linear",
            transforms=[
                ExtXyzInput(tag="input"),
                CxCalcTransform(tag="cx",
                    args={"reshape_as_matrix": True},
                    inputs={"configs": "input.configs"}),
                LinearRegression(inputs={"X": "cx.X", "y": "input.y"})
            ],
            hyper=GridHyper(
                Hyper({"LinearRegression.normalize": [False, True]})
            ),
            outputs={"y": "LinearRegression.y"}
        ),
        Module(
            tag="logd_physchem_rf",
            transforms=[
                ExtXyzInput(tag="input"),
                Physchem2D(tag="Physchem2D",
                    inputs={"configs": "input.configs"}),
                CxCalcTransform(tag="cx",
                    args={"reshape_as_matrix": True},
                    inputs={"configs": "input.configs"}),
                PhyschemUser(tag="PhyschemUser",
                    args={
                        "fields": custom_fields
                    },
                    inputs={"configs": "input.configs"}),
                Concatenate(tag="desc",
                    inputs={"X": [ "Physchem2D.X", "PhyschemUser.X", "cx.X" ]}),
                RandomForestRegressor(tag="pred",
                    inputs={"X": "desc.X", "y": "input.y"})
            ],
            hyper=GridHyper(
                Hyper({"pred.max_depth": [None]})
            ),
            broadcast={"meta": "input.meta"},
            outputs={"y": "pred.y"}
        ),
        Module(
            tag="delta_kernelcombo_logd_morgan",
            transforms=[
                ExtXyzInput(tag="input"),
                CxCalcTransform(tag="cx_alt",
                    args={"reshape_as_matrix": False},
                    inputs={"configs": "input.configs"}),
                Delta(
                    inputs={"target": "input.y", "ref": "cx_alt.X"}),
                CxCalcTransform(tag="cx",
                    args={"reshape_as_matrix": True},
                    inputs={"configs": "input.configs"}),
                KernelGaussian(
                    tag="kern_gaussian",
                    inputs={"X": "cx.X"}),
                MorganFP(
                    tag="desc",
                    args={"length": 4096, "radius": 2, "normalize": True},
                    inputs={"configs": "input.configs"}),
                KernelDot(
                    tag="kern",
                    inputs={"X": "desc.X"}),
                Add(tag="kern_combo",
                    args={"coeffs": [0.5,0.5]},
                    inputs={"X": ["kern_gaussian.K", "kern.K"] }),
                KernelRidge(
                    args={"alpha": 1e-5, "power": 2},
                    inputs={"K": "kern_combo.y", "y": "Delta.y"}),
                Add(
                    tag="out",
                    args={"coeffs": [1.,1.]},
                    inputs={"X": ["cx_alt.X", "KernelRidge.y"] })
            ],
            hyper=GridHyper(
                Hyper({ "desc.radius": [ 2 ] }),
                Hyper({ "KernelRidge.alpha": np.logspace(-6,+1, 8), }),
                Hyper({ "kern_gaussian.scale": [ 2. ] }),
                Hyper({ "kern_combo.coeffs": [ [f, 1.-f] \
                    for f in [ 1./3. ] ] }),
                Hyper({ "KernelRidge.power": [ 2. ] })
            ),
            broadcast={ "meta": "input.meta" },
            outputs={ "y":  "out.y" }
        ),
        Module(
            tag="delta_kernelcombo_logd_gylm",
            transforms=[
                ExtXyzInput(tag="input"),
                CxCalcTransform(tag="cx_alt",
                    args={"reshape_as_matrix": False},
                    inputs={"configs": "input.configs"}),
                Delta(
                    inputs={"target": "input.y", "ref": "cx_alt.X"}),
                CxCalcTransform(tag="cx",
                    args={"reshape_as_matrix": True},
                    inputs={"configs": "input.configs"}),
                KernelGaussian(
                    tag="kern_gaussian",
                    inputs={"X": "cx.X"}),
                GylmAverage(
                    tag="desc",
                    inputs={"configs": "input.configs"}),
                KernelDot(
                    tag="kern",
                    inputs={"X": "desc.X"}),
                Add(tag="kern_combo",
                    args={"coeffs": [0.5,0.5]},
                    inputs={"X": ["kern_gaussian.K", "kern.K"] }),
                KernelRidge(
                    args={"alpha": 1e-2, "power": 2},
                    inputs={"K": "kern_combo.y", "y": "Delta.y"}),
                Add(
                    tag="out",
                    args={"coeffs": [1.,1.]},
                    inputs={"X": ["cx_alt.X", "KernelRidge.y"] })
            ],
            hyper=GridHyper(
                Hyper({ "KernelRidge.alpha": np.logspace(-5,+1, 7), }),
                Hyper({ "kern_gaussian.scale": [ 2. ] }),
                Hyper({ "kern_combo.coeffs": [ [ 1./3., 2./3. ] ] }),
                Hyper({ "KernelRidge.power": [ 2. ] })
            ),
            #hyper=BayesianHyper(
            #    Hyper({ "KernelRidge.alpha": np.linspace(-5,+1, 7), }),
            #    Hyper({ "kern_gaussian.scale": [ 2., 2. ] }),
            #    Hyper({ "kern_combo.coeffs": [ 0.25, 0.25 ] }),
            #    Hyper({ "KernelRidge.power": [ 1., 6. ] }),
            #    init_points=10,
            #    n_iter=30,
            #    convert={
            #        "KernelRidge.alpha": "lambda p: 10**p",
            #        "kern_combo.coeffs": "lambda f: np.array([f, 1-f])"
            #    }),
            broadcast={ "meta": "input.meta" },
            outputs={ "y":  "out.y" }
        ),
        Module(
            tag="kernelcombo_logd_morgan",
            transforms=[
                ExtXyzInput(tag="input"),
                CxCalcTransform(tag="cx",
                    args={"reshape_as_matrix": True},
                    inputs={"configs": "input.configs"}),
                KernelGaussian(
                    tag="kern_gaussian",
                    inputs={"X": "cx.X"}),
                MorganFP(
                    tag="desc",
                    args={"length": 4096, "radius": 2, "normalize": True},
                    inputs={"configs": "input.configs"}),
                KernelDot(
                    tag="kern",
                    inputs={"X": "desc.X"}),
                Add(
                    args={"coeffs": [0.5,0.5]},
                    inputs={"X": ["kern_gaussian.K", "kern.K"] }),
                KernelRidge(
                    args={"alpha": 1e-5, "power": 2},
                    inputs={"K": "Add.y", "y": "input.y"}),
            ],
            hyper=GridHyper(
                Hyper({ "desc.radius": [ 2 ] }),
                Hyper({ "KernelRidge.alpha": np.logspace(-5,+1, 7), }),
                Hyper({ "kern_gaussian.scale": [ 1., 2. ] }),
                Hyper({ "Add.coeffs": [ [0.25,0.75] ] }),
                Hyper({ "KernelRidge.power": [ 2. ] })
            ),
            broadcast={ "meta": "input.meta" },
            outputs={ "y":  "KernelRidge.y" }
        ),
        Module(
            tag="kernelcombo_logd_gylm",
            transforms=[
                ExtXyzInput(tag="input"),
                CxCalcTransform(tag="cx",
                    args={"reshape_as_matrix": True},
                    inputs={"configs": "input.configs"}),
                KernelGaussian(
                    tag="kern_gaussian",
                    args={"scale": 1.},
                    inputs={"X": "cx.X"}),
                GylmAverage(
                    tag="desc",
                    inputs={"configs": "input.configs"}),
                KernelDot(
                    tag="kern",
                    inputs={"X": "desc.X"}),
                Add(
                    args={"coeffs": [0.5,0.5]},
                    inputs={"X": ["kern_gaussian.K", "kern.K"] }),
                KernelRidge(
                    args={"alpha": 1e-5, "power": 2},
                    inputs={"K": "Add.y", "y": "input.y"}),
            ],
            #hyper=BayesianHyper(
            #    Hyper({ "KernelRidge.alpha": np.linspace(-5,+1, 7), }),
            #    Hyper({ "kern_gaussian.scale": [ 0.25, 2. ] }),
            #    Hyper({ "Add.coeffs": [ 0.0, 1.0 ] }),
            #    Hyper({ "KernelRidge.power": [ 1., 4. ] }),
            #    init_points=10,
            #    n_iter=30,
            #    convert={
            #        "KernelRidge.alpha": "lambda p: 10**p",
            #        "Add.coeffs": "lambda f: np.array([f, 1-f])"
            #    }),
            hyper=GridHyper(
                Hyper({ "KernelRidge.alpha": np.logspace(-5,+1, 7), }),
                Hyper({ "kern_gaussian.scale": [ 2. ] }),
                Hyper({ "Add.coeffs": [ [0.25,0.75] ] }),
                Hyper({ "kern.power": [ 1. ] }),
                Hyper({ "KernelRidge.power": [ 2. ] })
            ),
            broadcast={ "meta": "input.meta" },
            outputs={ "y":  "KernelRidge.y" }
        ),
        Module(
            tag="delta_logd_gylm",
            transforms=[
                ExtXyzInput(tag="input"),
                CxCalcTransform(tag="cx",
                    inputs={"configs": "input.configs"}),
                Delta(
                    inputs={"target": "input.y", "ref": "cx.X"}),
                GylmAverage(
                    tag="desc",
                    inputs={"configs": "input.configs"}),
                KernelDot(
                    inputs={"X": "desc.X"}),
                KernelRidge(
                    args={"alpha": 1e-5, "power": 2},
                    inputs={"K": "KernelDot.K", "y": "Delta.y"}),
                Add(
                    args={"coeffs": [1.,1.]},
                    inputs={"X": ["cx.X", "KernelRidge.y"] })
            ],
            #hyper=BayesianHyper(
            #    Hyper({ "KernelRidge.alpha": np.linspace(-5,+1, 7), }),
            #    Hyper({ "KernelRidge.power": [ 1., 4. ] }),
            #    init_points=10,
            #    n_iter=30,
            #    convert={
            #        "KernelRidge.alpha": "lambda p: 10**p"}),
            hyper=GridHyper(
                Hyper({ "KernelRidge.alpha": np.logspace(-5,+1, 7), }),
                Hyper({ "KernelRidge.power": [ 2. ] }),
                init_points=10,
                n_iter=30,
                convert={
                    "KernelRidge.alpha": "lambda p: 10**p"}),
            broadcast={ "meta": "input.meta" },
            outputs={ "y": "Add.y" }
        ),
        Module(
            tag="delta_logd_morgan",
            transforms=[
                ExtXyzInput(tag="input"),
                CxCalcTransform(tag="cx",
                    inputs={"configs": "input.configs"}),
                Delta(
                    inputs={"target": "input.y", "ref": "cx.X"}),
                MorganFP(
                    tag="desc",
                    args={"length": 4096, "radius": 2, "normalize": True},
                    inputs={"configs": "input.configs"}),
                KernelDot(
                    tag="kern",
                    inputs={"X": "desc.X"}),
                KernelRidge(
                    args={"alpha": 1e-5, "power": 2},
                    inputs={"K": "kern.K", "y": "Delta.y"}),
                Add(
                    args={"coeffs": [1.,1.]},
                    inputs={"X": ["cx.X", "KernelRidge.y"] })
            ],
            hyper=GridHyper(
                Hyper({ "desc.radius": [ 2 ] }),
                Hyper({ "KernelRidge.alpha": np.logspace(-5,+1, 7), }),
                Hyper({ "KernelRidge.power": [ 2. ] })),
            broadcast={ "meta": "input.meta" },
            outputs={ "y": "Add.y" }
        ),
    ]

def register_all():
    return {
        "logd": compile_logd
    }
