import numpy as np
from ..transforms import *

def compile_logd_extensive(**kwargs):
    return [
        Module(
            tag="logd_gylm_minimal_rr",
            transforms=[
                ExtXyzInput(
                    tag="input"),
                GylmAtomic(
                    tag="descriptor_atomic",
                    args={
                        "normalize": False,
                        "rcut": 3.0,
                        "rcut_width": 0.5,
                        "nmax": 6,
                        "lmax": 4,
                        "sigma": 0.75,
                        "part_sigma": 0.5,
                        "wconstant": False,
                        "wscale": 1.0,
                        "wcentre": 1.0,
                        "ldamp": 4,
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
                    args={ "alpha": 1e2 },
                    inputs={"X": "whiten.X", "y": "input.y"}) ],
            hyper=GridHyper(
                Hyper({ "predictor.alpha": np.logspace(-1, +5, 7), })),
            broadcast={"meta": "input.meta"},
            outputs={ "y": "predictor.y" }
        ),
        Module(
            tag="logd_gylm_hybrid_rr",
            transforms=[
                ExtXyzInput(
                    tag="input"),
                GylmAtomic(
                    tag="descriptor_atomic",
                    args={
                        "normalize": False,
                        "rcut": 5.5,
                        "rcut_width": 0.5,
                        "nmax": 9,
                        "lmax": 6,
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
                    tag="descriptor_struct",
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
                CxCalcTransform(tag="cx",
                    args={"reshape_as_matrix": True},
                    inputs={"configs": "input.configs"}),
                Concatenate(tag="descriptor",
                    inputs={"X": [ "descriptor_struct.X", "cx.X" ]}),
                WhitenMatrix(
                    tag="whiten",
                    inputs={
                        "X": "descriptor.X"
                    }),
                Ridge(
                    tag="predictor",
                    args={ "alpha": 1e2 },
                    inputs={"X": "whiten.X", "y": "input.y"}) ],
            hyper=GridHyper(
                Hyper({ "predictor.alpha": np.logspace(-1, +5, 7), })),
            broadcast={"meta": "input.meta"},
            outputs={ "y": "predictor.y" }
        ),
        Module(
            tag="logd_gylm_extended_rr",
            transforms=[
                ExtXyzInput(
                    tag="input"),
                GylmAtomic(
                    tag="descriptor_atomic",
                    args={
                        "normalize": False,
                        "rcut": 5.5,
                        "rcut_width": 0.5,
                        "nmax": 9,
                        "lmax": 6,
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
                    args={ "alpha": 1e2 },
                    inputs={"X": "whiten.X", "y": "input.y"}) ],
            hyper=GridHyper(
                Hyper({ "predictor.alpha": np.logspace(-1, +5, 7), })),
            broadcast={"meta": "input.meta"},
            outputs={ "y": "predictor.y" }
        ),
    ]

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
            tag="logd_delta_hybrid_topo_krr",
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
            tag="logd_delta_hybrid_gylm_krr",
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
            tag="hybrid_logd_topo",
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
            tag="logd_topo_gp",
            transforms=[
                ExtXyzInput(tag="input"),
                CxCalcTransform(tag="cx",
                    args={"reshape_as_matrix": True},
                    inputs={"configs": "input.configs"}),
                KernelGaussian(
                    tag="kern_gaussian",
                    args={"self_kernel": True},
                    inputs={"X": "cx.X"}),
                MorganFP(
                    tag="desc",
                    args={"length": 4096, "radius": 2, "normalize": True},
                    inputs={"configs": "input.configs"}),
                KernelDot(
                    tag="kern",
                    args={"self_kernel": True},
                    inputs={"X": "desc.X"}),
                Add(
                    tag="add_k",
                    args={"coeffs": [0.5,0.5]},
                    inputs={"X": ["kern_gaussian.K", "kern.K"] }),
                Add(
                    tag="add_k_self",
                    args={"coeffs": [0.5,0.5]},
                    inputs={"X": ["kern_gaussian.K_self", "kern.K_self"] }),
                ResidualGaussianProcess(
                    tag="gp",
                    args={"alpha": 1e-5, "power": 2},
                    inputs={"K": "add_k.y", "K_self": "add_k_self.y", "y": "input.y"}),
            ],
            hyper=GridHyper(
                Hyper({ "desc.radius": [ 2 ] }),
                Hyper({ "kern_gaussian.scale": [ 1., 2. ] }),
                Hyper({ "gp.alpha": np.logspace(-5,+1, 7), }),
                Hyper(
                    { 
                      "add_k.coeffs": [ [0.25,0.75] ],
                      "add_k_self.coeffs": [ [0.25,0.75] ] 
                    }
                ),
                Hyper({ "gp.power": [ 2. ] })
            ),
            broadcast={ "meta": "input.meta" },
            outputs={ "y":  "gp.y", "dy": "gp.dy", "dk": "gp.dk" }
        ),
        Module(
            tag="logd_hybrid_topo_gp",
            transforms=[
                ExtXyzInput(tag="input"),
                CxCalcTransform(tag="cx",
                    args={"reshape_as_matrix": True},
                    inputs={"configs": "input.configs"}),
                KernelGaussian(
                    tag="kern_gaussian",
                    args={"self_kernel": True},
                    inputs={"X": "cx.X"}),
                MorganFP(
                    tag="desc",
                    args={"length": 4096, "radius": 2, "normalize": True},
                    inputs={"configs": "input.configs"}),
                KernelDot(
                    tag="kern",
                    args={"self_kernel": True},
                    inputs={"X": "desc.X"}),
                Add(
                    tag="add_k",
                    args={"coeffs": [0.5,0.5]},
                    inputs={"X": ["kern_gaussian.K", "kern.K"] }),
                Add(
                    tag="add_k_self",
                    args={"coeffs": [0.5,0.5]},
                    inputs={"X": ["kern_gaussian.K_self", "kern.K_self"] }),
                GaussianProcess(
                    args={"alpha": 1e-5, "power": 2},
                    inputs={"K": "add_k.y", "K_self": "add_k_self.y", "y": "input.y"}),
                #GaussianProcessRegressor(
                #    tag="GaussianProcess",
                #    args={"alpha": 1e-5, "power": 2},
                #    inputs={"K": "add_k.y", "K_self": "add_k_self.y", "y": "input.y"}),
            ],
            hyper=GridHyper(
                Hyper({ "desc.radius": [ 2 ] }),
                Hyper({ "GaussianProcess.alpha": np.logspace(-5,+1, 7), }),
                Hyper({ "kern_gaussian.scale": [ 1., 2. ] }),
                Hyper(
                    { 
                      "add_k.coeffs": [ [0.25,0.75] ],
                      "add_k_self.coeffs": [ [0.25,0.75] ] 
                    }
                ),
                Hyper({ "GaussianProcess.power": [ 2. ] })
            ),
            broadcast={ "meta": "input.meta" },
            outputs={ 
                "y":  "GaussianProcess.y", 
                "dy": "GaussianProcess.dy",
                "dy_rank": "GaussianProcess.dy_rank",
                "dy_zscore": "GaussianProcess.dy_zscore"}
        ),
        Module(
            tag="logd_hybrid_gylm_krr",
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
            tag="logd_delta_gylm_krr",
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
            tag="delta_logd_topo",
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
        "logd_extensive": compile_logd_extensive,
        "logd": compile_logd
    }
