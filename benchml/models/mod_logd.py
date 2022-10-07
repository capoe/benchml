import numpy as np

import benchml.transforms as btf
from benchml.hyper import GridHyper, Hyper
from benchml.models.common import get_logd_hybrid_topo_gp_kwargs


def compile_logd_extensive(**kwargs):
    return [
        btf.Module(
            tag="logd_gylm_minimal_rr",
            transforms=[
                btf.ExtXyzInput(tag="input"),
                btf.GylmAtomic(
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
                    inputs={"X": "descriptor_atomic.X"},
                ),
                btf.WhitenMatrix(tag="whiten", inputs={"X": "descriptor.X"}),
                btf.Ridge(
                    tag="predictor", args={"alpha": 1e2}, inputs={"X": "whiten.X", "y": "input.y"}
                ),
            ],
            hyper=GridHyper(
                Hyper(
                    {
                        "predictor.alpha": np.logspace(-1, +5, 7),
                    }
                )
            ),
            broadcast={"meta": "input.meta"},
            outputs={"y": "predictor.y"},
        ),
        btf.Module(
            tag="logd_gylm_hybrid_rr",
            transforms=[
                btf.ExtXyzInput(tag="input"),
                btf.GylmAtomic(
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
                    inputs={"configs": "input.configs"},
                ),
                btf.ReduceTypedMatrix(
                    tag="descriptor_struct",
                    args={
                        "reduce_op": "sum",
                        "normalize": False,
                        "reduce_by_type": False,
                        "types": None,
                        "epsilon": 1e-10,
                    },
                    inputs={"X": "descriptor_atomic.X"},
                ),
                btf.CxCalcTransform(
                    tag="cx", args={"reshape_as_matrix": True}, inputs={"configs": "input.configs"}
                ),
                btf.Concatenate(tag="descriptor", inputs={"X": ["descriptor_struct.X", "cx.X"]}),
                btf.WhitenMatrix(tag="whiten", inputs={"X": "descriptor.X"}),
                btf.Ridge(
                    tag="predictor", args={"alpha": 1e2}, inputs={"X": "whiten.X", "y": "input.y"}
                ),
            ],
            hyper=GridHyper(
                Hyper(
                    {
                        "predictor.alpha": np.logspace(-1, +5, 7),
                    }
                )
            ),
            broadcast={"meta": "input.meta"},
            outputs={"y": "predictor.y"},
        ),
        btf.Module(
            tag="logd_gylm_extended_rr",
            transforms=[
                btf.ExtXyzInput(tag="input"),
                btf.GylmAtomic(
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
                    inputs={"X": "descriptor_atomic.X"},
                ),
                btf.WhitenMatrix(tag="whiten", inputs={"X": "descriptor.X"}),
                btf.Ridge(
                    tag="predictor", args={"alpha": 1e2}, inputs={"X": "whiten.X", "y": "input.y"}
                ),
            ],
            hyper=GridHyper(
                Hyper(
                    {
                        "predictor.alpha": np.logspace(-1, +5, 7),
                    }
                )
            ),
            broadcast={"meta": "input.meta"},
            outputs={"y": "predictor.y"},
        ),
    ]


def compile_logd(custom_fields=None, **kwargs):
    if custom_fields is None:
        custom_fields = []
    return [
        btf.Module(
            tag="logd_lr",
            transforms=[
                btf.ExtXyzInput(tag="input"),
                btf.CxCalcTransform(
                    tag="cx", args={"reshape_as_matrix": True}, inputs={"configs": "input.configs"}
                ),
                btf.LinearRegression(inputs={"X": "cx.X", "y": "input.y"}),
            ],
            hyper=GridHyper(Hyper({"LinearRegression.normalize": [False, True]})),
            outputs={"y": "LinearRegression.y"},
        ),
        btf.Module(
            tag="logd_physchem_rfr",
            transforms=[
                btf.ExtXyzInput(tag="input"),
                btf.Physchem2D(tag="Physchem2D", inputs={"configs": "input.configs"}),
                btf.CxCalcTransform(
                    tag="cx", args={"reshape_as_matrix": True}, inputs={"configs": "input.configs"}
                ),
                btf.PhyschemUser(
                    tag="PhyschemUser",
                    args={"fields": custom_fields},
                    inputs={"configs": "input.configs"},
                ),
                btf.Concatenate(
                    tag="desc", inputs={"X": ["Physchem2D.X", "PhyschemUser.X", "cx.X"]}
                ),
                btf.RandomForestRegressor(tag="pred", inputs={"X": "desc.X", "y": "input.y"}),
            ],
            hyper=GridHyper(Hyper({"pred.max_depth": [None]})),
            broadcast={"meta": "input.meta"},
            outputs={"y": "pred.y"},
        ),
        btf.Module(
            tag="logd_delta_hybrid_topo_krr",
            transforms=[
                btf.ExtXyzInput(tag="input"),
                btf.CxCalcTransform(
                    tag="cx_alt",
                    args={"reshape_as_matrix": False},
                    inputs={"configs": "input.configs"},
                ),
                btf.Delta(inputs={"target": "input.y", "ref": "cx_alt.X"}),
                btf.CxCalcTransform(
                    tag="cx", args={"reshape_as_matrix": True}, inputs={"configs": "input.configs"}
                ),
                btf.KernelGaussian(tag="kern_gaussian", inputs={"X": "cx.X"}),
                btf.MorganFP(
                    tag="desc",
                    args={"length": 4096, "radius": 2, "normalize": True},
                    inputs={"configs": "input.configs"},
                ),
                btf.KernelDot(tag="kern", inputs={"X": "desc.X"}),
                btf.Add(
                    tag="kern_combo",
                    args={"coeffs": [0.5, 0.5]},
                    inputs={"X": ["kern_gaussian.K", "kern.K"]},
                ),
                btf.KernelRidge(
                    args={"alpha": 1e-5, "power": 2}, inputs={"K": "kern_combo.y", "y": "Delta.y"}
                ),
                btf.Add(
                    tag="out",
                    args={"coeffs": [1.0, 1.0]},
                    inputs={"X": ["cx_alt.X", "KernelRidge.y"]},
                ),
            ],
            hyper=GridHyper(
                Hyper({"desc.radius": [2]}),
                Hyper(
                    {
                        "KernelRidge.alpha": np.logspace(-6, +1, 8),
                    }
                ),
                Hyper({"kern_gaussian.scale": [2.0]}),
                Hyper({"kern_combo.coeffs": [[f, 1.0 - f] for f in [1.0 / 3.0]]}),
                Hyper({"KernelRidge.power": [2.0]}),
            ),
            broadcast={"meta": "input.meta"},
            outputs={"y": "out.y"},
        ),
        btf.Module(
            tag="logd_delta_hybrid_gylm_krr",
            transforms=[
                btf.ExtXyzInput(tag="input"),
                btf.CxCalcTransform(
                    tag="cx_alt",
                    args={"reshape_as_matrix": False},
                    inputs={"configs": "input.configs"},
                ),
                btf.Delta(inputs={"target": "input.y", "ref": "cx_alt.X"}),
                btf.Reshape(tag="cx", args={"shape": [-1, 1]}, inputs={"X": "cx_alt.X"}),
                btf.KernelGaussian(tag="kern_gaussian", inputs={"X": "cx.X"}),
                btf.GylmAverage(tag="desc", inputs={"configs": "input.configs"}),
                btf.KernelDot(tag="kern", inputs={"X": "desc.X"}),
                btf.Add(
                    tag="kern_combo",
                    args={"coeffs": [0.5, 0.5]},
                    inputs={"X": ["kern_gaussian.K", "kern.K"]},
                ),
                btf.KernelRidge(
                    args={"alpha": 1e-2, "power": 2}, inputs={"K": "kern_combo.y", "y": "Delta.y"}
                ),
                btf.Add(
                    tag="out",
                    args={"coeffs": [1.0, 1.0]},
                    inputs={"X": ["cx_alt.X", "KernelRidge.y"]},
                ),
            ],
            hyper=GridHyper(
                Hyper(
                    {
                        "KernelRidge.alpha": np.logspace(-5, +1, 7),
                    }
                ),
                Hyper({"kern_gaussian.scale": [2.0]}),
                Hyper({"kern_combo.coeffs": [[1.0 / 3.0, 2.0 / 3.0]]}),
                Hyper({"KernelRidge.power": [2.0]}),
            ),
            # hyper=BayesianHyper(
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
            broadcast={"meta": "input.meta"},
            outputs={"y": "out.y"},
        ),
        btf.Module(
            tag="logd_hybrid_topo_krr",
            transforms=[
                btf.ExtXyzInput(tag="input"),
                btf.CxCalcTransform(
                    tag="cx", args={"reshape_as_matrix": True}, inputs={"configs": "input.configs"}
                ),
                btf.KernelGaussian(tag="kern_gaussian", inputs={"X": "cx.X"}),
                btf.MorganFP(
                    tag="desc",
                    args={"length": 4096, "radius": 2, "normalize": True},
                    inputs={"configs": "input.configs"},
                ),
                btf.KernelDot(tag="kern", inputs={"X": "desc.X"}),
                btf.Add(args={"coeffs": [0.5, 0.5]}, inputs={"X": ["kern_gaussian.K", "kern.K"]}),
                btf.KernelRidge(
                    args={"alpha": 1e-5, "power": 2}, inputs={"K": "Add.y", "y": "input.y"}
                ),
            ],
            hyper=GridHyper(
                Hyper({"desc.radius": [2]}),
                Hyper(
                    {
                        "KernelRidge.alpha": np.logspace(-5, +1, 7),
                    }
                ),
                Hyper({"kern_gaussian.scale": [1.0, 2.0]}),
                Hyper({"Add.coeffs": [[0.25, 0.75]]}),
                Hyper({"KernelRidge.power": [2.0]}),
            ),
            broadcast={"meta": "input.meta"},
            outputs={"y": "KernelRidge.y"},
        ),
        btf.Module(
            tag="logd_hybrid_topo_rgp",
            transforms=[
                btf.ExtXyzInput(tag="input"),
                btf.CxCalcTransform(
                    tag="cx", args={"reshape_as_matrix": True}, inputs={"configs": "input.configs"}
                ),
                btf.KernelGaussian(
                    tag="kern_gaussian", args={"self_kernel": True}, inputs={"X": "cx.X"}
                ),
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
                    tag="add_k_diag",
                    args={"coeffs": [0.5, 0.5]},
                    inputs={"X": ["kern_gaussian.K_diag", "kern.K_diag"]},
                ),
                btf.ResidualGaussianProcess(
                    tag="gp",
                    args={"alpha": 1e-5, "power": 2},
                    inputs={"K": "add_k.y", "K_diag": "add_k_diag.y", "y": "input.y"},
                ),
            ],
            hyper=GridHyper(
                Hyper({"desc.radius": [2]}),
                Hyper({"kern_gaussian.scale": [1.0, 2.0]}),
                Hyper(
                    {
                        "gp.alpha": np.logspace(-5, +1, 7),
                    }
                ),
                Hyper({"add_k.coeffs": [[0.25, 0.75]], "add_k_diag.coeffs": [[0.25, 0.75]]}),
                Hyper({"gp.power": [2.0]}),
            ),
            broadcast={"meta": "input.meta"},
            outputs={"y": "gp.y", "dy": "gp.dy", "dk": "gp.dk"},
        ),
        btf.Module(tag="logd_hybrid_topo_gp", **get_logd_hybrid_topo_gp_kwargs()),
        btf.Module(
            tag="logd_hybrid_gylm_krr",
            transforms=[
                btf.ExtXyzInput(tag="input"),
                btf.CxCalcTransform(
                    tag="cx", args={"reshape_as_matrix": True}, inputs={"configs": "input.configs"}
                ),
                btf.KernelGaussian(tag="kern_gaussian", args={"scale": 1.0}, inputs={"X": "cx.X"}),
                btf.GylmAverage(tag="desc", inputs={"configs": "input.configs"}),
                btf.KernelDot(tag="kern", inputs={"X": "desc.X"}),
                btf.Add(args={"coeffs": [0.5, 0.5]}, inputs={"X": ["kern_gaussian.K", "kern.K"]}),
                btf.KernelRidge(
                    args={"alpha": 1e-5, "power": 2}, inputs={"K": "Add.y", "y": "input.y"}
                ),
            ],
            # hyper=BayesianHyper(
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
                Hyper(
                    {
                        "KernelRidge.alpha": np.logspace(-5, +1, 7),
                    }
                ),
                Hyper({"kern_gaussian.scale": [2.0]}),
                Hyper({"Add.coeffs": [[0.25, 0.75]]}),
                Hyper({"kern.power": [1.0]}),
                Hyper({"KernelRidge.power": [2.0]}),
            ),
            broadcast={"meta": "input.meta"},
            outputs={"y": "KernelRidge.y"},
        ),
        btf.Module(
            tag="logd_delta_gylm_krr",
            transforms=[
                btf.ExtXyzInput(tag="input"),
                btf.CxCalcTransform(tag="cx", inputs={"configs": "input.configs"}),
                btf.Delta(inputs={"target": "input.y", "ref": "cx.X"}),
                btf.GylmAverage(tag="desc", inputs={"configs": "input.configs"}),
                btf.KernelDot(inputs={"X": "desc.X"}),
                btf.KernelRidge(
                    args={"alpha": 1e-5, "power": 2}, inputs={"K": "KernelDot.K", "y": "Delta.y"}
                ),
                btf.Add(args={"coeffs": [1.0, 1.0]}, inputs={"X": ["cx.X", "KernelRidge.y"]}),
            ],
            # hyper=BayesianHyper(
            #    Hyper({ "KernelRidge.alpha": np.linspace(-5,+1, 7), }),
            #    Hyper({ "KernelRidge.power": [ 1., 4. ] }),
            #    init_points=10,
            #    n_iter=30,
            #    convert={
            #        "KernelRidge.alpha": "lambda p: 10**p"}),
            hyper=GridHyper(
                Hyper(
                    {
                        "KernelRidge.alpha": np.logspace(-5, +1, 7),
                    }
                ),
                Hyper({"KernelRidge.power": [2.0]}),
                init_points=10,
                n_iter=30,
                convert={"KernelRidge.alpha": "lambda p: 10**p"},
            ),
            broadcast={"meta": "input.meta"},
            outputs={"y": "Add.y"},
        ),
        btf.Module(
            tag="logd_delta_topo_krr",
            transforms=[
                btf.ExtXyzInput(tag="input"),
                btf.CxCalcTransform(tag="cx", inputs={"configs": "input.configs"}),
                btf.Delta(inputs={"target": "input.y", "ref": "cx.X"}),
                btf.MorganFP(
                    tag="desc",
                    args={"length": 4096, "radius": 2, "normalize": True},
                    inputs={"configs": "input.configs"},
                ),
                btf.KernelDot(tag="kern", inputs={"X": "desc.X"}),
                btf.KernelRidge(
                    args={"alpha": 1e-5, "power": 2}, inputs={"K": "kern.K", "y": "Delta.y"}
                ),
                btf.Add(args={"coeffs": [1.0, 1.0]}, inputs={"X": ["cx.X", "KernelRidge.y"]}),
            ],
            hyper=GridHyper(
                Hyper({"desc.radius": [2]}),
                Hyper(
                    {
                        "KernelRidge.alpha": np.logspace(-5, +1, 7),
                    }
                ),
                Hyper({"KernelRidge.power": [2.0]}),
            ),
            broadcast={"meta": "input.meta"},
            outputs={"y": "Add.y"},
        ),
    ]


def register_all():
    return {"logd_extensive": compile_logd_extensive, "logd": compile_logd}
