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
