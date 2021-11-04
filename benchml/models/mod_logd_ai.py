import numpy as np

import benchml.transforms as btf
from benchml.hyper import GridHyper, Hyper
from benchml.models.common import get_logd_hybrid_topo_gp_kwargs


def compile_logd_ai(*args, **kwargs):
    return [
        btf.Module(tag="logd_ai_hybrid_topo_cxlogp_gp", **get_logd_hybrid_topo_gp_kwargs()),
        btf.Module(
            tag="logd_ai_hybrid_topo_rdlogp_gp",
            transforms=[
                btf.ExtXyzInput(tag="input"),
                btf.Physchem2D(
                    tag="Physchem2D",
                    args={"select": ["mollogp"]},
                    inputs={"configs": "input.configs"},
                ),
                btf.KernelGaussian(
                    tag="kern_gaussian", args={"self_kernel": True}, inputs={"X": "Physchem2D.X"}
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
                    tag="add_k_self",
                    args={"coeffs": [0.5, 0.5]},
                    inputs={"X": ["kern_gaussian.K_self", "kern.K_self"]},
                ),
                btf.GaussianProcess(
                    args={"alpha": 1e-5, "power": 2},
                    inputs={"K": "add_k.y", "K_self": "add_k_self.y", "y": "input.y"},
                ),
            ],
            hyper=GridHyper(
                Hyper({"desc.radius": [2]}),
                Hyper(
                    {
                        "GaussianProcess.alpha": np.logspace(-5, +1, 7),
                    }
                ),
                Hyper({"kern_gaussian.scale": [1.0, 2.0]}),
                Hyper({"add_k.coeffs": [[0.25, 0.75]], "add_k_self.coeffs": [[0.25, 0.75]]}),
                Hyper({"GaussianProcess.power": [2.0]})
                # Hyper({ "desc.radius": [ 2 ] }),
                # Hyper({ "GaussianProcess.alpha": [ 0.1 ], }),
                # Hyper({ "kern_gaussian.scale": [ 2. ] }),
                # Hyper(
                #    {
                #      "add_k.coeffs": [ [0.25,0.75] ],
                #      "add_k_self.coeffs": [ [0.25,0.75] ]
                #    }
                # ),
                # Hyper({ "GaussianProcess.power": [ 2. ] })
            ),
            broadcast={"meta": "input.meta"},
            outputs={
                "y": "GaussianProcess.y",
                "dy": "GaussianProcess.dy",
                "dy_rank": "GaussianProcess.dy_rank",
                "dy_zscore": "GaussianProcess.dy_zscore",
            },
        ),
    ]


def register_all():
    return {
        "logd_ai": compile_logd_ai,
    }
