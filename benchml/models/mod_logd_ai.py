import numpy as np
from ..transforms import *

def compile_logd_ai(*args, **kwargs):
    return [
        Module(
            tag="logd_ai_hybrid_topo_cxlogp_gp",
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
                "dy_zscore": "GaussianProcess.dy_zscore"}),
        Module(
            tag="logd_ai_hybrid_topo_rdlogp_gp",
            transforms=[
                ExtXyzInput(tag="input"),
                Physchem2D(tag="Physchem2D",
                    args={"select": ["mollogp"]},
                    inputs={"configs": "input.configs"}),
                KernelGaussian(
                    tag="kern_gaussian",
                    args={"self_kernel": True},
                    inputs={"X": "Physchem2D.X"}),
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
                "dy_zscore": "GaussianProcess.dy_zscore"}),
        ]

def register_all():
    return {
        "logd_ai": compile_logd_ai,
    }
