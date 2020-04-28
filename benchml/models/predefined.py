import numpy as np
from ..transforms import *

def compile(groups):
    selected = [ model \
        for group in groups \
            for model in collections[group]() ]
    return selected

def compile_morgan():
    return [
        Module(
            tag="morgan_ridge",
            transforms=[
                ExtXyzInput(tag="input"),
                MorganFP(inputs={"configs": "input.configs"}),
                Ridge(inputs={"X": "MorganFP.X", "y": "input.y"})
            ],
            hypers=[
                Hyper({"MorganFP.length": [ 1024, 2048 ]}),
                Hyper({"Ridge.alpha": np.logspace(-2,2,5)})
            ],
            outputs={"y": "Ridge.y"}
        ),
        Module(
            tag="morgan_krr",
            transforms=[
                ExtXyzInput(tag="input"),
                MorganFP(
                    tag="desc", 
                    inputs={"configs": "input.configs"}),
                KernelDot(
                    tag="kern",
                    inputs={"X": "desc.X"}),
                KernelRidge(
                    args={"alpha": 1e-5, "power": 2},
                    inputs={"K": "kern.K", "y": "input.y"})
            ],
            hypers=[
                Hyper({ "KernelRidge.alpha": np.logspace(-3,+1, 5), }),
                Hyper({ "KernelRidge.power": [ 2. ] })
            ],
            broadcast={ "meta": "input.meta" },
            outputs={ "y": "KernelRidge.y" }
        ),
        Module(
            tag="morgan_krrx2",
            transforms=[
                ExtXyzInput(tag="input"),
                MorganKernel(
                    tag="A",
                    args={"x.fp_length": 1024, "x.fp_radius": 2},
                    inputs={"x.configs": "input.configs"}),
                MorganKernel(
                    tag="B",
                    args={"x.fp_length": 2048, "x.fp_radius": 4},
                    inputs={"x.configs": "input.configs"}),
                Add(
                    args={"coeffs": [ 0.5, 0.5 ]},
                    inputs={"X": ["A/k.K", "B/k.K"]}),
                KernelRidge(
                    args={"alpha": 0.1, "power": 2},
                    inputs={"K": "Add.y", "y": "input.y"})
            ],
            hypers=[
                Hyper({ "Add.coeffs": 
                    list(map(lambda f: [ f, 1.-f ], np.linspace(0.25, 0.75, 3)))
                }),
                Hyper({ "KernelRidge.alpha": 
                    np.logspace(-3,+1, 5),
                })
            ],
            broadcast={ "meta": "input.meta" },
            outputs={ "y": "KernelRidge.y" },
        ),
    ]

def compile_gylm():
    return [
        Module(
            tag="gylm",
            transforms=[
                ExtXyzInput(tag="input"),
                GylmAverage(
                    tag="desc", 
                    inputs={"configs": "input.configs"}),
                KernelDot(
                    tag="kern",
                    inputs={"X": "desc.X"}),
                KernelRidge(
                    args={"alpha": 1e-5, "power": 2},
                    inputs={"K": "kern.K", "y": "input.y"})
            ],
            hypers=[
                Hyper({ "KernelRidge.alpha": np.logspace(-5,+1, 7), }),
                Hyper({ "KernelRidge.power": [ 1., 2., 3., 6. ] })
            ],
            broadcast={ "meta": "input.meta" },
            outputs={ "y": "KernelRidge.y" }
        ),
    ]

collections = {
    "morgan": compile_morgan,
    "gylm": compile_gylm,
}
