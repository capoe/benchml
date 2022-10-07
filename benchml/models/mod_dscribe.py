import numpy as np

import benchml.transforms as btf
from benchml.hyper import GridHyper, Hyper


def compile_dscribe(**kwargs):
    return [
        btf.Module(
            tag=str(DescriptorClass.__name__)[7:].lower() + "_rr",
            transforms=[
                btf.ExtXyzInput(tag="input"),
                DescriptorClass(tag="descriptor", inputs={"configs": "input.configs"}),
                btf.ReduceMatrix(tag="reduce", inputs={"X": "descriptor.X"}),
                btf.Ridge(tag="predictor", inputs={"X": "reduce.X", "y": "input.y"}),
            ],
            hyper=GridHyper(
                Hyper(
                    {
                        "predictor.alpha": np.logspace(-5, +5, 7),
                    }
                )
            ),
            broadcast={"meta": "input.meta"},
            outputs={"y": "predictor.y"},
        )
        for DescriptorClass in [
            btf.DscribeCM,
            btf.DscribeACSF,
            btf.DscribeMBTR,
            btf.DscribeLMBTR,
            btf.DscribeSineMatrix,
        ]
    ]


def register_all():
    return {"dscribe": compile_dscribe}
