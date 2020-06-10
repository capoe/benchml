import numpy as np
from ..transforms import *

def compile_dscribe(**kwargs):
    return [
        Module(
            tag=DescriptorClass.__name__+"_ridge",
            transforms=[
                ExtXyzInput(tag="input"),
                DescriptorClass(
                    tag="descriptor",
                    inputs={"configs": "input.configs"}),
                ReduceMatrix(
                    tag="reduce",
                    inputs={"X": "descriptor.X"}),
                Ridge(
                    tag="predictor",
                    inputs={"X": "reduce.X", "y": "input.y"}) ],
            hyper=GridHyper(
                Hyper({ "predictor.alpha": np.logspace(-5,+5, 7), })),
            broadcast={"meta": "input.meta"},
            outputs={ "y": "predictor.y" }) \
        for DescriptorClass in [ DscribeCM, DscribeACSF, DscribeMBTR, DscribeLMBTR ]
    ]

def compile_dscribe_periodic(**kwargs):
    return [
        Module(
            tag=DescriptorClass.__name__+"_ridge",
            transforms=[
                ExtXyzInput(tag="input"),
                DescriptorClass(
                    tag="descriptor",
                    inputs={"configs": "input.configs"}),
                ReduceMatrix(
                    tag="reduce",
                    inputs={"X": "descriptor.X"}),
                Ridge(
                    tag="predictor",
                    inputs={"X": "reduce.X", "y": "input.y"}) ],
            hyper=GridHyper(
                Hyper({ "predictor.alpha": np.logspace(-5,+5, 7), })),
            broadcast={"meta": "input.meta"},
            outputs={ "y": "predictor.y" }) \
        for DescriptorClass in [ DscribeSineMatrix ]
    ]

def register_all():
    return {
        "dscribe": compile_dscribe,
        "dscribe_periodic": compile_dscribe_periodic,
    }
