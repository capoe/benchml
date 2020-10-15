import numpy as np
from ..transforms import *

def compile_xy_regressors(*args, **kwargs):
    return [
        Module(
            tag="xy_rf_regressor",
            transforms=[
                XyInput(tag="input"),
                RandomForestRegressor(tag="predictor",
                    inputs={"X": "input.X", "y": "input.y"}),
            ],
            hyper=GridHyper(
                Hyper({"predictor.max_depth": [None]})),
            broadcast={},
            outputs={"y": "predictor.y"}),
        ]

def compile_xy_classifiers(*args, **kwargs):
    return [
        Module(
            tag="xy_rf_classifier",
            transforms=[
                XyInput(tag="input"),
                RandomForestClassifier(tag="predictor",
                    inputs={"X": "input.X", "y": "input.y"}),
            ],
            hyper=GridHyper(
                Hyper({"predictor.max_depth": [None]})),
            broadcast={},
            outputs={"y": "predictor.y"}),
        ]

def register_all():
    return {
        "xy_regressors": compile_xy_regressors,
        "xy_classifiers": compile_xy_classifiers
    }
