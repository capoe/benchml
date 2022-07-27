import benchml.transforms as btf
from benchml.hyper import GridHyper, Hyper


def compile_xy_regressors(*args, **kwargs):
    return [
        btf.Module(
            tag="xy_rfr",
            transforms=[
                btf.ExttInput(tag="input"),
                btf.RandomForestRegressor(tag="predictor", inputs={"X": "input.X", "y": "input.Y"}),
            ],
            hyper=GridHyper(Hyper({"predictor.max_depth": [None]})),
            broadcast={},
            outputs={"y": "predictor.y"},
        ),
    ]


def compile_xy_classifiers(*args, **kwargs):
    return [
        btf.Module(
            tag="xy_rf_class",
            transforms=[
                btf.ExttInput(tag="input"),
                btf.RandomForestClassifier(
                    tag="predictor", inputs={"X": "input.X", "y": "input.Y"}
                ),
            ],
            hyper=GridHyper(Hyper({"predictor.max_depth": [None]})),
            broadcast={},
            outputs={"y": "predictor.y"},
            doc=dict(
                model_category=btf.Module.categories.classification,
            ),
        ),
    ]


def register_all():
    return {"xy_regressors": compile_xy_regressors, "xy_classifiers": compile_xy_classifiers}
