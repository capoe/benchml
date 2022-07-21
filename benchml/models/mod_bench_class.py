import numpy as np

import benchml.transforms as btf
from benchml.hyper import GridHyper, Hyper


def compile_physchem_class(custom_fields=None, with_hyper=False, **kwargs):
    if custom_fields is None:
        custom_fields = []
    models = []
    for descriptor_set in ["basic", "core", "logp", "extended"]:
        models.extend(
            [
                btf.Module(
                    tag="bmol_physchem_%s_rf_class" % descriptor_set,
                    transforms=[
                        btf.ExtXyzInput(tag="input"),
                        btf.Physchem2D(
                            tag="Physchem2D",
                            args={"select_predef": descriptor_set},
                            inputs={"configs": "input.configs"},
                        ),
                        btf.PhyschemUser(
                            tag="PhyschemUser",
                            args={"fields": custom_fields},
                            inputs={"configs": "input.configs"},
                        ),
                        btf.Concatenate(
                            tag="descriptor", inputs={"X": ["Physchem2D.X", "PhyschemUser.X"]}
                        ),
                        btf.RandomForestClassifier(
                            tag="predictor", inputs={"X": "descriptor.X", "y": "input.y"}
                        ),
                    ],
                    hyper=GridHyper(Hyper({"predictor.max_depth": [None]})),
                    broadcast={"meta": "input.meta"},
                    outputs={"y": "predictor.z"},
                ),
            ]
        )
    return models


def compile_ecfp_class():
    return [
        btf.Module(
            tag="bmol_ecfp_svm_class",
            transforms=[
                btf.ExtXyzInput(tag="input"),
                btf.MorganFP(
                    tag="descriptor",
                    args={"length": 4096, "radius": 2, "normalize": True},
                    inputs={"configs": "input.configs"},
                ),
                btf.KernelDot(tag="kernel", inputs={"X": "descriptor.X"}),
                btf.SupportVectorClassifier(
                    tag="predictor",
                    args={"C": None, "power": 2},
                    inputs={"K": "kernel.K", "y": "input.y"},
                ),
            ],
            hyper=GridHyper(
                Hyper(
                    {
                        "predictor.C": np.logspace(-9, +7, 17),
                    }
                ),
                Hyper({"predictor.power": [2.0]}),
            ),
            broadcast={"meta": "input.meta"},
            outputs={"y": "predictor.z"},
            doc=dict(
                summary="SVM Classifier",
                extended_summary="SVM Classifier with MorganFP.",
                model_category=btf.Module.categories.classification,
            ),
        ),
        btf.Module(
            tag="bmol_ecfp_lr_class",
            transforms=[
                btf.ExtXyzInput(tag="input"),
                btf.MorganFP(
                    tag="descriptor",
                    args={"length": 4096, "radius": 2, "normalize": True},
                    inputs={"configs": "input.configs"},
                ),
                btf.LogisticRegression(
                    tag="predictor", args={"C": None}, inputs={"X": "descriptor.X", "y": "input.y"}
                ),
            ],
            hyper=GridHyper(
                Hyper({"descriptor.length": [4096]}),
                Hyper({"descriptor.radius": [2]}),
                Hyper({"descriptor.normalize": [False]}),
                Hyper(
                    {
                        "predictor.C": np.logspace(-7, +4, 12),
                    }
                ),
            ),
            broadcast={"meta": "input.meta"},
            outputs={"y": "predictor.z"},
        ),
        btf.Module(
            tag="bmol_ecfp_rf_class",
            transforms=[
                btf.ExtXyzInput(tag="input"),
                btf.MorganFP(
                    tag="descriptor",
                    args={"length": 4096, "radius": 2, "normalize": True},
                    inputs={"configs": "input.configs"},
                ),
                btf.RandomForestClassifier(
                    tag="predictor",
                    args={"n_estimators": 20},
                    inputs={"X": "descriptor.X", "y": "input.y"},
                ),
            ],
            hyper=GridHyper(
                Hyper({"descriptor.length": [2048, 4096]}),
                Hyper({"descriptor.radius": [2, 3]}),
                Hyper({"descriptor.normalize": [False, True]}),
            ),
            broadcast={"meta": "input.meta"},
            outputs={"y": "predictor.z"},
        ),
        btf.Module(
            tag="bmol_ecfp_mplr_class",
            transforms=[
                btf.ExtXyzInput(tag="input"),
                btf.MorganFP(
                    tag="descriptor",
                    args={"length": 1024, "radius": 2, "normalize": True},
                    inputs={"configs": "input.configs"},
                ),
                btf.CleanMatrix(
                    tag="clean",
                    args={"axis": 0, "std_threshold": 1e-10},
                    inputs={"X": "descriptor.X"},
                ),
                btf.MarchenkoPasturFilter(tag="descriptor_mp", args={}, inputs={"X": "clean.X"}),
                btf.LogisticRegression(
                    tag="predictor",
                    args={"C": None},
                    inputs={"X": "descriptor_mp.X", "y": "input.y"},
                ),
            ],
            hyper=GridHyper(
                Hyper({"descriptor.length": [1024, 2048, 4096]}),
                Hyper({"descriptor.radius": [2]}),
                Hyper({"descriptor.normalize": [False]}),
                Hyper(
                    {
                        "predictor.C": np.logspace(-7, +4, 12),
                    }
                ),
            ),
            broadcast={"meta": "input.meta"},
            outputs={"y": "predictor.z"},
        ),
    ]


def compile_gylm_match_class(**kwargs):
    return [
        btf.Module(
            tag="bmol_gylm_match_class",
            transforms=[
                btf.ExtXyzInput(tag="input"),
                btf.GylmAtomic(
                    tag="descriptor", args={"heavy_only": True}, inputs={"configs": "input.configs"}
                ),
                btf.KernelSmoothMatch(tag="kernel", inputs={"X": "descriptor.X"}),
                btf.SupportVectorClassifier(
                    tag="predictor",
                    args={"C": None, "power": 2},
                    inputs={"K": "kernel.K", "y": "input.y"},
                ),
            ],
            hyper=GridHyper(
                Hyper(
                    {
                        "predictor.C": np.logspace(-9, +7, 17),
                    }
                ),
                Hyper({"predictor.power": [1.0, 2.0, 3.0, 4.0]}),
            ),
            broadcast={"meta": "input.meta"},
            outputs={"y": "predictor.z"},
        ),
        btf.Module(
            tag="bmol_gylm_match_class_norm",
            transforms=[
                btf.ExtXyzInput(tag="input"),
                btf.GylmAtomic(
                    tag="descriptor", args={"heavy_only": True}, inputs={"configs": "input.configs"}
                ),
                btf.KernelSmoothMatch(tag="kernel", inputs={"X": "descriptor.X"}),
                btf.SupportVectorClassifier(
                    tag="predictor",
                    args={"C": None, "power": 2},
                    inputs={"K": "kernel.K", "y": "input.y"},
                ),
                btf.RankNorm(tag="ranker", inputs={"z": "predictor.z"}),
            ],
            hyper=GridHyper(
                Hyper(
                    {
                        "predictor.C": np.logspace(-9, +7, 17),
                    }
                ),
                Hyper({"predictor.power": [1.0, 2.0, 3.0, 4.0]}),
            ),
            broadcast={"meta": "input.meta"},
            outputs={"y": "ranker.z"},
        ),
        btf.Module(
            tag="bmol_gylm_match_class_attr",
            transforms=[
                btf.ExtXyzInput(tag="input"),
                btf.GylmAtomic(
                    tag="descriptor", args={"heavy_only": True}, inputs={"configs": "input.configs"}
                ),
                btf.KernelSmoothMatch(tag="kernel", inputs={"X": "descriptor.X"}),
                btf.SupportVectorClassifier(
                    tag="predictor",
                    args={"C": None, "power": 2},
                    inputs={"K": "kernel.K", "y": "input.y"},
                ),
                btf.AttributeKernelSmoothMatchSVM(
                    tag="attribute",
                    args={"write_xyz": "attribution.xyz"},
                    inputs={
                        "configs": "input.configs",
                        "X": "kernel._X",
                        "X_probe": "descriptor.X",
                        "model": "predictor._model",
                    },
                ),
            ],
            hyper=GridHyper(
                Hyper({"predictor.C": np.logspace(-9, +7, 17)}),
                Hyper({"predictor.power": [2.0]}),
            ),
            broadcast={"meta": "input.meta"},
            outputs={"y": "predictor.y", "z": "predictor.z", "Z": "attribute.Z"},
        ),
    ]


def register_all():
    return {
        "bmol_physchem_class": compile_physchem_class,
        "bmol_ecfp_class": compile_ecfp_class,
        "bmol_gylm_match_class": compile_gylm_match_class,
    }
