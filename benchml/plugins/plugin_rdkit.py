import numpy as np

from benchml.kernels import KernelDot
from benchml.pipeline import Macro, Transform
from benchml.plugins.plugin_check import achem, check_rdkit_available, rchem
from benchml.utils import get_smiles


class MorganFP(Transform):
    default_args = {
        "radius": 3,
        "length": 2048,
        "normalize": True,
        "useChirality": False,
        "useFeatures": False,
    }
    allow_stream = {"X"}
    stream_samples = ("X",)
    precompute = True

    def check_available(self, *args, **kwargs):
        return check_rdkit_available(self, *args, **kwargs)

    def _setup(self):
        self.radius = self.args["radius"]
        self.length = self.args["length"]
        self.normalize = self.args["normalize"]
        self.useChirality = self.args["useChirality"]
        self.useFeatures = self.args["useFeatures"]

    def _map(self, inputs, stream):
        configs = inputs["configs"]
        smiles = [get_smiles(c) for c in configs]
        mols = [rchem.MolFromSmiles(s) for s in smiles]  # pylint: disable=E1101
        fps = [
            achem.GetMorganFingerprintAsBitVect(  # pylint: disable=E1101
                mol,
                radius=self.radius,
                nBits=self.length,
                useChirality=self.useChirality,
                useFeatures=self.useFeatures,
            )
            for mol in mols
        ]
        fps = np.array(fps, dtype="float64")
        if self.normalize:
            z = 1.0 / (np.sum(fps ** 2, axis=1) + 1e-10) ** 0.5
            fps = (fps.T * z).T
        stream.put("X", fps)


class MorganKernel(Macro):
    req_inputs = ("x.configs",)
    transforms = [
        {
            "class": MorganFP,
            "tag": "x",
            "args": {"length": 1024, "radius": 3},
            "inputs": {"configs": "?"},
        },
        {"class": KernelDot, "tag": "k", "args": {}, "inputs": {"X": "{self}x.X"}},
    ]
