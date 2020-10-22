from ..kernels import KernelDot
from ..pipeline import Transform, Macro
from ..logger import log
from ..utils import get_smiles
import numpy as np
from .plugin_check import *

class MorganFP(Transform):
    default_args = {
        "radius": 3,
        "length": 2048,
        "normalize": True }
    allow_stream = {'X'}
    stream_samples = ("X",)
    precompute = True
    def check_available():
        return check_rdkit_available(MorganFP)
    def _setup(self):
        self.radius = self.args["radius"]
        self.length = self.args["length"]
        self.normalize = self.args["normalize"]
    def _map(self, inputs, stream):
        configs = inputs["configs"]
        smiles = [ get_smiles(c) for c in configs ]
        mols = [ rchem.MolFromSmiles(s) for s in smiles ]
        fps = [ achem.GetMorganFingerprintAsBitVect(
            mol, radius=self.radius, nBits=self.length) for mol in mols ]
        fps = np.array(fps, dtype='float64')
        if self.normalize:
            z = 1./(np.sum(fps**2, axis=1)+1e-10)**0.5
            fps = (fps.T*z).T
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
        {
          "class": KernelDot,
          "tag": "k",
          "args": {},
          "inputs": {"X": "{self}x.X"}
        }
    ]
