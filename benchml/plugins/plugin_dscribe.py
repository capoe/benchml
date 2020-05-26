from ..pipeline import Transform, Macro
from ..logger import Mock
import numpy as np
try:
    import dscribe
    import dscribe.descriptors as dd
except ImportError:
    dscribe = None
    dd = Mock()
    dd.CoulombMatrix = None
    dd.ACSF = None

def check_dscribe_available(obj, require=False):
    if dscribe is None:
        if require: raise ImportError("'%s' requires dscribe" % obj.__class__.__name__)
        return False
    return True

class DscribeTransform(Transform):
    req_inputs = ("configs",)
    allow_params = ("calc",)
    allow_stream = ("X",)
    stream_samples = ("X",)
    precompute = True
    CalculatorClass = None
    def check_available():
        return check_dscribe_available(DscribeTransform)
    def _prepare(self, inputs):
        raise NotImplementedError()
        return {}
    def _fit(self, inputs):
        args = self._prepare(inputs)
        calc = self.CalculatorClass(**args)
        self.params().put("calc", calc)
        self._map(inputs)
    def _map(self, inputs):
        calc = self.params().get("calc")
        X = np.array([ calc.create(config) for config in inputs["configs"] ])
        self.stream().put("X", X)

class DscribeCM(DscribeTransform):
    default_args = {
        "n_atoms_max": None,
        "permutation": "sorted_l2",
        "sigma": None,
        "seed": None,
        "flatten": True,
        "sparse": False }
    CalculatorClass = dd.CoulombMatrix
    def _prepare(self, inputs):
        args = {}
        args.update(self.args)
        if self.args["n_atoms_max"] is None:
            n_max = 2*max([ len(c) for c in inputs["configs"] ])
            args["n_atoms_max"] = n_max
        return args

class DscribeACSF(DscribeTransform):
    default_args = {
        "species": None,
        "rcut": 6.0,
        "g2_params": [[1, 1], [1, 2], [1, 3]],
        "g3_params": None,
        "g4_params": [[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],
        "g5_params": None,
        "periodic": False,
        "sparse": False}
    CalculatorClass = dd.ACSF
    def _prepare(self, inputs):
        args = {}
        args.update(self.args)
        if "meta" in inputs:
            if "periodic" in inputs["meta"]:
                args["periodic"] = inputs["meta"]["periodic"]
            if "elements" in inputs["meta"]:
                args["species"] = inputs["meta"]["elements"]
        return args


