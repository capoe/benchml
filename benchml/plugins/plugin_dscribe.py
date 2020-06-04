from ..pipeline import Transform, Macro
from ..logger import log, Mock
import numpy as np
from .plugin_check import *

class DscribeTransform(Transform):
    req_inputs = ("configs",)
    allow_params = ("calc",)
    allow_stream = ("X",)
    stream_samples = ("X",)
    precompute = True
    CalculatorClass = None
    verbose = True
    log = log
    def check_available():
        return check_dscribe_available(DscribeTransform)
    def _prepare(self, inputs):
        args = {}
        args.update(self.args)
        if "meta" in inputs:
            if "periodic" in inputs["meta"]:
                args["periodic"] = inputs["meta"]["periodic"]
            if "elements" in inputs["meta"]:
                args["species"] = inputs["meta"]["elements"]
        return args
    def _fit(self, inputs):
        args = self._prepare(inputs)
        calc = self.CalculatorClass(**args)
        self.params().put("calc", calc)
        self._map(inputs)
    def _map(self, inputs):
        calc = self.params().get("calc")
        X = []
        for cidx, config in enumerate(inputs["configs"]):
            x = calc.create(config)
            if DscribeTransform.verbose:
                log << log.back << " %d/%d %-50s" % (
                    cidx, len(inputs["configs"]), str(config.symbols)) \
                    << x.shape << log.flush
            X.append(x)
        if DscribeTransform.verbose:
            log << log.endl
        X = np.array(X)
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

class DscribeMBTR(DscribeTransform):
    default_args = dict(
        species=None,
        k1={
            "geometry": {"function": "atomic_number"},
            "grid": {"min": 0, "max": 8, "n": 10, "sigma": 0.1},
        },
        k2={
            "geometry": {"function": "inverse_distance"},
            "grid": {"min": 0, "max": 1, "n": 10, "sigma": 0.1},
            "weighting": {"function": "exponential", "scale": 0.5, "cutoff": 1e-3},
        },
        k3={
            "geometry": {"function": "cosine"},
            "grid": {"min": -1, "max": 1, "n": 10, "sigma": 0.1},
            "weighting": {"function": "exponential", "scale": 0.5, "cutoff": 1e-3},
        },
        flatten=True,
        sparse=False,
        periodic=False,
        normalization="l2_each",
    )
    CalculatorClass = dd.MBTR

class DscribeLMBTR(DscribeTransform):
    default_args = dict(
        species=None,
        k2={
            "geometry": {"function": "distance"},
            "grid": {"min": 0, "max": 5, "n": 100, "sigma": 0.1},
            "weighting": {"function": "exponential", "scale": 0.5, "cutoff": 1e-3},
        },
        k3={
            "geometry": {"function": "angle"},
            "grid": {"min": 0, "max": 180, "n": 100, "sigma": 0.1},
            "weighting": {"function": "exponential", "scale": 0.5, "cutoff": 1e-3},
        },
        periodic=False,
        flatten=True,
        sparse=False,
        normalization="l2_each"
    )
    CalculatorClass = dd.LMBTR

class DscribeSineMatrix(DscribeTransform):
    default_args = dict(
        n_atoms_max=None,
        permutation="sorted_l2",
        sigma=None,
        sparse=False,
        flatten=True,
        seed=None
    )
    CalculatorClass = dd.SineMatrix
    def _prepare(self, inputs):
        args = {}
        args.update(self.args)
        if self.args["n_atoms_max"] is None:
            n_max = 2*max([ len(c) for c in inputs["configs"] ])
            args["n_atoms_max"] = n_max
        return args

class DscribeEwaldSumMatrix(DscribeTransform):
    default_args = dict(
        n_atoms_max=None,
        permutation="sorted_l2",
        sigma=None,
        sparse=False,
        flatten=True,
        seed=None
    )
    CalculatorClass = dd.EwaldSumMatrix
    def _prepare(self, inputs):
        args = {}
        args.update(self.args)
        if self.args["n_atoms_max"] is None:
            n_max = 2*max([ len(c) for c in inputs["configs"] ])
            args["n_atoms_max"] = n_max
        return args
