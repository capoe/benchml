from ..pipeline import Transform, Macro
from ..logger import log, Mock
from ..ptable import lookup
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
    def _fit(self, inputs, stream, params):
        args = self._prepare(inputs)
        calc = self.CalculatorClass(**args)
        params.put("calc", calc)
        self._map(inputs, stream)
    def _map(self, inputs, stream):
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
        stream.put("X", X)

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

class UniversalDscribeACSF(DscribeTransform):
    default_args = {
        "adjust_to_species": None,
        "scalerange": 1.0,
        "sharpness": 1.0,
        "verbose": False,
        "cutoff": None,
        "species": None,
        "periodic": False,
        "n_select_g2": None,
        "n_select_g4": None
    }
    CalculatorClass = dd.ACSF
    def check_available():
        return check_dscribe_available(UniversalDscribeACSF) \
            and check_asap_available(UniversalDscribeACSF)
    def _prepare(self, inputs):
        # Find "universal" hyperparameters for this set of elements
        if self.args["adjust_to_species"] is None:
            types = self.args["species"] if self.args["species"] is not None \
                else inputs["meta"]["elements"]
        else:   
            types = self.args["adjust_to_species"]
        types_z = [ lookup[t].z for t in types ]
        paramsets = asaplib.hypers.gen_default_acsf_hyperparameters(
            Zs=types_z,
            scalerange=self.args["scalerange"],
            sharpness=self.args["sharpness"],
            verbose=self.args["verbose"],
            cutoff=self.args["cutoff"])
        assert len(paramsets) == 1 # No multiresolution ACSF implemented
        pars = paramsets[list(paramsets.keys())[0]]
        rcut = pars.pop("cutoff")
        _ = pars.pop("type")
        # Collect in args
        args = {
            "rcut": rcut,
            **pars }
        # Add boundary settings and named elements
        if "meta" in inputs:
            if "periodic" in inputs["meta"]:
                args["periodic"] = inputs["meta"]["periodic"]
            if "elements" in inputs["meta"]:
                args["species"] = inputs["meta"]["elements"]
        # Subselect
        if self.args["n_select_g2"] is not None:
            log << log.mr << "WARNING Using random subselection" << log.endl
            s2 = np.arange(len(args["g2_params"]))
            np.random.shuffle(s2)
            s2 = s2[0:self.args["n_select_g2"]]
            args["g2_params"] = [ args["g2_params"][_] for _ in s2 ]
        if self.args["n_select_g4"] is not None:
            log << log.mr << "WARNING Using random subselection" << log.endl
            s4 = np.arange(len(args["g4_params"]))
            np.random.shuffle(s4)
            s4 = s4[0:self.args["n_select_g4"]]
            args["g4_params"] = [ args["g4_params"][_] for _ in s4 ]
        for key, val in args.items():
            log << "  Set %-15s = %s" % (
                key, str(val) if type(val) is not list \
                    else "[...%d items...]" % len(val)) << log.endl
        return args

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
