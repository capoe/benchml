from ..kernels import KernelDot
from ..pipeline import Transform, Macro
from ..logger import log, Mock
from ..ptable import lookup
import numpy as np
import multiprocessing as mp
import time
from .plugin_check import *

class SoapBase(Transform):
    default_args = {
        "rcut": 5.0,
        "nmax": 9,
        "lmax": 6,
        "sigma": 1.0,
        "types": None,
        "crossover": True,
        "periodic": None,
        "power": True,
        "normalize": False }
    req_inputs = ('configs',)
    allow_params = ("calc", "channel_dim")
    allow_stream = ("X", "T")
    stream_samples = ("X", "T")
    precompute = True
    log = log
    def check_available():
        return check_gylmxx_available(__class__)
    def evaluateSingle(self, calc, config, pos_centres):
        raise NotImplementedError("Missing method overload")
    def mapMultiSoap(self, configs, centres, calcs):
        log = self.log
        t0 = time.time()
        X = []
        T = []
        for cidx, config in enumerate(configs):
            if log and log.verbose: log << log.back << \
                "%d/%d" % (cidx+1, len(configs)) << log.flush
            if centres is None:
                # heavy, types_centres, pos_centres = config.getHeavy() # HACK
                pos_centres = None
                types_centres = config.symbols
            else:
                pos_centres = centres[cidx]
                types_centres = None
            x = np.concatenate(
                [ self.evaluateSingle(dcalc, config, pos_centres) \
                    for dcalc in calcs ], axis=1)
            X.append(x)
            T.append(types_centres)
        t1 = time.time()
        if log and log.verbose:
            log << "[Finished in %fs]" % (t1-t0) << log.flush
        X = np.array(X)
        T = np.array(T)
        return T, X
    def _fit(self, inputs, stream, params):
        if self.args["types"] is None:
            self.args["types"] = inputs["meta"]["elements"]
        if self.args["periodic"] is None:
            self.args["periodic"] = inputs["meta"]["periodic"]
        self.calc = gylm.SoapGtoCalculator(
            **self.args)
        self.channel_dim = self.args["nmax"]*self.args["nmax"]*(self.args["lmax"]+1)
        params.put("calc", self.calc)
        params.put("channel_dim", self.channel_dim)
        self._map(inputs, stream)
    def _map(self, inputs, stream):
        configs = inputs["configs"]
        dcalc = self.params().get("calc")
        centres = inputs["centres"] if "centres" in inputs else None
        T, X = self.mapMultiSoap(configs, centres, [ dcalc ])
        stream.put("X", X)
        stream.put("T", T)

class SoapGylmxx(SoapBase):
    def evaluateSingle(self, dcalc, config, centres):
        return dcalc.evaluate(system=config, positions=centres)

class UniversalSoapBase(SoapBase):
    default_args = {
        "types": None,
        "periodic": None,
        "normalize": True,
        "crossover": True,
        "power": True,
        "mode": "minimal" }
    allow_params = ("calcs",)
    CalculatorClass = None
    def check_available():
        return SoapBase.check_available() \
            and check_asap_available(__class__)
    def _fit(self, inputs, stream, params):
        types = self.args["types"] if self.args["types"] is not None \
            else inputs["meta"]["elements"]
        types_z = [ lookup[t].z for t in types ]
        self.pars = asaplib.hypers.universal_soap_hyper(
            global_species=types_z,
            fsoap_param=self.args["mode"],
            dump=False)
        self.pars = [ self.updateParams(par, inputs["meta"]) \
            for key, par in sorted(self.pars.items()) ]
        calcs = [ self.CalculatorClass(**par) for par in self.pars ]
        params.put("calcs", calcs)
        self._map(inputs, stream)
    def _map(self, inputs, stream):
        centres = inputs["centres"] if "centres" in inputs else None
        configs = inputs["configs"]
        T, X = self.mapMultiSoap(configs, centres, self.params().get("calcs"))
        stream.put("T", T)
        stream.put("X", X)

class UniversalSoapGylmxx(UniversalSoapBase):
    CalculatorClass = gylm.SoapGtoCalculator
    def check_available():
        return UniversalSoapBase.check_available \
            and check_gylmxx_available(__class__)
    def updateParams(self, par, meta={}):
        out = {}
        out["types"] = par.pop('species')
        out["nmax"] = par.pop('n')
        out["lmax"] = par.pop('l')
        out["rcut"] = par.pop("cutoff")
        out["sigma"] = par.pop("atom_gaussian_width")
        out["normalize"] = self.args["normalize"]
        out["crossover"] = self.args["crossover"]
        out["power"] = self.args["power"]
        out["periodic"] = meta["periodic"] if "periodic" in meta \
            else self.args["periodic"]
        return out
    def evaluateSingle(self, dcalc, config, centres):
        return dcalc.evaluate(system=config, positions=centres)

class UniversalSoapDscribe(UniversalSoapBase):
    CalculatorClass = dd.SOAP
    def check_available():
        return UniversalSoapBase.check_available \
            and check_dscribe_available(UniversalSoapDscribe)
    def updateParams(self, par, meta={}):
        out = {}
        out["species"] = par.pop('species')
        out["nmax"] = par.pop('n')
        out["lmax"] = par.pop('l')
        out["rcut"] = par.pop("cutoff")
        out["sigma"] = par.pop("atom_gaussian_width")
        out["crossover"] = self.args["crossover"]
        out["sparse"] = False
        out["periodic"] = meta["periodic"] if "periodic" in meta \
            else self.args["periodic"]
        return out
    def evaluateSingle(self, dcalc, config, centres):
        return dcalc.create(system=config, positions=centres)
