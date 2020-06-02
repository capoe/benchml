from ..kernels import KernelDot
from ..pipeline import Transform, Macro
from ..logger import log, Mock
from ..ptable import lookup
import numpy as np
import multiprocessing as mp
import time
try:
    import gylm
except ImportError:
    gylm = None
try:
    import asaplib.hypers
    import asaplib.data
except ImportError:
    asaplib = Mock()
    asaplib.hypers = None
    asaplib.data = None

def check_gylmxx_available(obj, require=False):
    if gylm is None:
        if require: raise ImportError("%s requires gylmxx" % obj.__name__)
        return False
    return True

def check_asap_available(obj, require=False):
    if asaplib.hypers is None:
        if require: raise ImportError("%s requires asaplib" % obj.__name__)
        return False
    return True

class SoapGto(Transform):
    default_args = {
        "rcut": 5.0,
        "nmax": 9,
        "lmax": 6,
        "sigma": 1.0,
        "types": None,
        "crossover": True,
        "periodic": None,
        "normalize": False }
    req_inputs = ('configs',)
    allow_params = ("calc", "channel_dim")
    allow_stream = ("X", "T")
    stream_samples = ("X", "T")
    precompute = True
    log = log
    def check_available():
        return check_gylmxx_available(SoapGto)
    def mapMultiSoap(self, configs, centres, calcs):
        log = self.log
        t0 = time.time()
        X = []
        T = []
        for cidx, config in enumerate(configs):
            if log and log.verbose: log << log.back << \
                "%d/%d" % (cidx+1, len(configs)) << log.flush
            if centres is None:
                heavy, types_centres, pos_centres = config.getHeavy()
            else:
                pos_centres = centres[cidx]
                types_centres = None
            x = np.concatenate(
                [ dcalc.evaluate(system=config, positions=pos_centres) \
                    for dcalc in calcs ], axis=1)
            X.append(x)
            T.append(types_centres)
        t1 = time.time()
        if log and log.verbose:
            log << "[Finished in %fs]" % (t1-t0) << log.flush
        X = np.array(X)
        T = np.array(T)
        return T, X
    def _fit(self, inputs):
        if self.args["types"] is None:
            self.args["types"] = inputs["meta"]["elements"]
        if self.args["periodic"] is None:
            self.args["periodic"] = inputs["meta"]["periodic"]
        self.calc = gylm.SoapGtoCalculator(
            **self.args)
        self.channel_dim = self.args["nmax"]*self.args["nmax"]*(self.args["lmax"]+1)
        self.params().put("calc", self.calc)
        self.params().put("channel_dim", self.channel_dim)
        self._map(inputs)
    def _map(self, inputs):
        configs = inputs["configs"]
        dcalc = self.params().get("calc")
        centres = inputs["centres"] if "centres" in inputs else None
        T, X = self.mapMultiSoap(configs, centres, [ dcalc ])
        self.stream().put("X", X)
        self.stream().put("T", T)

class UniversalSoapGto(SoapGto):
    default_args = {
        "types": None,
        "periodic": None,
        "normalize": True,
        "crossover": True,
        "mode": "minimal" }
    allow_params = ("calcs",)
    def check_available():
        return SoapGto.check_available() \
            and check_asap_available(UniversalSoapGto)
    def updateParams(self, par, meta={}):
        out = {}
        out["types"] = par.pop('species')
        out["nmax"] = par.pop('n')
        out["lmax"] = par.pop('l')
        out["rcut"] = par.pop("cutoff")
        out["sigma"] = par.pop("atom_gaussian_width")
        out["normalize"] = self.args["normalize"]
        out["crossover"] = self.args["crossover"]
        out["periodic"] = meta["periodic"] if "periodic" in meta \
            else self.args["periodic"]
        return out
    def _fit(self, inputs):
        types = self.args["types"] if self.args["types"] is not None \
            else inputs["meta"]["elements"]
        types_z = [ lookup[t].z for t in types ]
        self.pars = asaplib.hypers.universal_soap_hyper(
            global_species=types_z, 
            fsoap_param=self.args["mode"], 
            dump=False)
        self.pars = [ self.updateParams(par, inputs["meta"]) \
            for key, par in sorted(self.pars.items()) ]
        calcs = [ gylm.SoapGtoCalculator(**par) for par in self.pars ]
        self.params().put("calcs", calcs)
        self._map(inputs)
    def _map(self, inputs):
        centres = inputs["centres"] if "centres" in inputs else None
        configs = inputs["configs"]
        T, X = self.mapMultiSoap(configs, centres, self.params().get("calcs"))
        print(X[0].shape)
        self.stream().put("T", T)
        self.stream().put("X", X)

class ReduceTypedMatrix(Transform):
    default_args = {
        "reduce_op": "np.sum(x, axis=0)",
        "normalize": False,
        "reduce_by_type": False,
        "types": None,
        "epsilon": 1e-10,
    }
    req_inputs = ("X",)
    allow_stream = ("X",)
    allow_params = ("types",)
    allow_ops = {"sum", "mean"}
    stream_samples = ("X",)
    def _setup(self, *args, **kwargs):
        assert self.args["reduce_op"] in self.allow_ops # Only 'sum' and 'mean' allowed
        if self.args["reduce_by_type"]:
            assert "T" in self.inputs # Require input T if reduce_by_type = True
    def _fit(self, inputs):
        if self.args["reduce_by_type"]:
            if self.args["types"] is not None:
                self.types = self.args["types"]
            else:
                self.types = inputs["meta"]["elements"]
            self.type_to_idx = { t: tidx for tidx, t in \
                enumerate(self.types) }
            self.params().put("types", self.types)
        self._map(inputs)
    def _map(self, inputs):
        X_red = []
        for idx, x in enumerate(inputs["X"]):
            if self.args["reduce_by_type"]:
                x_red = np.zeros((len(self.types), x.shape[1]))
                t_count = np.zeros((len(self.type_to_idx),))
                for i, t in enumerate(map(lambda t: self.type_to_idx[t], 
                        inputs["T"][idx])):
                    x_red[t] = x_red[t] + x[i]
                    t_count[t] += 1
                if self.args["reduce_op"] == "mean":
                    x_red = (x_red.T/(t_count+self.args["epsilon"])).T
                x_red = x_red.flatten()
            else:
                if self.args["reduce_op"] == "sum":
                    x_red = np.sum(x, axis=0)
                elif self.args["reduce_op"] == "mean":
                    x_red = np.mean(x, axis=0)
            if self.args["normalize"]:
                x_red = x_red/(np.sqrt(np.dot(x_red, x_red))+self.args["epsilon"])
            X_red.append(x_red)
        X_red = np.array(X_red) 
        self.stream().put("X", X_red)




