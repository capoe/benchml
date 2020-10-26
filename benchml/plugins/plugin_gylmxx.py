from ..kernels import KernelDot
from ..pipeline import Transform, Macro
from ..logger import log
import numpy as np
import multiprocessing as mp
import time
from .plugin_check import *

class KernelSmoothMatch(Transform):
    default_args = {
        "base_kernel": "xi.dot(xj.T)",
        "base_power": 3,
        "gamma": 1e-2,
        "epsilon": 1e-6
    }
    req_inputs = ("X",)
    allow_stream = ("K",)
    allow_params = ("X",)
    stream_kernel = ("K",)
    precompute = True
    verbose = True
    log = log
    def check_available():
        return check_gylmxx_available(__class__)
    def evaluate(self, X1, X2, symmetric):
        K = np.zeros((X1.shape[0], X2.shape[0]))
        for i in range(X1.shape[0]):
            xi = X1[i]
            for j in range(i if symmetric else 0, X2.shape[0]):
                xj = X2[j]
                if self.verbose: log << log.back << " Match %4d/%-4d   " % (i, j) << log.flush
                kij = eval(self.args["base_kernel"])**self.args["base_power"]
                pij = np.zeros_like(kij)
                gylm.smooth_match(pij, kij, kij.shape[0], kij.shape[1],
                    self.args["gamma"], self.args["epsilon"], self.verbose)
                K[i,j] = np.sum(kij*pij)
                if symmetric: K[j,i] = K[i,j]
        if self.verbose: log << log.endl
        return K
    def _fit(self, inputs, stream, params):
        X = inputs["X"]
        K = self.evaluate(X, X, True)
        stream.put("K", K)
        params.put("X", np.copy(inputs["X"]))
    def _map(self, inputs, stream):
        X1 = inputs["X"]
        X2 = self.params().get("X")
        K = self.evaluate(X1, X2, False)
        stream.put("K", K)

class AttributeKernelSmoothMatchSVM(Transform):
    req_inputs = ("configs", "X", "X_probe", "z_probe", "model")
    default_args = {
        "write_xyz": "attribution.xyz",
        "gamma": "@kernel.gamma",
        "epsilon": "@kernel.epsilon",
        "base_kernel": "@kernel.base_kernel",
        "base_power": "@kernel.base_power",
        "power": "@predictor.power"
    }
    allow_stream = ("Z",)
    verbose = True
    def _map(self, inputs, stream):
        configs = inputs["configs"]
        model = inputs["model"]
        X_probe = inputs["X_probe"]
        X = inputs["X"]
        z_probe = inputs["z_probe"]
        if stream.parent is not None:
            stream.put("Z", None)
            return
        Z = []
        log << log.endl
        for i in range(len(X_probe)):
            log << "Attribute %d/%d" % (i+1, len(X_probe)) << log.endl
            xi = X_probe[i]
            k_attr = np.zeros((len(xi), len(X)))
            for j in range(len(X)):
                xj = X[j]
                kij = eval(self.args["base_kernel"])**self.args["base_power"]
                pij = np.zeros_like(kij)
                gylm.smooth_match(pij, kij, kij.shape[0], kij.shape[1],
                    self.args["gamma"], self.args["epsilon"], False)
                k_attr[:,j] = np.sum(kij*pij, axis=1)
            k = np.sum(k_attr, axis=0)
            ksub = k[model.support_]
            ksub_attr = k_attr[:,model.support_]
            w = ksub**(self.args["power"]-1)*model.dual_coef_[0]
            z_attr = model.intercept_/ksub_attr.shape[0] + ksub_attr.dot(w)
            Z.append(z_attr)
        if self.args["write_xyz"] != "":
            assert configs is not None # Require configs input to produce xyz
            from ..readwrite import write
            import json
            for cidx, config in enumerate(configs):
                config.info["z_attr"] = list(Z[cidx].tolist())
            write('attribution.xyz', configs)
        stream.put("Z", np.array(Z))

class GylmTransform(Transform):
    default_args = {
        "procs": 1,
        "rcut": 5.0,
        "rcut_width": 0.5,
        "nmax": 9,
        "lmax": 6,
        "sigma": 0.75,
        "part_sigma": 0.5,
        "wconstant": False,
        "wscale": 0.5,
        "wcentre": 0.5,
        "ldamp": 0.5,
        "power": True,
        "types": None,
        "heavy_only": False,
        "normalize": True}
    req_inputs = ('configs',)
    allow_params = ("calc",)
    allow_stream = ("X",)
    stream_samples = ("X",)
    precompute = True
    log = None
    def check_available():
        return check_gylmxx_available(GylmAverage)
    def _setup(self, *args):
        self.procs = self.args.pop("procs", 1)
        self.heavy_only = self.args.pop("heavy_only", True)
        if self.args["types"] is None:
            self.calc = None
        else:
            self.calc = gylm.GylmCalculator(**self.args)
    def _fit(self, inputs, stream, params):
        if self.args["types"] is None:
            self.args["types"] = inputs["meta"]["elements"]
        self.calc = gylm.GylmCalculator(
            **self.args)
        params.put("calc", self.calc)
        self._map(inputs, stream)

class NlocX(Transform):
    default_args = {
        "gylmb": {
            "rcut": 7.5,
            "rcut_width": 0.5,
            "nmax": 10,
            "lmax": 6,
            "sigma": 1.0,
            "part_sigma": 0.5,
            "wconstant": False,
            "wscale": 0.5,
            "wcentre": 0.5,
            "ldamp": 0.5,
            "power": False,
            "normalize": True,
            "types": ["C", "N", "O", "S", "H"]},
        "gylma": {
            "rcut": 7.5, # TODO Think about this
            "rcut_width": 0.5,
            "nmax": 10,
            "lmax": 6,
            "sigma": 1.0,
            "part_sigma": 0.5,
            "wconstant": False,
            "wscale": 0.5,
            "wcentre": 0.5,
            "ldamp": 0.5,
            "power": False,
            "normalize": True,
            "types": ["C", "N", "O", "S", "H"]
        }
    }
    def check_available():
        return check_gylmxx_available(NlocX)
    def _map(self, inputs, stream):
        calc_a = gylm.GylmCalculator(
            **self.args["gylma"])
        calc_b = gylm.GylmCalculator(
            **self.args["gylmb"])
        cA_list = inputs["centres_A"]
        cB_list = inputs["centres_B"]
        configs = inputs["configs"]
        for c in range(len(configs)):
            cA = cA_list[c]
            cB = cB_list[c]
            config = configs[c]
            xa = calc_a.evaluate(system=config, positions=cA)
            xb = calc_b.evaluate(system=config, positions=cB)
            print(len(config), len(cA), len(cB), "=>", xa.shape, xb.shape)
        return

class GylmAverage(GylmTransform):
    def _map(self, inputs, stream):
        if not hasattr(self, "heavy_only"): # NOTE For backwards-compatibility only
            self.heavy_only = True
        X = gylm_evaluate(
            configs=inputs["configs"],
            dcalc=self.calc,
            heavy_only=self.heavy_only,
            reduce_molecular=np.sum,
            norm_molecular=True,
            centres=inputs.pop("centres", None))
        stream.put("X", X)

class GylmAtomic(GylmTransform):
    def _map(self, inputs, stream):
        if not hasattr(self, "heavy_only"): # TODO For backwards-compatibility only
            self.heavy_only = True
        X = gylm_evaluate(
            configs=inputs["configs"],
            dcalc=self.calc,
            heavy_only=self.heavy_only,
            reduce_molecular=None,
            norm_molecular=False,
            centres=inputs.pop("centres", None))
        stream.put("X", X)

def gylm_evaluate_single(args):
    config = args["config"]
    dcalc = args["dcalc"]
    centres = args["centres"]
    if centres is None:
        heavy, types_centres, pos_centres = config.getHeavy()
    else:
        pos_centres = centres
    x = dcalc.evaluate(system=config, positions=pos_centres)
    return x

def gylm_evaluate_mp(
        configs,
        dcalc,
        procs,
        reduce_molecular=None,
        norm_molecular=False,
        centres=None):
    log = GylmTransform.log
    t0 = time.time()
    args_list = [ {
        "config": configs[i],
        "dcalc": dcalc,
        "centres": centres if centres is None else centres[i] } \
            for i in range(len(configs)) ]
    pool = mp.Pool(processes=procs)
    X = pool.map(gylm_evaluate_single, args_list)
    pool.close()
    for i in range(len(X)):
        x = X[i]
        if reduce_molecular is not None:
            x = reduce_molecular(x, axis=0)
        if norm_molecular:
            x = x/np.dot(x,x)**0.5
    t1 = time.time()
    if log:
        log << "[MP: Finished in %fs]" % (t1-t0) << log.flush
    X = np.array(X)
    return X

def gylm_evaluate(
        configs,
        dcalc,
        reduce_molecular=None,
        norm_molecular=False,
        heavy_only=True,
        centres=None):
    #log = GylmTransform.log
    log.verbose = False
    t0 = time.time()
    X = []
    for cidx, config in enumerate(configs):
        if log and log.verbose: log << log.back << \
            "%d/%d" % (cidx+1, len(configs)) << log.flush
        if centres is not None:
            pos_centres = centres[cidx]
        elif heavy_only:
            heavy, types_centres, pos_centres = config.getHeavy()
        else:
            pos_centres = config.positions
        x = dcalc.evaluate(system=config,
            positions=pos_centres)
        if reduce_molecular is not None:
            x = reduce_molecular(x, axis=0)
        if norm_molecular:
            x = x/np.dot(x,x)**0.5
        X.append(x)
    t1 = time.time()
    if log:
        log << "[Finished in %fs]" % (t1-t0) << log.flush
    X = np.array(X)
    return X

class GylmReduceConvolve(Transform):
    req_args = {
        "nmax", "lmax", "types"
    }
    default_args = {
        "nmax": None,
        "lmax": None,
        "types": None,
        "epsilon": 1e-10,
        "normalize": True
    }
    req_inputs = {
        "Q"
    }
    allow_stream = ("X",)
    stream_samples = ("X",)
    precompute = True
    def _map(self, inputs, stream):
        # Find dimensions
        n_types = len(self.args["types"])
        nmax = self.args["nmax"]
        lmax = self.args["lmax"]
        dim_orig = nmax*(lmax+1)**2*n_types
        dim_conv = nmax**2*(lmax+1)*n_types*(n_types+1)//2
        # Convolve
        Q_list = inputs["Q"]
        Q_red = list(map(
            lambda Q: np.sum(Q, axis=0).reshape((1,-1)), 
            Q_list))
        Q_red = np.concatenate(Q_red, axis=0)
        assert Q_red.shape[1] == dim_orig # Unexpected input shape in GylmConvolve
        n_src = Q_red.shape[0]
        X = np.zeros((n_src, dim_conv), dtype=Q_red.dtype)
        gylm.evaluate_power(
            X, Q_red, n_src, n_types, nmax, lmax)
        # Normalize
        if self.args["normalize"]:
            z = 1./(np.sum(X**2, axis=1)+self.args["epsilon"])**0.5
            X = (X.T*z).T
        # Store
        stream.put("X", X)

