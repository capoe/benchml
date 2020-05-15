from ..kernels import KernelDot
from ..pipeline import Transform, Macro
from ..logger import log
import numpy as np
import multiprocessing as mp
import time
try:
    import gylm
except ImportError:
    gylm = None

def check_gylmxx_available(obj, require=False):
    if gylm is None:
        if require: raise ImportError("%s requires gylmxx" % obj.__name__)
        return False
    return True

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
        if self.args["types"] is None:
            self.calc = None
        else:
            self.calc = gylm.GylmCalculator(**self.args)
    def _fit(self, inputs):
        if self.args["types"] is None:
            self.args["types"] = inputs["meta"]["elements"]
        self.calc = gylm.GylmCalculator(
            **self.args)
        self.params().put("calc", self.calc)
        self._map(inputs)

class GylmAverage(GylmTransform):
    def _map(self, inputs):
        X = gylm_evaluate(
            configs=inputs["configs"],
            dcalc=self.calc,
            reduce_molecular=np.sum,
            norm_molecular=True,
            centres=inputs.pop("centres", None))
        self.stream().put("X", X)

class GylmAtomic(GylmTransform):
    def _map(self, inputs):
        if self.procs == 1:
            X = gylm_evaluate(
                configs=inputs["configs"],
                dcalc=self.calc,
                reduce_molecular=None,
                norm_molecular=False,
                centres=inputs.pop("centres", None))
        else:
            X = gylm_evaluate_mp(
                configs=inputs["configs"],
                dcalc=self.calc,
                procs=self.procs,
                reduce_molecular=None,
                norm_molecular=False,
                centres=inputs.pop("centres", None))
        self.stream().put("X", X)

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
        centres=None):
    log = GylmTransform.log
    t0 = time.time()
    X = []
    for cidx, config in enumerate(configs):
        if log and log.verbose: log << log.back << \
            "%d/%d" % (cidx+1, len(configs)) << log.flush
        if centres is None:
            heavy, types_centres, pos_centres = config.getHeavy()
        else:
            pos_centres = centres[cidx]
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
