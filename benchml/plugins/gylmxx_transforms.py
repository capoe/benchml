from ..kernels import KernelDot
from ..pipeline import Transform, Macro
from ..logger import log
import numpy as np
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
    def _fit(self, inputs):
        if self.args["types"] is None:
            self.args["types"] = inputs["meta"]["elements"]
        calc = gylm.GylmCalculator(
            **self.args)
        self.params().put("calc", calc)
        self._map(inputs)

class GylmAverage(GylmTransform):
    def _map(self, inputs):
        X = gylm_evaluate(
            configs=inputs["configs"],
            dcalc=self.params().get("calc"),
            reduce_molecular=np.sum,
            norm_molecular=True)
        self.stream().put("X", X)

class GylmAtomic(GylmTransform):
    def _map(self, inputs):
        X = gylm_evaluate(
            configs=inputs["configs"],
            dcalc=self.params().get("calc"),
            reduce_molecular=None,
            norm_molecular=False)
        self.stream().put("X", X)

def gylm_evaluate(
        configs, 
        dcalc, 
        reduce_molecular=None, 
        norm_molecular=False, 
        get_centres=None):
    log = GylmTransform.log
    if get_centres is None:
        get_centres = lambda config: np.where(np.array(config.symbols) != 'H')[0]
    t0 = time.time()
    X = []
    for cidx, config in enumerate(configs):
        if log: log << log.back << "%d/%d" % (cidx+1, len(configs)) << log.flush
        heavy = get_centres(config)
        x = dcalc.evaluate(system=config,
            positions=config.positions[heavy])
        if reduce_molecular is not None:
            x = reduce_molecular(x, axis=0)
        if norm_molecular:
            x = x/np.dot(x,x)**0.5
        X.append(x)
    t1 = time.time()
    if log: 
        log << log.endl
        log << "Finished in %fs" % (t1-t0) << log.endl
    X = np.array(X)
    return X

