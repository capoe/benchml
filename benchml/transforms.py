from .pipeline import Module, Macro, Transform
from .hyper import Hyper, GridHyper, BayesianHyper
from .logger import log
import numpy as np
import sklearn.linear_model
import sklearn.kernel_ridge

class ExtXyzInput(Transform):
    allow_stream = {'configs', 'y', 'meta'}
    stream_copy = ("meta",)
    stream_samples = ("configs", "y")
    def _feed(self, data):
        self.stream().put("configs", data)
        self.stream().put("y", data.y)
        self.stream().put("meta", data.meta)

class Add(Transform):
    req_args = ('coeffs',)
    req_inputs = ('X',)
    allow_stream = {'y'}
    def __init__(self, **kwargs):
        Transform.__init__(self, **kwargs)
    def _map(self, inputs):
        coeffs = self.args["coeffs"]
        assert len(coeffs) == len(inputs["X"])
        y = np.zeros_like(inputs["X"][0])
        for i in range(len(inputs["X"])):
            y = y + coeffs[i]*inputs["X"][i]
        self.stream().put("y", y)

class ReduceMatrix(Transform):
    req_inputs = ("X",)
    default_args = {
        "reduce": "np.sum(x, axis=0)",
        "norm": True,
        "epsilon": 1e-10 }
    allow_stream = ("X",)
    stream_samples = ("X",)
    def _map(self, inputs):
        X = map(lambda x: eval(self.args["reduce"]), inputs["X"])
        X = map(lambda x: x/(np.dot(x,x)**0.5+self.args["epsilon"]), X)
        X = map(lambda x: x.reshape((1,-1)), X)
        X = np.concatenate(list(X), axis=0)
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

class WhitenMatrix(Transform):
    default_args = {
        "centre": True,
        "scale": True,
        "epsilon": 1e-10 }
    req_inputs = ("X",)
    allow_params = ("x_avg", "x_std")
    allow_stream = ("X",)
    def _fit(self, inputs):
        x_avg = np.mean(inputs["X"], axis=0)
        x_std = np.std(inputs["X"], axis=0) + self.args["epsilon"]
        self.params().put("x_avg", x_avg)
        self.params().put("x_std", x_std)
        self._map(inputs)
    def _map(self, inputs):
        if self.args["centre"]:
            X_w = inputs["X"]-self.params().get("x_avg")
        else:
            X_w = inputs["X"]
        if self.args["scale"]:
            X_w = X_w/self.params().get("x_std")
        self.stream().put("X", X_w)

from .plugins import *
from .descriptors import *
from .kernels import *
from .predictors import *

def transform_info(tf, log, verbose=True):
    if verbose:
        log << log.mg << "<%s>" % tf.__name__ << log.endl
        log << "  Required args:       "
        for arg in tf.req_args: log << "'%s'" % arg
        log << log.endl
        log << "  Required inputs:     "
        for arg in tf.req_inputs: log << "'%s'" % arg
        log << log.endl
        log << "  Allow stream:        "
        for item in tf.allow_stream: log << "'%s'" % item
        log << log.endl
        log << "    - Stream samples:  "
        for item in tf.stream_samples: log << "'%s'" % item
        log << log.endl
        log << "    - Stream copy:     "
        for item in tf.stream_copy: log << "'%s'" % item
        log << log.endl
        log << "    - Stream kernel:   "
        for item in tf.stream_kernel: log << "'%s'" % item
        log << log.endl
        log << "  Allow params:        "
        for item in tf.allow_params: log << "'%s'" % item
        log << log.endl
        log << "  Precompute:           " << bool(tf.precompute) << log.endl
    else:
        log << "%-22s" % ("<"+tf.__name__+">")
        argstr = ",".join(tf.req_args)
        inputstr = ",".join(tf.req_inputs)
        streamstr = ",".join(tf.allow_stream)
        available = tf.check_available()
        log << "requires:   args=%-15s inputs=%-15s   outputs:  %-15s   available: %s" % (
            argstr, inputstr, streamstr, available)
        log << log.endl

def get_bases_recursive(obj):
    bases = list(obj.__bases__)
    sub = []
    for b in bases:
        sub = sub + get_bases_recursive(b)
    bases = bases + sub
    return bases

def list_all(verbose=False):
    import sys
    import inspect
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj):
            if Transform in get_bases_recursive(obj):
                transform_info(obj, log=log, verbose=verbose)
