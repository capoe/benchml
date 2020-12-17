from .pipeline import Module, Macro, Transform
from .hyper import Hyper, GridHyper, BayesianHyper
from .logger import log
import numpy as np
import inspect

class ExtXyzInput(Transform):
    allow_stream = {'configs', 'y', 'meta'}
    stream_copy = ("meta",)
    stream_samples = ("configs", "y")
    def _feed(self, data, stream):
        stream.put("configs", data)
        if hasattr(data, "y"):
            stream.put("y", data.y)
        else:
            stream.put("y", [])
        if hasattr(data, "meta"):
            stream.put("meta", data.meta)
        else:
            stream.put("meta", {})

class XyInput(Transform):
    allow_stream = {'X', 'y'}
    stream_samples = {'X', 'y'}
    def _feed(self, data, stream):
        stream.put("X", data.X)
        stream.put("y", data.y)

class Add(Transform):
    req_args = ('coeffs',)
    req_inputs = ('X',)
    allow_stream = {'y'}
    def _map(self, inputs, stream):
        coeffs = self.args["coeffs"]
        assert len(coeffs) == len(inputs["X"])
        y = np.zeros_like(inputs["X"][0])
        for i in range(len(inputs["X"])):
            y = y + coeffs[i]*inputs["X"][i]
        stream.put("y", y)

class Mult(Transform):
    req_inputs = ('X',)
    allow_stream = {'y'}
    def _map(self, inputs, stream):
        y = np.ones_like(inputs["X"][0])
        for i in range(len(inputs["X"])):
            y = y*inputs["X"][i]
        stream.put("y", y)

class Delta(Transform):
    allow_stream = {'y'}
    req_inputs = {'target', 'ref'}
    stream_samples = ("y",)
    def _map(self, inputs, stream):
        stream.put("y", None)
    def _fit(self, inputs, stream, params):
        delta = inputs["target"] - inputs["ref"]
        stream.put("y", delta)

class Concatenate(Transform):
    req_inputs = ('X',)
    allow_stream = {'X',}
    stream_samples = ('X',)
    precompute = True
    default_args = {
        "axis": 1
    }
    def _map(self, inputs, stream):
        X_out = np.concatenate(inputs["X"], axis=self.args["axis"])
        stream.put("X", X_out)

class ReduceMatrix(Transform):
    req_inputs = ("X",)
    default_args = {
        "reduce": "np.sum(x, axis=0)",
        "norm": True,
        "epsilon": 1e-10 }
    allow_stream = ("X",)
    stream_samples = ("X",)
    def _map(self, inputs, stream):
        X = map(lambda x: eval(self.args["reduce"]), inputs["X"])
        X = map(lambda x: x/(np.dot(x,x)**0.5+self.args["epsilon"]), X)
        X = map(lambda x: x.reshape((1,-1)), X)
        X = np.concatenate(list(X), axis=0)
        stream.put("X", X)

class ReduceTypedMatrix(Transform):
    default_args = {
        "reduce_op": "sum",
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
    def _fit(self, inputs, stream, params):
        if self.args["reduce_by_type"]:
            if self.args["types"] is not None:
                self.types = self.args["types"]
            else:
                self.types = inputs["meta"]["elements"]
            self.type_to_idx = { t: tidx for tidx, t in \
                enumerate(self.types) }
            params.put("types", self.types)
        self._map(inputs, stream)
    def _map(self, inputs, stream):
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
        stream.put("X", X_red)

class WhitenMatrix(Transform):
    default_args = {
        "centre": True,
        "scale": True,
        "epsilon": 1e-10 }
    req_inputs = ("X",)
    allow_params = ("x_avg", "x_std")
    allow_stream = ("X",)
    def _fit(self, inputs, stream, params):
        x_avg = np.mean(inputs["X"], axis=0)
        x_std = np.std(inputs["X"], axis=0) + self.args["epsilon"]
        params.put("x_avg", x_avg)
        params.put("x_std", x_std)
        self._map(inputs, stream)
    def _map(self, inputs, stream):
        if self.args["centre"]:
            X_w = inputs["X"]-self.params().get("x_avg")
        else:
            X_w = inputs["X"]
        if self.args["scale"]:
            X_w = X_w/self.params().get("x_std")
        stream.put("X", X_w)

class SubsampleMatrix(Transform):
    default_args = {
        "info_key": None,
    }
    allow_stream = {"X",}
    def _map(self, inputs, stream):
        raise NotImplementedError() # TODO Generalize
        X = inputs["X"]
        configs = inputs["configs"]
        X_out = []
        for i in range(len(configs)):
            assert len(X[i]) == len(configs[i].symbols)
            sel = int(configs[i].info[self.args["info_key"]])
            X_out.append(X[i][sel])
        X_out = np.array(X_out)
        stream.put("X", X_out)

class RankNorm(Transform):
    allow_params = {"z",}
    allow_stream = {"z",}
    def _fit(self, inputs, stream, params):
        z = inputs["z"]
        z_ranked = np.sort(z)
        params.put("z", z_ranked)
        self._map(inputs, stream)
    def _map(self, inputs, stream):
        ranked = np.searchsorted(
            self.params().get("z"), 
            inputs["z"])/len(self.params().get("z"))
        stream.put("z", ranked)

class DoDivideBySize(Transform):
    default_args = {
        "config_to_size": "lambda c: len(c)",
        "skip_if_not_force": False,
        "force": False
    }
    req_inputs = ("y", "configs", "meta")
    allow_stream = ("y", "sizes")
    def checkDoDivide(self, inputs):
        do_divide_by_size = False
        if self.args["force"]:
            do_divide_by_size = True
        elif self.args["skip_if_not_force"]:
            pass
        elif inputs["meta"]["scaling"] == "additive":
            do_divide_by_size = True
        elif inputs["meta"]["scaling"] == "unknown":
            pass
        elif inputs["meta"]["scaling"] == "non-additive":
            pass
        else:
            raise ValueError("Scaling should be one of additive|non-additive|unknown")
        return do_divide_by_size
    def _map(self, inputs, stream):
        do_div = self.checkDoDivide(inputs)
        y_in = inputs["y"]
        if not do_div:
            sizes = np.ones_like(inputs["y"])
            y_out = np.copy(y_in)
        else:
            if type(self.args["config_to_size"]) is str:
                s_fct = eval(self.args["config_to_size"])
            else:
                s_fct = self.args["config_to_size"]
            configs = inputs["configs"]
            sizes = np.array(list(map(s_fct, configs)))
            assert np.min(sizes) > 0 # DoDivideBySize: sample size <= 0 not allowed
            y_out = y_in/sizes
        stream.put("y", y_out)
        stream.put("sizes", sizes)

class UndoDivideBySize(Transform):
    req_inputs = ("y", "sizes",)
    allow_stream = ("y",)
    def _map(self, inputs, stream):
        y_in = inputs["y"]
        sizes = inputs["sizes"]
        assert y_in.shape[0] == sizes.shape[0] # UndoDivideBySize: inconsistent input dim
        y_out = y_in*sizes
        stream.put("y", y_out)

from .plugins import *
from .descriptors import *
from .kernels import *
from .predictors import *

def transform_info(tf, log, verbose=True):
    if verbose:
        log << log.mg << "%-25s from %s" % (
            "<%s>" % tf.__name__, inspect.getfile(tf))<< log.endl
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
        log << "%-25s" % ("<"+tf.__name__+">")
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
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj):
            if Transform in get_bases_recursive(obj):
                if obj is Module: continue
                transform_info(obj, log=log, verbose=verbose)
