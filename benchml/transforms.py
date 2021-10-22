import inspect

import numpy as np

from .logger import log
from .pipeline import Module, Transform


class ExttInput(Transform):
    allow_stream = {"X", "Y", "meta"}
    stream_copy = {
        "meta",
    }
    stream_samples = {"X", "Y"}

    def _feed(self, data, stream):
        for key, v in data.arrays.items():
            stream.put(key, v, force=True)
        stream.put("meta", data.meta)


class ExtXyzInput(Transform):
    allow_stream = {"configs", "y", "meta"}
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
    allow_stream = {"X", "y"}
    stream_samples = {"X", "y"}

    def _feed(self, data, stream):
        stream.put("X", data.X)
        stream.put("y", data.y)


class Add(Transform):
    req_args = {
        "coeffs",
    }
    req_inputs = {
        "X",
    }
    allow_stream = {
        "y",
    }

    def _map(self, inputs, stream):
        coeffs = self.args["coeffs"]
        assert len(coeffs) == len(inputs["X"])
        y = np.zeros_like(inputs["X"][0])
        for i in range(len(inputs["X"])):
            y = y + coeffs[i] * inputs["X"][i]
        stream.put("y", y)


class Mult(Transform):
    req_inputs = ("X",)
    allow_stream = {"y"}

    def _map(self, inputs, stream):
        y = np.ones_like(inputs["X"][0])
        for i in range(len(inputs["X"])):
            y = y * inputs["X"][i]
        stream.put("y", y)


class Exp(Transform):
    req_inputs = {
        "X",
    }
    default_args = {"coeff": +1}
    allow_stream = {
        "X",
    }

    def _map(self, inputs, stream):
        stream.put("X", np.exp(self.args["coeff"] * inputs["X"]))


class Delta(Transform):
    allow_stream = {"y"}
    req_inputs = {"target", "ref"}
    stream_samples = ("y",)

    def _map(self, inputs, stream):
        stream.put("y", None)

    def _fit(self, inputs, stream, params):
        delta = inputs["target"] - inputs["ref"]
        stream.put("y", delta)


class RankNorm(Transform):
    req_inputs = {
        "z",
    }
    allow_params = {
        "z",
    }
    allow_stream = {
        "z",
    }

    def _fit(self, inputs, stream, params):
        z = inputs["z"]
        z_ranked = np.sort(z)
        params.put("z", z_ranked)
        self._map(inputs, stream)

    def _map(self, inputs, stream):
        ranked = np.searchsorted(self.params().get("z"), inputs["z"]) / len(self.params().get("z"))
        stream.put("z", ranked)


class SliceMatrix(Transform):
    allow_params = {
        "slice",
    }
    allow_stream = {
        "X",
    }
    default_args = {"axis": None}
    req_inputs = {"slice", "X"}

    def _fit(self, inputs, stream, params):
        if self.args["axis"] is None:
            slice = inputs["slice"]
        elif isinstance(self.args["axis"], int):
            slice = [slice(None) for r in len(inputs["X"].shape)]
            slice[self.args["axis"]] = inputs["slice"]
        else:
            raise ValueError("SliceMatrix arg 'slice' expects None or int")
        params.put("slice", slice)
        return self._map(inputs, stream)

    def _map(self, inputs, stream):
        s = self.params().get("slice")
        stream.put("X", inputs["X"][s])


class DoDivideBySize(Transform):
    default_args = {
        "config_to_size": "lambda c: len(c)",
        "skip_if_not_force": False,
        "force": False,
    }
    req_inputs = ("y", "configs", "meta")
    allow_stream = ("y", "sizes")
    allow_params = ("divide_by_size",)

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

    def _fit(self, inputs, stream, params):
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
            assert np.min(sizes) > 0  # DoDivideBySize: sample size <= 0 not allowed
            y_out = y_in / sizes
        params.put("divide_by_size", do_div)
        stream.put("y", y_out)
        stream.put("sizes", sizes)

    def _map(self, inputs, stream):
        do_div = self.params().get("divide_by_size")
        configs = inputs["configs"]
        if not do_div:
            sizes = np.ones((len(configs),))
        else:
            if type(self.args["config_to_size"]) is str:
                s_fct = eval(self.args["config_to_size"])
            else:
                s_fct = self.args["config_to_size"]
            sizes = np.array(list(map(s_fct, configs)))
            assert np.min(sizes) > 0  # DoDivideBySize: sample size <= 0 not allowed
        stream.put("y", None)
        stream.put("sizes", sizes)


class UndoDivideBySize(Transform):
    req_inputs = (
        "y",
        "sizes",
    )
    allow_stream = ("y",)

    def _map(self, inputs, stream):
        y_in = inputs["y"]
        sizes = inputs["sizes"]
        assert y_in.shape[0] == sizes.shape[0]  # UndoDivideBySize: inconsistent input dim
        y_out = y_in * sizes
        stream.put("y", y_out)


from .conformal import *  # noqa: F401
from .descriptors import *  # noqa: F401
from .ensemble import *  # noqa: F401
from .filters import *  # noqa: F401
from .kernels import *  # noqa: F401
from .matrix import *  # noqa: F401
from .plugins import *  # noqa: F401
from .predictors import *  # noqa: F401


def transform_info(tf, log, verbose=True):
    if verbose:
        log << log.mg << "%-25s from %s" % ("<%s>" % tf.__name__, inspect.getfile(tf)) << log.endl
        log << "  Required args:       "
        for arg in tf.req_args:
            log << "'%s'" % arg
        log << log.endl
        log << "  Required inputs:     "
        for arg in tf.req_inputs:
            log << "'%s'" % arg
        log << log.endl
        log << "  Allow stream:        "
        for item in tf.allow_stream:
            log << "'%s'" % item
        log << log.endl
        log << "    - Stream samples:  "
        for item in tf.stream_samples:
            log << "'%s'" % item
        log << log.endl
        log << "    - Stream copy:     "
        for item in tf.stream_copy:
            log << "'%s'" % item
        log << log.endl
        log << "    - Stream kernel:   "
        for item in tf.stream_kernel:
            log << "'%s'" % item
        log << log.endl
        log << "  Allow params:        "
        for item in tf.allow_params:
            log << "'%s'" % item
        log << log.endl
        log << "  Precompute:           " << bool(tf.precompute) << log.endl
    else:
        log << "%-35s" % ("<" + tf.__name__ + ">")
        argstr = ",".join(tf.req_args)
        inputstr = ",".join(tf.req_inputs)
        streamstr = ",".join(tf.allow_stream)
        available = tf.check_available()
        log << "requires:   args=%-15s inputs=%-30s   outputs:  %-25s   %s" % (
            argstr,
            inputstr,
            streamstr,
            "[installed]" if available else "[missing]",
        )
        log << log.endl


def get_bases_recursive(obj):
    bases = list(obj.__bases__)
    sub = []
    for b in bases:
        sub = sub + get_bases_recursive(b)
    bases = bases + sub
    return bases


def get_all():
    import sys

    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj):
            if Transform in get_bases_recursive(obj):
                if obj is Module:
                    continue
                yield obj


def list_all(verbose=False):
    import sys

    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj):
            if Transform in get_bases_recursive(obj):
                if obj is Module:
                    continue
                transform_info(obj, log=log, verbose=verbose)
