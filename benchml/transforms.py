from .pipeline import Module, Macro, Transform, Hyper
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

class DescriptorRandom(Transform):
    req_args = ('dim',)
    req_inputs = ('configs',)
    allow_stream = {'X'}
    stream_samples = ("X",)
    precompute = True
    def _map(self, inputs):
        X = np.random.uniform(0., 1., size=(len(inputs["configs"]), self.args["dim"]))
        self.stream().put("X", X)

from .kernels import *
from .plugins import *

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
        log << "%-20s" % ("<"+tf.__name__+">")
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
