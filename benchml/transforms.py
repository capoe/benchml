import inspect

from benchml.logger import log
from benchml.pipeline import Module, Transform

from .basic import *  # noqa: F401, F403
from .conformal import *  # noqa: F401, F403
from .descriptors import *  # noqa: F401, F403
from .ensemble import *  # noqa: F401, F403
from .filters import *  # noqa: F401, F403
from .inputs import *  # noqa: F401, F403
from .kernels import *  # noqa: F401, F403
from .matrix import *  # noqa: F401, F403
from .plugins import *  # noqa: F401, F403
from .predictors import *  # noqa: F401, F403


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
