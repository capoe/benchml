from benchml.logger import Mock
from benchml.pipeline import Transform

try:
    import asaplib
    import asaplib.data
except ImportError:
    asaplib = Mock()
    asaplib.data = None


def check_asap_available(obj, require=False):
    if asaplib.data is None:
        if require:
            raise ImportError("'%s' requires asap" % obj.__class__.__name__)
        return False
    return True


class AsapTransform(Transform):
    def check_available(self, *args, **kwargs):
        return check_asap_available(self, *args, **kwargs)


class SparseKRR(AsapTransform):
    # TODO
    pass


class SparseSVM(AsapTransform):
    # TODO
    pass


class KernelDensity(AsapTransform):
    # TODO
    pass


class SparseKPCA(AsapTransform):
    # TODO
    pass


class DimReduce(AsapTransform):
    # TODO
    pass


class D2K(AsapTransform):
    # TODO
    pass
