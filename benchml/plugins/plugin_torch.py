from ..logger import log, Mock
from ..pipeline import Transform
try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = Mock()
    nn.Module = None

def check_torch_available():
    return torch is not None

class TorchModuleTransform(Transform, nn.Module):
    def is_available():
        return check_torch_available() 
    def __init__(self, **kwargs):
        Transform.__init__(self, **kwargs)
        nn.Module.__init__(self)
    def setDevice(self, devstr):
        if devstr.startswith('@'):
            tf, val = devstr[1:].split(".")
            devstr = self.module[tf] = val
        log << "[Set device[%s] = '%s']" % (self.tag, devstr) << log.flush
        self.device = torch.device(devstr) \
            if devstr != "" else None
        self.to(device=self.device)
        return self.device
    def _parametrize(self, *args):
        raise NotImplementedError("_parametrize overload missing")
    def _optimizer(self, *args, **kwargs):
        raise NotImplementedError("_optimizer overload missing")
    def _setup(self, *args, **kwargs):
        self.setDevice(self.args["device"])
        self._parametrize(*args, **kwargs)
        self.to(device=self.device)
        self._optimizer(*args, **kwargs)
