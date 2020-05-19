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
    default_args = {
        "device": "",
        "reset_parameters": False,
        "reset_optimizer": False
    }
    def is_available():
        return check_torch_available()
    def __init__(self, **kwargs):
        Transform.__init__(self, **kwargs)
        nn.Module.__init__(self)
        self.device = None
    def setDevice(self, devstr):
        if devstr.startswith('@'):
            raise ValueError("setDevice: Device address must be resolved first")
        log << "[Set device[%s] = '%s']" % (self.tag, devstr) << log.flush
        self.args["device"] = devstr
        self.device = torch.device(devstr) \
            if devstr != "" else None
        self.to(device=self.device)
        return self.device
    def _parametrize(self, *args):
        return
    def _optimizer(self, *args, **kwargs):
        return
    def _setup(self, *args, **kwargs):
        self.setDevice(self.args["device"])
        if not self._is_setup or self.args["reset_parameters"]:
            self._parametrize(*args, **kwargs)
        self.to(device=self.device)
        if not self._is_setup or self.args["reset_optimizer"]:
            self._optimizer(*args, **kwargs)

class TorchDevice(TorchModuleTransform):
    allow_stream = ("device",)
    def _feed(self, *args, **kwargs):
        self.stream().put("device", self.device)
        return # <- This automatically calls _setup
