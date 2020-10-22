from ..logger import log, Mock
from ..pipeline import Transform, sopen
from ..readwrite import load
from .plugin_check import *
import numpy as np

class TorchModuleTransform(Transform, nn.Module):
    default_args = {
        "device": "",
        "reset_parameters": False,
        "reset_optimizer": False
    }
    def check_available():
        return check_torch_available(__class__)
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
    def freeze(self, freeze=True):
        Transform.freeze(self, freeze=freeze)
        for par in self.parameters():
            par.requires_grad = False if freeze else True
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

class TorchTransformDescriptor(Transform):
    allow_stream = ("X",)
    default_args = {
        "path": None,
        "feed": None,
        "descriptor": "X",
        "field": "X",
        "device": "cpu",
        "to_numpy": True,
        "pop_inputs": []
    }
    req_args = {"path", "feed"}
    req_inputs = {"configs",}
    precompute = True
    stream_samples = ("X",)
    def _setup(self, *args, **kwargs):
        self.model = load(
            self.args["path"], 
            method=torch, 
            map_location=torch.device(self.args["device"]))
        for p in self.args["pop_inputs"]:
            self.model[self.args["descriptor"]].inputs.pop(p, None)
        self.model[self.args["descriptor"]].clearDependencies()
        self.model[self.args["descriptor"]].updateDependencies()
    def _map(self, inputs, stream):
        feed_fct = eval(self.args["feed"])
        feed = feed_fct(inputs)
        stream = self.model.open(feed)
        self.model.map(stream, endpoint=[self.args["descriptor"]], verbose=True)
        X = stream.resolve("%s.%s" % (self.args["descriptor"], self.args["field"]))
        if self.args["to_numpy"]:
            X = np.array([ X[i].detach().cpu().numpy() for i in range(len(X)) ])
        stream.put("X", X)

class TorchDevice(TorchModuleTransform):
    allow_stream = ("device",)
    def _feed(self, data, stream, *args, **kwargs):
        stream.put("device", self.device)
        return # <- This automatically calls _setup

