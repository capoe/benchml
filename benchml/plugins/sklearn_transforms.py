from ..pipeline import Transform, Macro
from ..logger import log
import numpy as np
try:
    import sklearn
    import sklearn.linear_model
    import sklearn.kernel_ridge
except ImportError:
    sklearn = None

def check_sklearn_available(obj, require=False):
    if sklearn is None:
        if require: raise ImportError("%s requires sklearn" % obj.__name__)
        return False
    return True

class SklearnTransform(Transform):
    def check_available():
        return check_sklearn_available(SklearnTransform)

class Ridge(SklearnTransform):
    default_args = { 'alpha': 1. }
    req_args = tuple()
    req_inputs = ('X', 'y')
    allow_params = {'model'}
    allow_stream = {'y'}
    def __init__(self, **kwargs):
        Transform.__init__(self, **kwargs)
    def _fit(self, inputs):
        model = sklearn.linear_model.Ridge(**self.args)
        model.fit(X=inputs["X"], y=inputs["y"])
        yp = model.predict(inputs["X"])
        self.params().put("model", model)
        self.stream().put("y", yp)
    def _map(self, inputs):
        y = self.params().get("model").predict(inputs["X"])
        self.stream().put("y", y)

class KernelRidge(SklearnTransform):
    req_args = ('alpha',)
    default_args = {'power': 1}
    req_inputs = ('K', 'y')
    allow_params = {'model', 'y_mean', 'y_std', 'y'}
    allow_stream = {'y'}
    def __init__(self, **kwargs):
        Transform.__init__(self, **kwargs)
    def readArgs(self):
        self.power = self.args["power"]
    def _fit(self, inputs):
        y_mean = np.mean(inputs["y"])
        y_std = np.std(inputs["y"])
        y_train = (inputs["y"]-y_mean)/y_std
        model = sklearn.kernel_ridge.KernelRidge(
            kernel='precomputed', alpha=self.args["alpha"])
        model.fit(inputs["K"]**self.power, y_train)
        y_pred = model.predict(inputs["K"]**self.power)*y_std + y_mean
        self.params().put("model", model)
        self.params().put("y_mean", y_mean)
        self.params().put("y_std", y_std)
        self.stream().put("y", y_pred)
    def _map(self, inputs):
        y = self.params().get("model").predict(inputs["K"]**self.power)
        y = y*self.params().get("y_std") + self.params().get("y_mean")
        self.stream().put("y", y)

