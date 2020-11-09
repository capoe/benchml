from ..pipeline import Transform, Macro
from ..logger import log
import numpy as np
try:
    import sklearn
    import sklearn.linear_model
    import sklearn.kernel_ridge
    import sklearn.ensemble
    import sklearn.gaussian_process
    import sklearn.svm
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

class LinearRegression(SklearnTransform):
    req_inputs = ('X', 'y')
    allow_params = {'model'}
    allow_stream = {'y'}
    def _fit(self, inputs, stream, params):
        model = sklearn.linear_model.LinearRegression(**self.args)
        model.fit(X=inputs["X"], y=inputs["y"])
        yp = model.predict(inputs["X"])
        params.put("model", model)
        stream.put("y", yp)
    def _map(self, inputs, stream):
        y = self.params().get("model").predict(inputs["X"])
        stream.put("y", y)

class Ridge(SklearnTransform):
    default_args = { 'alpha': 1. }
    req_inputs = ('X', 'y')
    allow_params = {'model', 'y_mean', 'y_std'}
    allow_stream = {'y'}
    def _fit(self, inputs, stream, params):
        y_mean = np.mean(inputs["y"])
        y_std = np.std(inputs["y"])
        y_train = (inputs["y"]-y_mean)/y_std
        model = sklearn.linear_model.Ridge(**self.args)
        model.fit(X=inputs["X"], y=y_train)
        yp = model.predict(inputs["X"])*y_std + y_mean
        params.put("model", model)
        params.put("y_mean", y_mean)
        params.put("y_std", y_std)
        stream.put("y", yp)
    def _map(self, inputs, stream):
        y = self.params().get("model").predict(inputs["X"])
        y = self.params().get("y_std")*y + self.params().get("y_mean")
        stream.put("y", y)

class GradientBoosting(SklearnTransform):
    allow_stream = {"y"}
    allow_params = {"model"}
    req_inputs = {"X", "y"}
    def _fit(self, inputs, stream, params):
        model = sklearn.ensemble.GradientBoostingRegressor()
        model.fit(inputs["X"], inputs["y"])
        y_pred = model.predict(inputs["X"])
        params.put("model", model)
        stream.put("y", y_pred)
    def _map(self, inputs, stream):
        y_pred = self.params().get("model").predict(inputs["X"])
        stream.put("y", y_pred)

class RandomForestRegressor(SklearnTransform):
    default_args = dict(
        n_estimators=100, 
        criterion='mse', 
        max_depth=None, 
        min_samples_split=2, 
        min_samples_leaf=1, 
        min_weight_fraction_leaf=0.0, 
        max_features='auto', 
        max_leaf_nodes=None, 
        min_impurity_decrease=0.0, 
        min_impurity_split=None, 
        bootstrap=True, 
        oob_score=False, 
        n_jobs=None, 
        random_state=None, 
        verbose=0, 
        warm_start=False, 
        ccp_alpha=0.0, 
        max_samples=None)
    allow_stream = {"y"}
    allow_params = {"model"}
    req_inputs = {"X", "y"}
    def _fit(self, inputs, stream, params):
        model = sklearn.ensemble.RandomForestRegressor(**self.args)
        model.fit(inputs["X"], inputs["y"])
        y_pred = model.predict(inputs["X"])
        params.put("model", model)
        stream.put("y", y_pred)
    def _map(self, inputs, stream):
        y_pred = self.params().get("model").predict(inputs["X"])
        stream.put("y", y_pred)

class RandomForestClassifier(SklearnTransform):
    default_args = dict(
        n_estimators=100, 
        criterion='gini', 
        max_depth=None, 
        min_samples_split=2, 
        min_samples_leaf=1, 
        min_weight_fraction_leaf=0.0, 
        max_features='auto', 
        max_leaf_nodes=None, 
        min_impurity_decrease=0.0, 
        min_impurity_split=None, 
        bootstrap=True, 
        oob_score=False, 
        n_jobs=None, 
        random_state=None, 
        verbose=0, 
        warm_start=False, 
        class_weight=None,
        ccp_alpha=0.0, 
        max_samples=None)
    allow_stream = {"y", "z"}
    allow_params = {"model"}
    req_inputs = {"X", "y"}
    def _fit(self, inputs, stream, params):
        model = sklearn.ensemble.RandomForestClassifier(**self.args)
        model.fit(inputs["X"], inputs["y"])
        y_pred = model.predict(inputs["X"])
        z_pred = model.predict_proba(inputs["X"])[:,0]
        params.put("model", model)
        stream.put("y", y_pred)
        stream.put("z", z_pred)
    def _map(self, inputs, stream):
        y_pred = self.params().get("model").predict(inputs["X"])
        z_pred = self.params().get("model").predict_proba(inputs["X"])[:,0]
        stream.put("y", y_pred)
        stream.put("z", z_pred)

class KernelRidge(SklearnTransform):
    req_args = ('alpha',)
    default_args = {'power': 1}
    req_inputs = ('K', 'y')
    allow_params = {'model', 'y_mean', 'y_std', 'y'}
    allow_stream = {'y'}
    def __init__(self, **kwargs):
        Transform.__init__(self, **kwargs)
    def _setup(self):
        self.power = self.args["power"]
    def _fit(self, inputs, stream, params):
        y_mean = np.mean(inputs["y"])
        y_std = np.std(inputs["y"])
        y_train = (inputs["y"]-y_mean)/y_std
        model = sklearn.kernel_ridge.KernelRidge(
            kernel='precomputed', alpha=self.args["alpha"])
        model.fit(inputs["K"]**self.power, y_train)
        y_pred = model.predict(inputs["K"]**self.power)*y_std + y_mean
        params.put("model", model)
        params.put("y_mean", y_mean)
        params.put("y_std", y_std)
        stream.put("y", y_pred)
    def _map(self, inputs, stream):
        y = self.params().get("model").predict(inputs["K"]**self.power)
        y = y*self.params().get("y_std") + self.params().get("y_mean")
        stream.put("y", y)

class GaussianProcessRegressor(SklearnTransform):
    req_args = ('alpha',)
    default_args = {'power': 1}
    req_inputs = ('K', 'y')
    allow_params = {'model', 'y_mean', 'y_std', 'y'}
    allow_stream = {'y', 'dy'}
    def _setup(self):
        self.power = self.args["power"]
    def _fit(self, inputs, stream, params):
        y_mean = np.mean(inputs["y"])
        y_std = np.std(inputs["y"])
        y_train = (inputs["y"]-y_mean)/y_std
        model = sklearn.gaussian_process.GaussianProcessRegressor(
            kernel='precomputed', alpha=self.args["alpha"])
        model.fit(inputs["K"]**self.power, y_train)
        y_pred, dy_pred = model.predict(inputs["K"]**self.power, return_std=True)
        dy_pred = dy_pred*y_std 
        y_pred = y_pred*y_std + y_mean
        params.put("model", model)
        params.put("y_mean", y_mean)
        params.put("y_std", y_std)
        stream.put("y", y_pred)
        stream.put("dy", dy)
    def _map(self, inputs, stream):
        y, dy = self.params().get("model").predict(inputs["K"]**self.power, return_std=True)
        y = y*self.params().get("y_std") + self.params().get("y_mean")
        dy = dy*self.params().get("y_std")
        stream.put("y", y)
        stream.put("dy", dy)

class SupportVectorClassifier(SklearnTransform):
    default_args = dict(
        C=1.,
        power=2,
        kernel='precomputed',
        class_weight=None)
    req_inputs = {"K", "y"}
    allow_params = {'model',}
    allow_stream = {"y", "z"}
    def _setup(self):
        self.power = self.args["power"]
    def _fit(self, inputs, stream, params):
        Kp = inputs["K"]**self.power
        model = sklearn.svm.SVC(
            kernel=self.args["kernel"],
            C=self.args["C"], 
            class_weight=self.args["class_weight"])
        model.fit(Kp, inputs["y"])
        y_pred = model.predict(Kp)
        z_pred = model.decision_function(Kp)
        params.put("model", model)
        stream.put("y", y_pred)
        stream.put("z", z_pred)
    def _map(self, inputs, stream):
        Kp = inputs["K"]**self.power
        y = self.params().get("model").predict(Kp)
        z = self.params().get("model").decision_function(Kp)
        stream.put("y", y)
        stream.put("z", z)

