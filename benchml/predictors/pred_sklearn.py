from abc import ABC

import numpy as np

from benchml.pipeline import FitTransform

try:
    import sklearn
    import sklearn.ensemble
    import sklearn.gaussian_process
    import sklearn.kernel_ridge
    import sklearn.linear_model
    import sklearn.svm
except ImportError:
    sklearn = None


def check_sklearn_available(obj, require=False):
    if sklearn is None:
        if require:
            raise ImportError("%s requires sklearn" % obj.__name__)
        return False
    return True


class SklearnTransform(FitTransform, ABC):
    def check_available(self, *args, **kwargs):
        return check_sklearn_available(self, *args, **kwargs)


class LinearRegression(SklearnTransform):
    req_inputs = ("X", "y")
    allow_params = {"model"}
    allow_stream = {"y"}

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
    default_args = {"alpha": 1.0}
    req_inputs = ("X", "y")
    allow_params = {"model", "y_mean", "y_std"}
    allow_stream = {"y"}

    def _fit(self, inputs, stream, params):
        y_mean = np.mean(inputs["y"])
        y_std = np.std(inputs["y"])
        y_train = (inputs["y"] - y_mean) / y_std
        model = sklearn.linear_model.Ridge(**self.args)
        model.fit(X=inputs["X"], y=y_train)
        yp = model.predict(inputs["X"]) * y_std + y_mean
        params.put("model", model)
        params.put("y_mean", y_mean)
        params.put("y_std", y_std)
        stream.put("y", yp)

    def _map(self, inputs, stream):
        y = self.params().get("model").predict(inputs["X"])
        y = self.params().get("y_std") * y + self.params().get("y_mean")
        stream.put("y", y)


class RidgeClassifier(SklearnTransform):
    default_args = {"alpha": 1.0, "class_weight": "balanced"}
    req_inputs = ("X", "y")
    allow_params = {
        "model",
    }
    allow_stream = {"y", "z"}

    def _fit(self, inputs, stream, params):
        y_train = inputs["y"]
        model = sklearn.linear_model.RidgeClassifier(**self.args)
        model.fit(X=inputs["X"], y=y_train)
        y = model.predict(inputs["X"])
        z = model.decision_function(inputs["X"])
        params.put("model", model)
        stream.put("y", y)
        stream.put("z", z)

    def _map(self, inputs, stream):
        model = self.params().get("model")
        y = model.predict(inputs["X"])
        z = model.decision_function(inputs["X"])
        stream.put("y", y)
        stream.put("z", z)


class ElasticNetClassifier(SklearnTransform):
    default_args = dict(alpha=1.0, l1_ratio=0.5, margin=1e-5)
    req_inputs = {"X", "y"}
    allow_params = {"model"}
    allow_stream = {"y", "z"}

    def _setup(self):
        if "margin" in self.args:
            self.margin = self.args.pop("margin")

    def _fit(self, inputs, stream, params):
        y = inputs["y"]
        y_spin = 2 * (y - 0.5)
        i0 = np.where(np.abs(y - 0.0) < 1e-3)[0]
        i1 = np.where(np.abs(y - 1.0) < 1e-3)[0]
        n0 = len(i0)
        n1 = len(i1)
        assert (n0 + n1) == len(y)
        w0 = float(n1) / (n0 + n1)
        w1 = 1.0 - w0
        w = 1.0 * np.ones_like(y)
        w[i0] = w0
        w[i1] = w1
        model = sklearn.linear_model.ElasticNet(**self.args)
        model.fit(X=inputs["X"], y=y_spin, sample_weight=w)
        params.put("model", model)
        self._map(inputs, stream)

    def _map(self, inputs, stream):
        z = self.params().get("model").predict(inputs["X"])
        z[np.where(np.abs(z) < self.margin)] = 0
        y = np.zeros_like(z)
        y[np.where(z > 0.0)] = 1.0
        stream.put("y", y)
        stream.put("z", z)


class OMPClassifier(FitTransform):
    default_args = dict(
        n_nonzero_coefs=5,
    )
    req_inputs = {"X", "y"}
    allow_params = {"model"}
    allow_stream = {"y", "z"}

    def _fit(self, inputs, stream, params):
        y = inputs["y"]
        y_spin = 2 * (y - 0.5)
        model = sklearn.linear_model.OrthogonalMatchingPursuit(**self.args)
        model.fit(X=inputs["X"], y=y_spin)
        params.put("model", model)
        self._map(inputs, stream)

    def _map(self, inputs, stream):
        z = self.params().get("model").predict(inputs["X"])
        y = np.zeros_like(z)
        y[np.where(z > 0.0)] = 1.0
        stream.put("y", y)
        stream.put("z", y)


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
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="auto",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        # min_impurity_split=None,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
    )
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
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="auto",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        # min_impurity_split=None,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
    )
    allow_stream = {"y", "z"}
    allow_params = {"model"}
    req_inputs = {"X", "y"}

    def _fit(self, inputs, stream, params):
        model = sklearn.ensemble.RandomForestClassifier(**self.args)
        model.fit(inputs["X"], inputs["y"])
        y_pred = model.predict(inputs["X"])
        z_pred = model.predict_proba(inputs["X"])[:, 1]
        params.put("model", model)
        stream.put("y", y_pred)
        stream.put("z", z_pred)

    def _map(self, inputs, stream):
        y_pred = self.params().get("model").predict(inputs["X"])
        z_pred = self.params().get("model").predict_proba(inputs["X"])[:, 1]
        stream.put("y", y_pred)
        stream.put("z", z_pred)


class KernelRidge(SklearnTransform):
    req_args = ("alpha",)
    default_args = {"power": 1}
    req_inputs = ("K", "y")
    allow_params = {"model", "w", "y_mean", "y_std", "y"}
    allow_stream = {"y"}

    def _setup(self):
        self.power = self.args["power"]

    def _fit(self, inputs, stream, params):
        y_mean = np.mean(inputs["y"])
        y_std = np.std(inputs["y"])
        y_train = (inputs["y"] - y_mean) / y_std
        model = sklearn.kernel_ridge.KernelRidge(kernel="precomputed", alpha=self.args["alpha"])
        model.fit(inputs["K"] ** self.args["power"], y_train)
        y_pred = model.predict(inputs["K"] ** self.args["power"]) * y_std + y_mean
        params.put("model", model)
        params.put("y_mean", y_mean)
        params.put("y_std", y_std)
        params.put("w", model.dual_coef_)
        stream.put("y", y_pred)

    def _map(self, inputs, stream):
        y = self.params().get("model").predict(inputs["K"] ** self.args["power"])
        y = y * self.params().get("y_std") + self.params().get("y_mean")
        stream.put("y", y)


class GaussianProcessRegressor(SklearnTransform):
    req_args = ("alpha",)
    default_args = {"power": 1}
    req_inputs = ("K", "y")
    allow_params = {"model", "y_mean", "y_std", "y"}
    allow_stream = {"y", "dy"}

    def _setup(self):
        self.power = self.args["power"]

    def _fit(self, inputs, stream, params):
        y_mean = np.mean(inputs["y"])
        y_std = np.std(inputs["y"])
        y_train = (inputs["y"] - y_mean) / y_std
        model = sklearn.gaussian_process.GaussianProcessRegressor(
            kernel="precomputed", alpha=self.args["alpha"]
        )
        model.fit(inputs["K"] ** self.power, y_train)
        y_pred, dy_pred = model.predict(inputs["K"] ** self.power, return_std=True)
        dy_pred = dy_pred * y_std
        y_pred = y_pred * y_std + y_mean
        params.put("model", model)
        params.put("y_mean", y_mean)
        params.put("y_std", y_std)
        stream.put("y", y_pred)
        stream.put("dy", dy_pred)

    def _map(self, inputs, stream):
        y, dy = self.params().get("model").predict(inputs["K"] ** self.power, return_std=True)
        y = y * self.params().get("y_std") + self.params().get("y_mean")
        dy = dy * self.params().get("y_std")
        stream.put("y", y)
        stream.put("dy", dy)


class SupportVectorClassifier(SklearnTransform):
    default_args = dict(C=1.0, power=2, kernel="precomputed", probability=False, class_weight=None)
    req_inputs = {"K", "y"}
    allow_params = {"model"}
    allow_stream = {"y", "z", "p"}

    def _setup(self):
        self.power = self.args["power"]

    def _fit(self, inputs, stream, params):
        Kp = inputs["K"] ** self.power
        model = sklearn.svm.SVC(
            kernel=self.args["kernel"],
            C=self.args["C"],
            probability=self.args["probability"],
            class_weight=self.args["class_weight"],
        )
        model.fit(Kp, inputs["y"])
        y_pred = model.predict(Kp)
        z_pred = model.decision_function(Kp)
        params.put("model", model)
        stream.put("y", y_pred)
        stream.put("z", z_pred)
        if self.args["probability"]:
            p_pred = model.predict_proba(Kp)
            stream.put("p", p_pred)

    def _map(self, inputs, stream):
        Kp = inputs["K"] ** self.power
        y = self.params().get("model").predict(Kp)
        z = self.params().get("model").decision_function(Kp)
        stream.put("y", y)
        stream.put("z", z)
        if self.args["probability"]:
            p_pred = self.params().get("model").predict_proba(Kp)
            stream.put("p", p_pred)


class SupportVectorRegressor(SklearnTransform):
    default_args = dict(
        kernel="rbf",
        degree=3,
        gamma="scale",
        coef0=0.0,
        tol=0.001,
        C=1.0,
        epsilon=0.1,
        shrinking=True,
        cache_size=200,
        verbose=False,
        max_iter=-1,
    )
    req_inputs = {"X", "y"}
    allow_params = {"model"}
    allow_stream = {"y"}

    def _fit(self, inputs, stream, params):
        model = sklearn.svm.SVR(**self.args)
        model.fit(X=inputs["X"], y=inputs["y"])
        params.put("model", model)
        self._map(inputs, stream)

    def _map(self, inputs, stream):
        y = self.params().get("model").predict(inputs["X"])
        stream.put("y", y)


class LogisticRegression(SklearnTransform):
    default_args = dict(
        penalty="l2",
        dual=False,
        tol=0.0001,
        C=1.0,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        random_state=None,
        solver="lbfgs",
        max_iter=100,
        multi_class="auto",
        verbose=0,
        warm_start=False,
        n_jobs=None,
        l1_ratio=None,
    )
    req_inputs = {"X", "y"}
    allow_params = {"model"}
    allow_stream = {"y", "z"}

    def _fit(self, inputs, stream, params):
        model = sklearn.linear_model.LogisticRegression(**self.args)
        model.fit(X=inputs["X"], y=inputs["y"])
        params.put("model", model)
        self._map(inputs, stream)

    def _map(self, inputs, stream):
        y = self.params().get("model").predict(inputs["X"])
        z = self.params().get("model").decision_function(inputs["X"])
        stream.put("y", y)
        stream.put("z", z)


class ElasticNet(SklearnTransform):
    default_args = dict(alpha=1.0, l1_ratio=0.5)
    req_inputs = {"X", "y"}
    allow_params = {"model"}
    allow_stream = {"y", "z"}

    def _fit(self, inputs, stream, params):
        model = sklearn.linear_model.ElasticNet(**self.args)
        model.fit(X=inputs["X"], y=inputs["y"])
        params.put("model", model)
        self._map(inputs, stream)

    def _map(self, inputs, stream):
        y = self.params().get("model").predict(inputs["X"])
        stream.put("y", y)
        stream.put("z", y)


class KernelMatern(FitTransform):
    default_args = {"length_scale": 1.0, "nu": 1.5}
    req_inputs = ("X",)
    allow_params = {"X"}
    allow_stream = {"K", "K_diag"}
    stream_kernel = ("K",)
    stream_samples = ("K_diag",)
    precompute = True

    def evaluate(self, x1, x2=None, diagonal_only=False):
        kern = sklearn.gaussian_process.kernels.Matern(**self.args)
        if x2 is None:
            x2 = x1
        if diagonal_only:
            return kern.diag(x1)
        else:
            return kern(x1, x2)

    def _fit(self, inputs, stream, params):
        K = self.evaluate(inputs["X"])
        params.put("X", np.copy(inputs["X"]))
        stream.put("K", K)
        stream.put("K_diag", K.diagonal())

    def _map(self, inputs, stream):
        K = self.evaluate(inputs["X"], self.params().get("X"))
        stream.put("K", K)
        K_diag = self.evaluate(inputs["X"], inputs["X"], diagonal_only=True)
        stream.put("K_diag", K_diag)


class OrthogonalMatchingPursuit(SklearnTransform):
    default_args = dict(
        n_nonzero_coefs=None, tol=None, fit_intercept=True, normalize=True, precompute="auto"
    )
    req_inputs = {"X", "y"}
    allow_params = {"model"}
    allow_stream = {"y", "z"}

    def _fit(self, inputs, stream, params):
        model = sklearn.linear_model.OrthogonalMatchingPursuit(**self.args)
        model.fit(X=inputs["X"], y=inputs["y"])
        params.put("model", model)
        self._map(inputs, stream)

    def _map(self, inputs, stream):
        y = self.params().get("model").predict(inputs["X"])
        stream.put("y", y)
        stream.put("z", y)


class Lasso(SklearnTransform):
    # TODO
    pass


class KNeighborsRegressor(SklearnTransform):
    # TODO
    pass


class KNeighborsClassifier(SklearnTransform):
    # TODO
    pass


class AdaBoost(SklearnTransform):
    # TODO
    pass
