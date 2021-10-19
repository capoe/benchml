import numpy as np

from .logger import log
from .pipeline import Params, Transform


class EnsembleBase(Transform):
    default_args = {
        "size": 100,
        "bootstrap_samples": True,
        "bootstrap_features": False,
        "feature_fraction": 0.1,
        "forward_inputs": {"X": "X", "y": "y"},
        "input_type": "descriptor",
    }
    slice_funcs_fit = {"kernel": "lambda X, s: X[s][:,s]", "descriptor": "lambda X, s: X[s]"}
    slice_funcs_map = {"kernel": "lambda X, s: X[:,s]", "descriptor": "None"}
    req_inputs = {"X", "y", "base_transform"}
    allow_stream = {"y", "dy"}
    allow_params = {"samples", "features", "params"}


class EnsembleRegressor(EnsembleBase):
    def fitSingle(self, base, stream, X, y):
        params_s = Params(tag="", tf=base)
        fwd = self.args["forward_inputs"]
        slice_func_fit = eval(self.slice_funcs_fit[self.args["input_type"]])
        sel_samples = None
        sel_features = None
        Xs = X
        ys = y
        if self.args["bootstrap_samples"]:
            sel_samples = np.random.randint(0, X.shape[0], size=(X.shape[0],))
            Xs = slice_func_fit(Xs, sel_samples)
            ys = ys[sel_samples]
        if self.args["bootstrap_features"]:
            n_feature_sel = int(Xs.shape[1] * self.args["feature_fraction"])
            sel_features = np.arange(0, Xs.shape[1])
            np.random.shuffle(sel_features)
            sel_features = sel_features[0:n_feature_sel]
            sel_features = sorted(sel_features)
            Xs = Xs[:, sel_features]
        base._fit({fwd["X"]: Xs, fwd["y"]: ys}, stream, params_s)
        return params_s, sel_samples, sel_features

    def _fit(self, inputs, stream, params):
        base_trafo = inputs["base_transform"]
        X = inputs["X"]
        y = inputs["y"]
        self.allow_stream = self.allow_stream.union(base_trafo.allow_stream)
        samples_list = []
        features_list = []
        params_list = []
        for s in range(self.args["size"]):
            log << log.debug << "Ensemble fit %s" % s << log.endl
            pars, samples, features = self.fitSingle(base_trafo, stream, X, y)
            params_list.append(pars)
            samples_list.append(samples)
            features_list.append(features)
        self.params().put("params", params_list)
        self.params().put("samples", samples_list)
        self.params().put("features", features_list)
        self._map(inputs, stream)

    def _map(self, inputs, stream):
        Y = []
        base = inputs["base_transform"]
        fwd = self.args["forward_inputs"]
        slice_func_map = eval(self.slice_funcs_map[self.args["input_type"]])
        for f, s, pars in zip(
            self.params().get("features"), self.params().get("samples"), self.params().get("params")
        ):
            base.active_params = pars
            Xs = inputs["X"]
            if f is not None:
                Xs = Xs[:, f]
            if slice_func_map is not None:
                Xs = slice_func_map(Xs, s)
            base._map({fwd["X"]: Xs}, stream)
            Y.append(stream.get("y"))
        Y = np.array(Y)
        y = np.mean(Y, axis=0)
        dy = np.std(Y, axis=0)
        stream.put("y", y)
        stream.put("dy", dy)
