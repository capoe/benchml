import numpy as np
from .pipeline import Transform, Params
from .logger import log

class EnsembleRegressor(Transform):
    default_args = {
        "size": 100,
        "bootstrap_samples": True,
        "bootstrap_features": False,
        "feature_fraction": 0.1
    }
    req_inputs = {"X","y","base_transform"}
    allow_stream = {"y", "dy"}
    allow_params = {"samples", "features", "params"}
    def fitSingle(self, base, stream, X, y):
        params_s = Params(tag="", tf=base)
        sel_samples = None
        sel_features = None
        Xs = X
        ys = y
        if self.args["bootstrap_samples"]:
            sel_samples = np.random.randint(0, X.shape[0], size=(X.shape[0],))
            Xs = Xs[sel_samples]
            ys = ys[sel_samples]
        if self.args["bootstrap_features"]:
            n_feature_sel = int(Xs.shape[1]*self.args["feature_fraction"])
            sel_features = np.arange(0, Xs.shape[1])
            np.random.shuffle(sel_features)
            sel_features = sel_features[0:n_feature_sel]
            sel_features = sorted(sel_features)
            Xs = Xs[:,sel_features]
        base._fit({"X": Xs, "y": ys}, stream, params_s)
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
        for f, pars in zip(
                self.params().get("features"), 
                self.params().get("params")):
            Xs = inputs["X"]
            base.active_params = pars
            if f is not None:
                base._map({"X": Xs[:,f]}, stream)
            else:
                base._map({"X": Xs}, stream)
            Y.append(stream.get("y"))
        Y = np.array(Y)
        y = np.mean(Y, axis=0)
        dy = np.std(Y, axis=0)
        stream.put("y", y)
        stream.put("dy", dy)

