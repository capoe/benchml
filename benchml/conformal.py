import numpy as np
from .pipeline import Transform, Params
from .splits import Split
from .logger import log

class ConformalRegressor(Transform):
    default_args = {
        "confidence": [ 0.67 ],
        "split": {
            "method": "random",
            "n_splits": 10,
            "train_fraction": 0.9
        },
        "epsilon": 1e-10,
    }
    req_inputs = {'X','y','base_transform'}
    allow_stream = {'y','dy'}
    allow_params = {'params', 'alpha'}
    def _fit(self, inputs, stream, params):
        base = inputs["base_transform"]
        inputs_base = base.resolveInputs(stream)
        X = inputs["X"]
        y = inputs["y"]
        Y = []
        Y_pred = []
        dY_pred = []
        # Cross-calibrate
        for info, train, calibrate in Split(len(y), **self.args["split"]):
            log << log.debug << "Conformal fit %s" % info << log.endl
            params = Params(tag="", tf=base)
            base.active_params = params
            base._fit({"X": X[train], "y": y[train], **inputs_base}, stream, params)
            base._map({"X": X[calibrate], **inputs_base}, stream)
            Y.append(y[calibrate])
            Y_pred.append(stream.get("y"))
            dY_pred.append(stream.get("dy"))
        Y = np.concatenate(Y)
        Y_pred = np.concatenate(Y_pred)
        dY_pred = np.concatenate(dY_pred)
        scores = np.abs(Y-Y_pred)/(dY_pred+self.args["epsilon"])
        scores = np.sort(scores)
        alpha = np.percentile(scores, list(map(lambda c: 100*c, self.args["confidence"])))
        # Refit on entire dataset
        params = Params(tag="", tf=base)
        base.active_params = params
        base._fit({**inputs, **inputs_base}, stream, params)
        self.params().put("alpha", alpha)
        self.params().put("params", params)
        self._map(inputs, stream)
    def _map(self, inputs, stream):
        base = inputs["base_transform"]
        inputs_base = base.resolveInputs(stream)
        base.active_params = self.params().get("params")
        base._map({**inputs, **inputs_base}, stream)
        dy = stream.get("dy")
        dy_calibrated = stream.get("dy")*self.params().get("alpha")
        stream.put("dy", dy_calibrated)

