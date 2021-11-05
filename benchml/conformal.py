from abc import ABC

import numpy as np
from scipy.optimize import curve_fit

from benchml.logger import log
from benchml.pipeline import FitTransform, Params
from benchml.splits import Split


def fsigmoid(x, a, b):
    return 1.0 / (1.0 + np.exp(-a * (x - b)))


class ConformalBase(FitTransform, ABC):
    default_args = {
        "split": {"method": "random", "n_splits": 10, "train_fraction": 0.9},
        "epsilon": 1e-10,
        "forward_inputs": {"X": "X", "y": "y"},
        "input_type": "descriptor",
    }
    slice_funcs_fit = {"kernel": "lambda X, s: X[s][:,s]", "descriptor": "lambda X, s: X[s]"}
    slice_funcs_map = {"kernel": "lambda X, s, r: X[s][:,r]", "descriptor": "lambda X, s, r: X[s]"}
    req_inputs = {"X", "y", "base_transform"}
    allow_stream = {"y", "dy", "z", "p", "dy_noncalibrated", "p_noncalibrated"}
    allow_params = {"params", "alpha", "scores", "sigmoid_a", "sigmoid_b"}


class ConformalRegressor(ConformalBase):
    default_args = {"confidence": [0.67], **ConformalBase.default_args}

    def _fit(self, inputs, stream, params):
        base = inputs["base_transform"]
        base._setup()
        self.allow_stream = self.allow_stream.union(base.allow_stream)
        inputs_base = base.resolveInputs(stream)
        fwd = self.args["forward_inputs"]
        inputs_fwd = {fwd[k]: v for k, v in inputs.items() if k in fwd}
        slice_func_fit = eval(self.slice_funcs_fit[self.args["input_type"]])
        slice_func_map = eval(self.slice_funcs_map[self.args["input_type"]])
        Y = []
        Y_pred = []
        dY_pred = []
        # Cross-calibrate
        for info, train, calibrate in Split(len(inputs["y"]), **self.args["split"]):
            params_cal = Params(tag="", tf=base)
            base.active_params = params_cal
            base._fit(
                {
                    fwd["X"]: slice_func_fit(inputs["X"], train),
                    fwd["y"]: inputs["y"][train],
                    **inputs_base,
                },
                stream,
                params_cal,
            )
            base._map(
                {fwd["X"]: slice_func_map(inputs["X"], calibrate, train), **inputs_base}, stream
            )
            Y.append(inputs["y"][calibrate])
            Y_pred.append(stream.get("y"))
            dY_pred.append(stream.get("dy"))
        Y = np.concatenate(Y)
        Y_pred = np.concatenate(Y_pred)
        dY_pred = np.concatenate(dY_pred)
        scores = np.abs(Y - Y_pred) / (dY_pred + self.args["epsilon"])
        scores = np.sort(scores)
        params.put("scores", scores)
        alpha = np.percentile(scores, list(map(lambda c: 100 * c, self.args["confidence"])))
        # Refit on entire dataset
        params_cal = Params(tag="", tf=base)
        base.active_params = params_cal
        base._fit({**inputs_fwd, **inputs_base}, stream, params_cal)
        params.put("alpha", alpha)
        params.put("params", params_cal)
        self._map(inputs, stream)

    def _map(self, inputs, stream):
        base = inputs["base_transform"]
        inputs_base = base.resolveInputs(stream)
        base.active_params = self.params().get("params")
        base._map({**inputs, **inputs_base}, stream)
        dy = stream.get("dy")
        dy_calibrated = dy * self.params().get("alpha")
        stream.put("dy", dy_calibrated)
        stream.put("dy_noncalibrated", dy)


class ConformalClassifier(ConformalBase):
    default_args = {
        "epsilon": 1e-10,
        "class_threshold": 0.5,
        "sigmoid_fit": False,
        **ConformalBase.default_args,
    }

    def calibrate(self, scores):
        alpha = self.params().get("alpha")
        nonc_neg = scores
        nonc_pos = -scores
        rank_neg = np.searchsorted(alpha[0], nonc_neg)
        rank_pos = np.searchsorted(alpha[1], nonc_pos)
        rank_neg = 1.0 - 1.0 * rank_neg / len(alpha[0])
        rank_pos = 1.0 - 1.0 * rank_pos / len(alpha[1])
        return np.array([rank_neg, rank_pos]).T

    def _fit(self, inputs, stream, params):
        base = inputs["base_transform"]
        base._setup()
        # TODO: check that the base transform doesn't have input defined
        inputs_base = base.resolveInputs(stream)
        fwd = self.args["forward_inputs"]
        inputs_fwd = {fwd[k]: v for k, v in inputs.items() if k in fwd}
        slice_func_fit = eval(self.slice_funcs_fit[self.args["input_type"]])
        slice_func_map = eval(self.slice_funcs_map[self.args["input_type"]])
        Y = []
        Z_pred = []
        # Cross-calibrate
        for info, train, calibrate in Split(len(inputs["y"]), **self.args["split"]):
            params_cal = Params(tag="", tf=base)
            base.active_params = params_cal
            base._fit(
                {
                    fwd["X"]: slice_func_fit(inputs["X"], train),
                    fwd["y"]: inputs["y"][train],
                    **inputs_base,
                },
                stream,
                params_cal,
            )
            base._map(
                {fwd["X"]: slice_func_map(inputs["X"], calibrate, train), **inputs_base}, stream
            )
            Y.append(inputs["y"][calibrate])
            Z_pred.append(stream.get("z"))
            log << log.debug << f"  [{self.tag}]: calibration split {info}" << log.endl
        Y = np.concatenate(Y)
        Z_pred = np.concatenate(Z_pred)
        # use sigmoid to allow for maximum separation
        if self.args["sigmoid_fit"]:
            popt, pcov = curve_fit(fsigmoid, Z_pred, Y, method="dogbox")
            self.params().put("sigmoid_a", popt[0])
            self.params().put("sigmoid_b", popt[1])
            Z_pred = fsigmoid(Z_pred, popt[0], popt[1])
        # Evaluate non-conformity
        neg = np.where(Y < self.args["class_threshold"])
        pos = np.where(Y >= self.args["class_threshold"])
        Z_pred_neg = Z_pred[neg]
        Z_pred_pos = Z_pred[pos]
        nonc_neg = np.sort(Z_pred_neg)
        nonc_pos = np.sort(-Z_pred_pos)
        alpha = [nonc_neg, nonc_pos]
        # Refit on entire dataset
        params_cal = Params(tag="", tf=base)
        base.active_params = params_cal
        base._fit({**inputs_fwd, **inputs_base}, stream, params_cal)
        params.put("alpha", alpha)
        params.put("params", params_cal)
        self._map(inputs, stream)

    def _map(self, inputs, stream):
        # TODO: check that the base transform doesn't have input defined
        base = inputs["base_transform"]
        inputs_base = base.resolveInputs(stream)
        fwd = self.args["forward_inputs"]
        inputs_fwd = {fwd[k]: v for k, v in inputs.items() if k in fwd}
        base.active_params = self.params().get("params")
        base._map({**inputs_fwd, **inputs_base}, stream)
        if self.args["sigmoid_fit"]:
            a = self.params().get("sigmoid_a")
            b = self.params().get("sigmoid_b")
            Z_pred = fsigmoid(stream.get("z"), a, b)
        else:
            Z_pred = stream.get("z")
        stream.put("p_noncalibrated", Z_pred)
        probs = self.calibrate(Z_pred)
        stream.put("p", probs)


class ConformalMultiClassifier(ConformalBase):
    default_args = {"epsilon": 1e-10, "class_threshold": 0.5, **ConformalBase.default_args}

    def calibrate(self, scores):
        alphav = self.params().get("alpha")
        resp = []
        resn = []
        for i, alpha in enumerate(alphav):
            nonc_neg = scores[:, i]
            nonc_pos = -scores[:, i]
            rank_neg = np.searchsorted(alpha[0], nonc_neg)
            rank_pos = np.searchsorted(alpha[1], nonc_pos)
            rank_neg = 1.0 - 1.0 * rank_neg / len(alpha[0])
            rank_pos = 1.0 - 1.0 * rank_pos / len(alpha[1])
            resp.append(np.array(rank_pos))
            resn.append(np.array(rank_neg))
        return np.array([resn, resp]).T

    def _fit(self, inputs, stream, params):
        base = inputs["base_transform"]
        base._setup()
        # TODO: check that the base transform doesn't have input defined
        inputs_base = base.resolveInputs(stream)
        fwd = self.args["forward_inputs"]
        inputs_fwd = {fwd[k]: v for k, v in inputs.items() if k in fwd}
        slice_func_fit = eval(self.slice_funcs_fit[self.args["input_type"]])
        slice_func_map = eval(self.slice_funcs_map[self.args["input_type"]])
        Y = []
        Z_pred = []
        # Cross-calibrate
        for info, train, calibrate in Split(len(inputs["y"]), **self.args["split"]):
            log << log.debug << "Conformal fit %s" % info << log.endl
            params_cal = Params(tag="", tf=base)
            base.active_params = params_cal
            base._fit(
                {
                    fwd["X"]: slice_func_fit(inputs["X"], train),
                    fwd["y"]: inputs["y"][train],
                    **inputs_base,
                },
                stream,
                params_cal,
            )
            base._map(
                {fwd["X"]: slice_func_map(inputs["X"], calibrate, train), **inputs_base}, stream
            )
            Y.append(inputs["y"][calibrate])
            Z_pred.append(stream.get("z"))
        Y = np.concatenate(Y)
        Z_pred = np.concatenate(Z_pred)
        # Evaluate non-conformity
        classes = set(Y)
        alpha = []
        for i, _ in enumerate(classes):
            neg = np.where(Z_pred[:, i] < self.args["class_threshold"])
            pos = np.where(Z_pred[:, i] >= self.args["class_threshold"])
            Z_pred_neg = Z_pred[neg][:, i]
            Z_pred_pos = Z_pred[pos][:, i]
            nonc_neg = np.sort(Z_pred_neg)
            nonc_pos = np.sort(-Z_pred_pos)
            alpha.append([nonc_neg, nonc_pos])
        # Refit on entire dataset
        params_cal = Params(tag="", tf=base)
        base.active_params = params_cal
        base._fit({**inputs_fwd, **inputs_base}, stream, params_cal)
        params.put("alpha", alpha)
        params.put("params", params_cal)
        self._map(inputs, stream)

    def _map(self, inputs, stream):
        # TODO: check that the base transform doesn't have input defined
        base = inputs["base_transform"]
        inputs_base = base.resolveInputs(stream)
        fwd = self.args["forward_inputs"]
        inputs_fwd = {fwd[k]: v for k, v in inputs.items() if k in fwd}
        base.active_params = self.params().get("params")
        base._map({**inputs_fwd, **inputs_base}, stream)
        Z_pred = stream.get("z")
        stream.put("p_noncalibrated", Z_pred)
        probs = self.calibrate(Z_pred)
        stream.put("p", probs)
