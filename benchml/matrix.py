import numpy as np

from benchml.pipeline import Transform


class Reshape(Transform):
    default_args = {"shape": [-1, 1], "calc_shape": None}
    req_args = {
        "shape",
    }
    req_inputs = {
        "X",
    }
    allow_stream = {
        "X",
    }

    def _default_reshape(self, X):
        return X.reshape(self.args["shape"])

    def _map(self, inputs, stream):
        if self.args["shape"] is not None:
            to_shape = self._default_reshape
        else:
            to_shape = eval(self.args["calc_shape"])
        stream.put("X", to_shape(inputs["X"]))


class Concatenate(Transform):
    req_inputs = ("X",)
    allow_stream = {
        "X",
    }
    stream_samples = ("X",)
    precompute = True
    default_args = {"axis": 1}

    def _map(self, inputs, stream):
        X_out = np.concatenate(inputs["X"], axis=self.args["axis"])
        stream.put("X", X_out)


class WhitenMatrix(Transform):
    default_args = {"centre": True, "scale": True, "epsilon": 1e-10}
    req_inputs = ("X",)
    allow_params = ("x_avg", "x_std")
    allow_stream = ("X",)

    def _fit(self, inputs, stream, params):
        x_avg = np.mean(inputs["X"], axis=0)
        x_std = np.std(inputs["X"], axis=0) + self.args["epsilon"]
        params.put("x_avg", x_avg)
        params.put("x_std", x_std)
        self._map(inputs, stream)

    def _map(self, inputs, stream):
        if self.args["centre"]:
            X_w = inputs["X"] - self.params().get("x_avg")
        else:
            X_w = inputs["X"]
        if self.args["scale"]:
            X_w = X_w / self.params().get("x_std")
        stream.put("X", X_w)


class SubsampleMatrix(Transform):
    default_args = {
        "info_key": None,
    }
    allow_stream = {
        "X",
    }

    def _map(self, inputs, stream):
        raise NotImplementedError()  # TODO Generalize
        X = inputs["X"]
        configs = inputs["configs"]
        X_out = []
        for i in range(len(configs)):
            assert len(X[i]) == len(configs[i].symbols)
            sel = int(configs[i].info[self.args["info_key"]])
            X_out.append(X[i][sel])
        X_out = np.array(X_out)
        stream.put("X", X_out)


class ReduceMatrix(Transform):
    req_inputs = ("X",)
    default_args = {"reduce": "np.sum(x, axis=0)", "norm": True, "epsilon": 1e-10}
    allow_stream = ("X",)
    stream_samples = ("X",)

    def _map(self, inputs, stream):
        X = map(lambda x: eval(self.args["reduce"]), inputs["X"])
        X = map(lambda x: x / (np.dot(x, x) ** 0.5 + self.args["epsilon"]), X)
        X = map(lambda x: x.reshape((1, -1)), X)
        X = np.concatenate(list(X), axis=0)
        stream.put("X", X)


class ReduceTypedMatrix(Transform):
    default_args = {
        "reduce_op": "sum",
        "normalize": False,
        "reduce_by_type": False,
        "types": None,
        "epsilon": 1e-10,
    }
    req_inputs = ("X",)
    allow_stream = ("X",)
    allow_params = ("types",)
    allow_ops = {"sum": np.sum, "mean": np.mean}
    stream_samples = ("X",)

    def _setup(self, *args, **kwargs):
        assert self.args["reduce_op"] in self.allow_ops.keys()  # Only 'sum' and 'mean' allowed
        if self.args["reduce_by_type"]:
            assert "T" in self.inputs  # Require input T if reduce_by_type = True

    def _fit(self, inputs, stream, params):
        if self.args["reduce_by_type"]:
            if self.args["types"] is not None:
                self.types = self.args["types"]
            else:
                self.types = inputs["meta"]["elements"]
            self.type_to_idx = {t: tidx for tidx, t in enumerate(self.types)}
            params.put("types", self.types)
        self._map(inputs, stream)

    def _map(self, inputs, stream):
        X_red = []
        for idx, x in enumerate(inputs["X"]):
            if self.args["reduce_by_type"]:
                x_red = np.zeros((len(self.types), x.shape[1]))
                t_count = np.zeros((len(self.type_to_idx),))
                for i, t in enumerate(map(lambda t: self.type_to_idx[t], inputs["T"][idx])):
                    x_red[t] = x_red[t] + x[i]
                    t_count[t] += 1
                if self.args["reduce_op"] == "mean":
                    x_red = (x_red.T / (t_count + self.args["epsilon"])).T
                x_red = x_red.flatten()
            else:
                chosen_op = self.allow_ops[self.args["reduce_op"]]
                x_red = chosen_op(x, axis=0)
            if self.args["normalize"]:
                x_red = x_red / (np.sqrt(np.dot(x_red, x_red)) + self.args["epsilon"])
            X_red.append(x_red)
        X_red = np.array(X_red)
        stream.put("X", X_red)
