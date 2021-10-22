import numpy as np

from benchml.pipeline import Transform


class Add(Transform):
    req_args = {
        "coeffs",
    }
    req_inputs = {
        "X",
    }
    allow_stream = {
        "y",
    }

    def _map(self, inputs, stream):
        coeffs = self.args["coeffs"]
        assert len(coeffs) == len(inputs["X"])
        y = np.zeros_like(inputs["X"][0])
        for i in range(len(inputs["X"])):
            y = y + coeffs[i] * inputs["X"][i]
        stream.put("y", y)


class Mult(Transform):
    req_inputs = ("X",)
    allow_stream = {"y"}

    def _map(self, inputs, stream):
        y = np.ones_like(inputs["X"][0])
        for i in range(len(inputs["X"])):
            y = y * inputs["X"][i]
        stream.put("y", y)


class Exp(Transform):
    req_inputs = {
        "X",
    }
    default_args = {"coeff": +1}
    allow_stream = {
        "X",
    }

    def _map(self, inputs, stream):
        stream.put("X", np.exp(self.args["coeff"] * inputs["X"]))


class Delta(Transform):
    allow_stream = {"y"}
    req_inputs = {"target", "ref"}
    stream_samples = ("y",)

    def _map(self, inputs, stream):
        stream.put("y", None)

    def _fit(self, inputs, stream, params):
        delta = inputs["target"] - inputs["ref"]
        stream.put("y", delta)


class RankNorm(Transform):
    req_inputs = {
        "z",
    }
    allow_params = {
        "z",
    }
    allow_stream = {
        "z",
    }

    def _fit(self, inputs, stream, params):
        z = inputs["z"]
        z_ranked = np.sort(z)
        params.put("z", z_ranked)
        self._map(inputs, stream)

    def _map(self, inputs, stream):
        ranked = np.searchsorted(self.params().get("z"), inputs["z"]) / len(self.params().get("z"))
        stream.put("z", ranked)


class SliceMatrix(Transform):
    allow_params = {
        "slice",
    }
    allow_stream = {
        "X",
    }
    default_args = {"axis": None}
    req_inputs = {"slice", "X"}

    def _fit(self, inputs, stream, params):
        if self.args["axis"] is None:
            slice = inputs["slice"]
        elif isinstance(self.args["axis"], int):
            slice = [slice(None) for r in len(inputs["X"].shape)]
            slice[self.args["axis"]] = inputs["slice"]
        else:
            raise ValueError("SliceMatrix arg 'slice' expects None or int")
        params.put("slice", slice)
        return self._map(inputs, stream)

    def _map(self, inputs, stream):
        s = self.params().get("slice")
        stream.put("X", inputs["X"][s])


class DoDivideBySize(Transform):
    default_args = {
        "config_to_size": "lambda c: len(c)",
        "skip_if_not_force": False,
        "force": False,
    }
    req_inputs = ("y", "configs", "meta")
    allow_stream = ("y", "sizes")
    allow_params = ("divide_by_size",)

    def checkDoDivide(self, inputs):
        do_divide_by_size = False
        if self.args["force"]:
            do_divide_by_size = True
        elif self.args["skip_if_not_force"]:
            pass
        elif inputs["meta"]["scaling"] == "additive":
            do_divide_by_size = True
        elif inputs["meta"]["scaling"] == "unknown":
            pass
        elif inputs["meta"]["scaling"] == "non-additive":
            pass
        else:
            raise ValueError("Scaling should be one of additive|non-additive|unknown")
        return do_divide_by_size

    def _fit(self, inputs, stream, params):
        do_div = self.checkDoDivide(inputs)
        y_in = inputs["y"]
        if not do_div:
            sizes = np.ones_like(inputs["y"])
            y_out = np.copy(y_in)
        else:
            if type(self.args["config_to_size"]) is str:
                s_fct = eval(self.args["config_to_size"])
            else:
                s_fct = self.args["config_to_size"]
            configs = inputs["configs"]
            sizes = np.array(list(map(s_fct, configs)))
            assert np.min(sizes) > 0  # DoDivideBySize: sample size <= 0 not allowed
            y_out = y_in / sizes
        params.put("divide_by_size", do_div)
        stream.put("y", y_out)
        stream.put("sizes", sizes)

    def _map(self, inputs, stream):
        do_div = self.params().get("divide_by_size")
        configs = inputs["configs"]
        if not do_div:
            sizes = np.ones((len(configs),))
        else:
            if type(self.args["config_to_size"]) is str:
                s_fct = eval(self.args["config_to_size"])
            else:
                s_fct = self.args["config_to_size"]
            sizes = np.array(list(map(s_fct, configs)))
            assert np.min(sizes) > 0  # DoDivideBySize: sample size <= 0 not allowed
        stream.put("y", None)
        stream.put("sizes", sizes)


class UndoDivideBySize(Transform):
    req_inputs = (
        "y",
        "sizes",
    )
    allow_stream = ("y",)

    def _map(self, inputs, stream):
        y_in = inputs["y"]
        sizes = inputs["sizes"]
        assert y_in.shape[0] == sizes.shape[0]  # UndoDivideBySize: inconsistent input dim
        y_out = y_in * sizes
        stream.put("y", y_out)
