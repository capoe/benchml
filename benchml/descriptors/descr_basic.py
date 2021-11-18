import numpy as np

from benchml.pipeline import Transform


class DescriptorRandom(Transform):
    req_args = ("dim",)
    req_inputs = ("configs",)
    allow_stream = {"X"}
    stream_samples = ("X",)
    precompute = True

    def _map(self, inputs, stream):
        X = np.random.uniform(0.0, 1.0, size=(len(inputs["configs"]), self.args["dim"]))
        stream.put("X", X)
