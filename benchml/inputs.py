from benchml.pipeline import InputTransform


class ExttInput(InputTransform):
    """An EXTendedTxt Data Input Node.

    This node is supposed to be the beginning of a Pipeline,
    providing EXTendedTxt type of data.

    See Also
    --------
    benchml.data.ExttDataset : The type of Dataset it expects.
    """

    allow_stream = {"X", "Y", "meta"}
    stream_copy = {
        "meta",
    }
    stream_samples = {"X", "Y"}

    def _feed(self, data, stream):
        for key, v in data.arrays.items():
            stream.put(key, v, force=True)
        stream.put("meta", data.meta)


class ExtXyzInput(InputTransform):
    """An ExtXyz Data Input Node.

    This node is supposed to be the beginning of a Pipeline,
    providing ExtXyz type of data.

    See Also
    --------
    benchml.data.Dataset : ExtXyz Dataset - the type of dataset it expects.
    """

    allow_stream = {"configs", "y", "meta"}
    stream_copy = ("meta",)
    stream_samples = ("configs", "y")

    def _feed(self, data, stream):
        stream.put("configs", data)
        if hasattr(data, "y"):
            stream.put("y", data.y)
        else:
            stream.put("y", [])
        if hasattr(data, "meta"):
            stream.put("meta", data.meta)
        else:
            stream.put("meta", {})
