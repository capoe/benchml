from benchml.pipeline import Transform


class ExttInput(Transform):
    allow_stream = {"X", "Y", "meta"}
    stream_copy = {
        "meta",
    }
    stream_samples = {"X", "Y"}

    def _feed(self, data, stream):
        for key, v in data.arrays.items():
            stream.put(key, v, force=True)
        stream.put("meta", data.meta)


class ExtXyzInput(Transform):
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
