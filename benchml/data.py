from __future__ import print_function

import copy
import json
import os

import numpy as np

from benchml.readwrite import ExtendedTxt, read_extt, read_xyz


class BenchmarkData(object):
    def __init__(self, root, filter_fct=lambda meta: True):
        self.paths = map(
            lambda sdf: sdf[0],
            filter(lambda subdir_dirs_files: "meta.json" in subdir_dirs_files[2], os.walk(root)),
        )
        self.dataits = map(lambda path: DatasetIterator(path, filter_fct=filter_fct), self.paths)
        self.data = []

    def __iter__(self):
        self.data = []
        for datait in self.dataits:
            for dataset in datait:
                self.data.append(dataset)
                yield dataset
        return

    def __len__(self):
        return len(self.data)


class DatasetIterator(object):
    def __init__(self, path=None, filter_fct=None, meta_json=None):
        if meta_json is None:
            self.path = path
            self.meta = json.load(open(os.path.join(path, "meta.json")))
        else:
            self.path = os.path.dirname(meta_json)
            self.meta = json.load(open(meta_json))
        if filter_fct is None:
            self.filter = lambda meta: True
        else:
            self.filter = filter_fct
        return

    def __iter__(self):
        for target, target_info in self.meta["targets"].items():
            for didx, dataset in enumerate(self.meta["datasets"]):
                meta_this = copy.deepcopy(self.meta)
                meta_this.pop("datasets")
                meta_this["name"] = "{0}:{1}:{2}".format(self.meta["name"], target, dataset)
                meta_this["target"] = target
                meta_this.update(target_info)
                if self.filter(meta_this):
                    yield Dataset(os.path.join(self.path, dataset), meta_this)
        return


class Dataset(object):
    target_converter = {
        "": "lambda y: y",
        "log": "lambda y: np.log(y)",
        "log10": "lambda y: np.log10(y)",
        "-log10": "lambda y: -np.log10(y)",
    }

    def __init__(self, ext_xyz=None, meta=None, configs=None):
        self.configs = configs
        if ext_xyz is not None:
            self.configs = self.read_data(ext_xyz)
        if meta is None:
            meta = {}
        self.meta = meta
        self.convert = self.target_converter[self.meta.pop("convert", "")]
        if "target" in meta:
            self.y = eval(self.convert)(
                np.array([float(s.info[meta["target"]]) for s in self.configs])
            )
        return

    def info(self):
        s = "{name:30s}  #configs={size:<5d}  task={task:8s}  metrics={metrics:s}   std={std:1.2e}"
        return s.format(
            name=self.meta["name"],
            size=len(self.configs),
            task=self.meta["task"],
            metrics=",".join(self.meta["metrics"]),
            std=np.std(self.y),
        )

    def __getitem__(self, key):
        if np.issubdtype(type(key), np.integer):
            return self.configs[key]
        elif type(key) in {list, np.ndarray}:
            return Dataset(configs=[self.configs[_] for _ in key], meta=self.meta)
        elif type(key) is str:
            return self.meta[key]
        else:
            raise TypeError("Invalid type in __getitem__: %s" % type(key))

    def __len__(self):
        return len(self.configs)

    def __str__(self):
        return self.info()

    def __iter__(self):
        return self.configs.__iter__()

    @staticmethod
    def read_data(data, index=None):
        if type(data) is str:
            configs = read_xyz(data, index)
        else:
            configs = []
            for input_file in data:
                configs.extend(read_xyz(input_file, index))
        return configs

    @classmethod
    def create_from_file(cls, data, *args, **kwargs):
        index = kwargs.pop("index", None)
        configs = cls.read_data(data, index)
        return Dataset(configs=configs, *args, **kwargs)


class ExttDataset(object):
    def __init__(self, extt, meta=None):
        self.meta = extt.meta if meta is None else meta
        self.arrays = extt.arrays

    def __getitem__(self, key):
        if np.issubdtype(type(key), np.integer):
            return self.arrays["X"][key]
        elif type(key) in {list, np.ndarray}:
            return self.slice(key)
        elif type(key) is str:
            return self.meta[key]
        else:
            raise TypeError("Invalid type in __getitem__: %s" % type(key))

    def slice(self, idcs):
        arrays_sliced = {k: v[idcs] for k, v in self.arrays.items()}
        return ExttDataset(extt=ExtendedTxt(arrays=arrays_sliced, meta=self.meta))

    def __len__(self):
        return len(self.arrays[list(self.arrays.keys())[0]])

    def __str__(self):
        return self.info()

    def __contains__(self, key):
        return key in self.meta

    def info(self):
        s = "ExttDataset with %d arrays: " % (len(self.arrays))
        for name, x in self.arrays.items():
            s += "Array[%s%s] " % (name, repr(x.shape))
        return s

    @staticmethod
    def read_data_from_file(input_file):
        return read_extt(input_file)

    @classmethod
    def create_from_file(cls, input_file, *args, **kwargs):
        extt = cls.read_data_from_file(input_file)
        return ExttDataset(extt=extt, *args, **kwargs)


def compile(root="./data", filter_fct=lambda meta: True):
    return BenchmarkData(root, filter_fct=filter_fct)


DATASET_FORMATS = {
    ".extt": ExttDataset,
    ".xyz": Dataset,
}


def load_dataset(filename, *args, **kwargs):
    base, ext = os.path.splitext(filename)
    dataset = DATASET_FORMATS[ext].create_from_file(filename, *args, **kwargs)
    return dataset


if __name__ == "__main__":
    bench = compile()
    for data in bench:
        print(data)
