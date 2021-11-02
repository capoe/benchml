"""Module for Datasets classes used in BenchML."""
from __future__ import print_function

import copy
import json
import os
import warnings

import numpy as np

from benchml.readwrite import ExtendedTxt, read_extt, read_xyz


def _all_true_filter(meta):  # pylint: disable=W0613
    return True


class BenchmarkData:
    def __init__(self, root, filter_fct=None):
        if filter_fct is None:
            filter_fct = _all_true_filter
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

    def __len__(self):
        return len(self.data)


class DatasetIterator:
    def __init__(self, path=None, filter_fct=None, meta_json=None):
        self.meta = None
        if meta_json is None:
            self.path = path
            meta_json = os.path.join(path, "meta.json")
        else:
            self.path = os.path.dirname(meta_json)
        with open(meta_json, encoding="utf-8") as meta_f:
            self.meta = json.load(meta_f)
        if filter_fct is None:
            self.filter = _all_true_filter
        else:
            self.filter = filter_fct

    def __iter__(self):
        for target, target_info in self.meta["targets"].items():
            for _, dataset in enumerate(self.meta["datasets"]):
                meta_this = copy.deepcopy(self.meta)
                meta_this.pop("datasets")
                meta_this["name"] = f'{self.meta["name"]}:{target}:{dataset}'
                meta_this["target"] = target
                meta_this.update(target_info)
                if self.filter(meta_this):
                    yield Dataset(os.path.join(self.path, dataset), meta_this)


class Dataset:
    target_converter = {
        "": lambda y: y,
        "log": np.log,
        "log10": np.log10,
        "-log10": lambda y: -np.log10(y),
    }

    def __init__(self, ext_xyz=None, meta=None, configs=None):
        if configs is None:
            configs = []
        self.configs = configs
        if ext_xyz is not None:
            self.configs = self.read_data(ext_xyz)
        if meta is None:
            meta = {"name": "UNNAMED", "task": "UNKNOWN", "metrics": []}
        self.meta = meta
        self.convert = self.target_converter[self.meta.pop("convert", "")]
        self.y = None  # pylint: disable=C0103
        if "target" in meta:
            self.y = self.convert(np.array([float(s.info[meta["target"]]) for s in self.configs]))

    def __getitem__(self, key):
        item = None
        if np.issubdtype(type(key), np.integer):
            item = self.configs[key]
        elif isinstance(key, (list, np.ndarray)):
            item = Dataset(configs=[self.configs[_] for _ in key], meta=self.meta)
        elif isinstance(key, str):
            item = self.meta[key]
        else:
            raise TypeError(f"Invalid type in __getitem__: {type(key)}")
        return item

    def __len__(self):
        return len(self.configs)

    def __str__(self):
        return self.info()

    def __iter__(self):
        return self.configs.__iter__()

    def info(self):
        tmpl = (
            "{name:30s}  #configs={size:<5d}  task={task:8s}  "
            "metrics={metrics:s}   std={std:1.2e}"
        )
        if self.y is not None:
            y_std = np.std(self.y)
        else:
            y_std = np.NAN
        return tmpl.format(
            name=self.meta["name"],
            size=len(self.configs),
            task=self.meta["task"],
            metrics=",".join(self.meta["metrics"]),
            std=y_std,
        )

    @staticmethod
    def read_data(inputs, index=None):
        if isinstance(inputs, str):
            configs = read_xyz(inputs, index)
        else:
            configs = []
            for input_file in inputs:
                configs.extend(read_xyz(input_file, index))
        return configs

    @classmethod
    def create_from_file(cls, inputs, *args, **kwargs):
        index = kwargs.pop("index", None)
        configs = cls.read_data(inputs, index)
        return Dataset(configs=configs, *args, **kwargs)


class ExttDataset:
    def __init__(self, extt=None, meta=None):
        if extt is None:
            extt = ExtendedTxt()
        self.meta = extt.meta if meta is None else meta
        self.arrays = extt.arrays

    def __getitem__(self, key):
        item = None
        if np.issubdtype(type(key), np.integer):
            input_array_name = "X"
            array_names = tuple(self.arrays.keys())
            if len(array_names) > 0:
                if input_array_name not in self.arrays.keys():
                    input_array_name = array_names[0]
            else:
                raise KeyError("No arrays")
            item = self.arrays[input_array_name][key]
        elif isinstance(key, (list, np.ndarray)):
            item = self.slice(key)
        elif isinstance(key, str):
            item = self.meta[key]
        else:
            raise TypeError(f"Invalid type in __getitem__: {type(key)}")
        return item

    def __iter__(self):
        return self.arrays.__iter__()

    def __len__(self):
        array_names = list(self.arrays.keys())
        if len(array_names) < 1:
            res = 0
        else:
            res = len(self.arrays[array_names[0]])
        return res

    def __str__(self):
        return self.info()

    def __contains__(self, key):
        return key in self.meta

    def slice(self, idcs):
        arrays_sliced = {k: v[idcs] for k, v in self.arrays.items()}
        return ExttDataset(extt=ExtendedTxt(arrays=arrays_sliced, meta=self.meta))

    def info(self):
        tmpl = f"ExttDataset with {len(self.arrays)} arrays:"
        arr_info = [f"Array[{name}{repr(arr.shape)}]" for name, arr in self.arrays.items()]
        return " ".join([tmpl, *arr_info])

    @staticmethod
    def read_data_from_file(input_file):
        return read_extt(input_file)

    @classmethod
    def create_from_file(cls, input_file, *args, **kwargs):
        extt = cls.read_data_from_file(input_file)
        return ExttDataset(extt=extt, *args, **kwargs)


def compile(root=None, **kwargs):
    warnings.warn("Use BenchmarkData() directly", DeprecationWarning)
    if root is None:
        root = "./data"
    return BenchmarkData(root=root, **kwargs)


DATASET_FORMATS = {
    ".extt": ExttDataset,
    ".xyz": Dataset,
}


def load_dataset(filename, *args, **kwargs):
    _, ext = os.path.splitext(filename)
    try:
        dataset_class = DATASET_FORMATS[ext]
    except KeyError as no_format:
        raise ValueError(f"Unsupported dataset format: {ext}") from no_format
    dataset = dataset_class.create_from_file(filename, *args, **kwargs)
    return dataset


if __name__ == "__main__":
    bench = BenchmarkData("./data")
    for data in bench:
        print(data)
