from __future__ import print_function
import numpy as np
import json
import os
import copy
from .readwrite import read

class BenchmarkData(object):
    def __init__(self, root, filter_fct=lambda meta: True):
        self.paths = map(lambda sdf: sdf[0], filter(
            lambda subdir_dirs_files: "meta.json" in subdir_dirs_files[2],
                os.walk(root)))
        self.dataits = map(lambda path: DatasetIterator(
            path, filter_fct=filter_fct), self.paths)
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
    def __init__(self, path=None, filter_fct=lambda meta: True, meta_json=None):
        if meta_json is None:
            self.path = path
            self.meta = json.load(open(os.path.join(path, "meta.json")))
        else:
            self.path = os.path.dirname(meta_json)
            self.meta = json.load(open(meta_json))
        self.filter = filter_fct
        return
    def __iter__(self):
        for target, target_info in self.meta["targets"].items():
            for didx, dataset in enumerate(self.meta["datasets"]):
                meta_this = copy.deepcopy(self.meta)
                meta_this.pop("datasets")
                meta_this["name"] = "{0}:{1}:{2}".format(
                        self.meta["name"], target, dataset)
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
        "plog": "lambda y: -np.log10(y)",
    }
    def __init__(self, ext_xyz=None, meta=None, configs=None):
        self.configs = configs
        if ext_xyz is not None:
            if type(ext_xyz) is str:
                self.configs = read(ext_xyz)
            else:
                self.configs = []
                for xyz in ext_xyz:
                    self.configs.extend(read(xyz))
        self.meta = meta
        self.convert = self.target_converter[self.meta.pop("convert", "")]
        if meta is not None and "target" in meta:
            self.y = eval(self.convert)(
                np.array([ float(s.info[meta["target"]]) \
                    for s in self.configs ]))
        return
    def info(self):
        return "{name:50s}  #configs={size:<5d}  task={task:8s}  metrics={metrics:s}   std={std:1.2e}".format(
            name=self.meta["name"], size=len(self.configs),
            task=self.meta["task"], metrics=",".join(self.meta["metrics"]),
            std=np.std(self.y))
    def __getitem__(self, key):
        if np.issubdtype(type(key), np.integer):
            return self.configs[key]
        elif type(key) in {list, np.ndarray}:
            return Dataset(
                configs=[ self.configs[_] for _ in key ],
                meta=self.meta)
        elif type(key) is str:
            return self.meta[key]
        else: raise TypeError("Invalid type in __getitem__: %s" % type(key))
    def __len__(self):
        return len(self.configs)
    def __str__(self):
        return self.info()
    def __iter__(self):
        return self.configs.__iter__()

class XyDataset(object):
    def __init__(self, X, y, name="?", **kwargs):
        self.X = X
        self.y = y
        self.meta = kwargs
        self.meta["name"] = name
    def info(self):
        return "{name:50s}  #samples={size:<5d}  metrics={metrics:s}   std={std:1.2e}".format(
            name=self.meta["name"], size=len(self),
            metrics=",".join(self.meta["metrics"]),
            std=np.std(self.y))
    def __getitem__(self, key):
        if np.issubdtype(type(key), np.integer):
            return self.X[key]
        elif type(key) in {list, np.ndarray}:
            return XyDataset(
                X=self.X[key],
                y=self.y[key],
                meta=self.meta)
        elif type(key) is str:
            return self.meta[key]
        else: raise TypeError("Invalid type in __getitem__: %s" % type(key))
    def __len__(self):
        return len(self.X)
    def __str__(self):
        return self.info()
    def __contains__(self, key):
        return key in self.meta

def compile(root="./data", filter_fct=lambda meta: True):
    return BenchmarkData(root, filter_fct=filter_fct)

if __name__ == "__main__":
    bench = compile()
    for data in bench:
        print(data)
