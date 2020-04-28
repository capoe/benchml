import numpy as np
import copy
import json
from .logger import log

def dict_lookup_path(dictionary, path):
    path = path.split('/')
    v = dictionary
    for p in path:
        v = v[p]
    return v

def dict_set_path(dictionary, path, value):
    path = path.split('/')
    v = dictionary
    for p in path[:-1]:
        v = v[p]
    v[path[-1]] = value
    return

def dict_compile(options, fields, mode):
    if mode == "combinatorial":
        return dict_compile_combinatorial(options, fields)
    elif mode == "linear":
        return dict_compile_linear(options, fields)
    else: raise NotImplementedError(mode)

def dict_compile_combinatorial(options, fields):
    options_array = [ options ]
    for scan in fields:
        path = scan["path"]
        values = scan["vals"]
        options_array_out = []
        for options in options_array:
            for v in values:
                options_mod = copy.deepcopy(options)
                target = dict_set_path(options_mod, path, v)
                options_array_out.append(options_mod)
        options_array = options_array_out
    return options_array

def dict_compile_linear(options, fields):
    options_array = []
    n_combos = len(fields[0][1])
    for i in range(n_combos):
        options_mod = copy.deepcopy(options)
        for field in fields:
            dict_set_path(options_mod, field["path"], field["vals"][i])
        options_array.append(options_mod)
    return options_array

