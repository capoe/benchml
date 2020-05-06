from .accumulator import Accumulator
from .logger import log
from .splits import Split
import copy
import hashlib
import json
import itertools
VERBOSE = False

def generate_hash_id(data):
    data_md5 = hashlib.md5(
        json.dumps(data, sort_keys=True).encode('utf-8'))
    return data_md5.hexdigest()

def deps_from_inputs(inps):
    # { x1: a.x, x2: [ b.y, c.z ], ... } -> { a, b, c, ... }
    return set([ dep for path, inp in inps.items() \
        for dep in deps_from_input(inp) ])

def deps_from_input(inp):
    # Case 1:   a.x -> [ a ]
    # Case 2: [ a.x, b.y, ... ] -> [ a, b, ... ]
    if type(inp) is str:
        yield inp.split(".")[0]
    elif type(inp) is list:
        for item in inp:
            yield item.split(".")[0]

class Params(object):
    def __init__(self, tag, tf):
        self.tag = tag
        self.tf = tf
        self.storage = { "version": None }
    def version(self, hash):
        self.storage["version"] = hash
    def put(self, key, value, force=False):
        if not force and not key in self.tf.allow_params:
            raise ValueError("Param '%s' not allowed in transform '%s'" % (
                key, self.tf.tag))
        self.storage[key] = value
    def get(self, key):
        return self.storage[key]
    def has(self, key):
        return key in self.storage
    def items(self):
        return self.storage.items()

class Stream(object):
    def __init__(self, tag, tf,
            data=None,
            parent=None,
            slice=None,
            slice_ax2=None):
        self.tag = tag
        self.tf = tf
        self.storage = { "version": None }
        self.data = data
        self.split_iterator = None
        self.parent = parent
        self.slice = slice
        self.slice_ax2 = slice_ax2 if slice_ax2 is not None else self.slice
        if self.parent is not None:
            assert self.slice is not None
            self.sliceStorage()
    def __len__(self):
        return len(self.data) if (self.data is not None) else len(self.slice)
    def split(self, **kwargs):
        self.split_iterator = Split(self, **kwargs)
        for info, train, test in self.split_iterator:
            s1 = self.tf.module.openStream(
                tag=self.tag+".train",
                data=None,
                parent_tag=self.tag,
                slice=train,
                slice_ax2=None)
            s2 = self.tf.module.openStream(
                tag=self.tag+".test",
                data=None,
                parent_tag=self.tag,
                slice=test,
                slice_ax2=train)
            yield s1, s2
    def version(self, hash):
        self.storage["version"] = hash
    def put(self, key, value, force=False):
        if not force and not key in self.tf.allow_stream:
            raise ValueError("Stream '%s' not allowed in transform '%s'" % (
                key, self.tf.tag))
        self.storage[key] = value
    def get(self, key):
        return self.storage[key]
    def resolve(self, addr):
        tf, field = addr.split(".")
        return self.tf.module[tf].stream(self.tag).get(field)
    def sliceStorage(self, verbose=VERBOSE):
        if self.parent.has("version") and self.parent.get("version") != None:
            self.put("version", self.parent.get("version"), force=True)
            for key in self.parent.tf.stream_copy:
                if VERBOSE: print("    COPY  ", self.tf.tag, self.tag,
                    '<-', self.parent.tag, key)
                self.put(key, self.parent.get(key))
            for key in self.parent.tf.stream_samples:
                if VERBOSE: print("    SLICE ", self.tf.tag, self.tag,
                    '<-', self.parent.tag, key)
                self.put(key, self.parent.get(key)[self.slice])
            for key in self.parent.tf.stream_kernel:
                if VERBOSE: print("    KERN  ", self.tf.tag, self.tag,
                    '<-', self.parent.tag, key)
                self.put(key, self.parent.get(key)[self.slice][:,self.slice_ax2])
    def has(self, key):
        return key in self.storage
    def items(self):
        return self.storage.items()

class Macro(object):
    req_inputs = tuple()
    req_args = tuple()
    transforms = []
    is_macro = True
    def __init__(self, tag=None, args={}, inputs={}, **kwargs):
        self.tag = tag if tag is not None else self.__class__.__name__
        self.args = args
        self.inputs = inputs
        self.transforms = copy.deepcopy(self.__class__.transforms)
        for req in self.req_inputs:
            if not req in self.inputs:
                raise KeyError("Macro '%s' requires input '%s'" % (
                    self.__class__.__name__, req))
        for req in self.req_args:
            if not req in self.args:
                raise KeyError("Macro '%s' requires arg '%s'" % (
                    self.__class__.__name__, req))
    def __iter__(self):
        for tf_args in self.transforms:
            tf_class = tf_args.pop("class")
            tf = tf_class(**tf_args)
            # Link inputs to parent module
            for key, addr in tf.inputs.items():
                tf.inputs[key] = self.inputs.pop(
                    ".".join([tf.tag, key]),
                    "/".join([self.tag, tf.inputs[key]]))
            # Update args
            for key in tf.args:
                tf.args[key] = self.args.pop(
                    ".".join([tf.tag, key]),
                    tf.args[key])
            tf.tag = self.tag+"/"+tf.tag
            yield tf

class Transform(object):
    default_args = {}
    req_args = tuple()
    req_inputs = tuple()
    precompute = False
    allow_stream = {}
    allow_params = {}
    stream_copy = tuple()
    stream_samples = tuple()
    stream_kernel = tuple()
    def check_available():
        return True
    def __init__(self, **kwargs):
        self.tag = kwargs.pop("tag", self.__class__.__name__)
        self.module = None
        self.is_setup = False
        # Default args, inputs, outputs
        self.args = copy.deepcopy(self.default_args)
        self.args.update(kwargs.pop("args", {}))
        self.args_links = { key: link for key, link in self.args.items() \
            if type(link) is str and link.startswith('@') }
        self.inputs = kwargs["inputs"] if "inputs" in kwargs else {}
        self.outputs = kwargs["outputs"] if "outputs" in kwargs else {}
        self.checkRequire()
        # Streams
        self.map_streams = {}
        self.cached_streams = {}
        self.active_stream = None
        # Param sets
        self.map_params = {}
        self.cached_params = {}
        self.active_params = None
        # Dependency and dependency hash
        self.deps = None
        self.hash_self = None
        self.hash_deps = None
        self.hash_total = None
        self.hash_prev = None
    def attach(self, module):
        self.module = module
    def resolveArgs(self):
        # Read "@tf_tag.field_name" -> module["tf_tag"].args["field_name"]
        for key, val in self.args_links.items():
            if type(val) is str and val.startswith('@'):
                tf, field = val[1:].split(".")
                self.args[key] = self.module[tf].args[field]
    # HASH VERSIONING
    def getHash(self):
        return self.hash_total
    def hashState(self):
        self.hash_prev = self.hash_total
        self.hash_self = generate_hash_id(self.args)
        self.hash_deps = "".join(
            [ self.module[d].getHash() for d in sorted(list(self.deps)) ])
        self.hash_total = generate_hash_id(self.hash_self+self.hash_deps)
    def hashChanged(self):
        return self.hash_prev != self.hash_total
    # STREAM
    def clearStreams(self):
        self.map_streams.clear()
        self.active_stream = None
    def openStream(self, data, tag, parent_tag,
            slice, slice_ax2, verbose=VERBOSE):
        if VERBOSE:
            print("  trafo=%-15s: create stream=%-15s  [parent:%s]" % (
                self.tag, tag, parent_tag))
        parent = self.map_streams[parent_tag] if parent_tag is not None else None
        stream = Stream(tag=tag, data=data, parent=parent,
            slice=slice, slice_ax2=slice_ax2, tf=self)
        self.map_streams[tag] = stream
        self.active_stream = stream
    def activateStream(self, stream_tag):
        self.active_stream = self.map_streams[stream_tag]
    def stream(self, tag=None):
        if tag is None: return self.active_stream
        else: return self.map_streams[tag]
    # PARAMS
    def clearParams(self, keep_active=True):
        self.map_params.clear()
        if keep_active and self.active_params is not None:
            self.map_params[self.active_params.tag] = self.active_params
        else:
            self.active_params = None
    def openParams(self, params_tag):
        params = Params(tag=params_tag, tf=self)
        self.map_params[params_tag] = params
        self.active_params = params
    def activateParams(self, params_tag):
        self.active_params = self.map_params[params_tag]
    def params(self):
        return self.active_params
    # DEPENDENCIES
    def clearDependencies(self):
        self.deps = None
    def updateDependencies(self):
        if self.deps  is not None: return self.deps
        deps = deps_from_inputs(self.inputs)
        if self.tag in deps: deps.remove(self.tag)
        deps_parents = set()
        for dep in deps:
            deps_parents = deps_parents.union(self.module[dep].updateDependencies())
        deps = deps.union(deps_parents)
        if self.tag in deps: deps.remove(self.tag)
        self.deps = deps
        return deps
    # RESOLVE & CHECK ARGS & INPUTS
    def resolveInputs(self):
        res = {}
        for key, addr in self.inputs.items():
            if type(addr) is str:
                tf, k = tuple(addr.split("."))
                if k.startswith("_"):
                    res[key] = self.module.map_transforms[tf].params().get(k[1:])
                else:
                    res[key] = self.module.map_transforms[tf].stream().get(k)
            elif type(addr) is list:
                res[key] = list(map(
                    lambda tf_k: self.module.map_transforms[tf_k[0]].stream().get(tf_k[1]),
                        map(lambda item: tuple(item.split(".")), addr)
                ))
        return res
    def checkRequire(self):
        self.requireArgs(*self.__class__.req_args)
        self.requireInputs(*self.__class__.req_inputs)
    def requireArgs(self, *args):
        for arg in args:
            if not arg in self.args:
                raise KeyError("Missing argument: <%s> requires '%s'" % (
                    self.__class__.__name__, arg))
    def requireInputs(self, *inputs):
        for inp in inputs:
            if not inp in self.inputs:
                raise KeyError("Missing input: <%s> requires '%s'" % (
                    self.__class__.__name__, inp))
    # EXECUTION
    def feed(self, data):
        self.hashState()
        self._feed(data)
        self.stream().version(self.getHash())
    def fit(self, stream_tag, verbose=VERBOSE):
        self.activateStream(stream_tag)
        self.hashState()
        inputs = self.resolveInputs()
        if self.precompute and self.stream().get("version") == self.getHash():
            if verbose: log << "[ hash matches, use cache ]" << log.flush
        else:
            self.openParams(stream_tag)
            self.setup()
            self._fit(inputs)
            self.params().version(self.getHash())
            self.stream().version(self.getHash())
    def map(self, stream_tag, verbose=VERBOSE):
        self.activateStream(stream_tag)
        self.hashState()
        inputs = self.resolveInputs()
        if self.precompute and self.stream().get("version") == self.getHash():
            if verbose: log << "[ hash matches, use cache ]" << log.flush
        else:
            self.setup()
            self._map(inputs)
            self.stream().version(self.getHash())
    def _map(self, inputs):
        return
    def setup(self):
        if self.is_setup: return
        self.resolveArgs()
        self._setup()
        self.is_setup = True
    def _setup(self):
        return
    # LOGGING
    def __str__(self):
        info = "%-15s <- %s" % (self.tag, str(self.inputs))
        info += "\n    State:    " + str(self.getHash())
        info += "\n    Precomp.: " + str(self.precompute)
        info += "\n    Depends:"
        if self.deps is not None:
            for dep in self.deps:
                info += " '%s'" % dep
        if len(self.map_streams):
            info += "\n    Streams:"
            for stream_tag, stream in self.map_streams.items():
                info += " '%s'" % stream_tag
        if len(self.map_params):
            info += "\n    Params: "
            for params_tag, params in self.map_params.items():
                info += " '%s'" % params_tag
        info += "\n    Active:"
        if self.stream() is not None:
            info += "  '%s'" % self.stream().tag
            info += "\n      Storage: ["
            for key, val in self.stream().items():
                info += " '%s'" % key
            info += "]"
        if self.params() is not None:
            info += "\n      Params:  ["
            for key, val in self.params().items():
                info += " '%s'" % key
            info += "]"
        return info

class Module(Transform):
    def __init__(self,
            tag="module",
            broadcast={},
            transforms=[],
            hyper=[],
            **kwargs):
        Transform.__init__(self, tag=tag, **kwargs)
        self.broadcast = broadcast
        self.transforms = []
        self.map_transforms = {}
        self.hyper = hyper
        for t in transforms:
            if hasattr(t, "is_macro"):
                for sub in t: self.append(sub)
            else: self.append(t)
        self.updateDependencies()
    # Transforms
    def __getitem__(self, tag):
        return self.map_transforms[tag]
    def append(self, transform):
        transform.inputs.update(self.broadcast)
        self.transforms.append(transform)
        if transform.tag in self.map_transforms:
            raise ValueError("Transformation with name '%s' already exists" % (
                transform.tag))
        self.map_transforms[transform.tag] = transform
        transform.attach(self)
        return self
    # Dependencies
    def clearDependencies(self):
        for tf in self.transforms: tf.clearDependencies()
    def updateDependencies(self):
        self.clearDependencies()
        for tf in self.transforms: tf.updateDependencies()
    # Params
    def clearParams(self, keep_active=True):
        for tf in self.transforms:
            tf.clearParams(keep_active=keep_active)
    # Streams
    def open(self, data=None, **kwargs):
        return self.openStream(data=data, **kwargs)
    def close(self, check=True):
        self.clearStreams()
        param_streams = { tf.params().tag for tf in self.transforms \
            if tf.params() is not None }
        if check and len(param_streams) > 1:
            log << log.mr << "WARNING Model parametrized using more " \
             "than one stream (did you perhaps use .precompute?)" << log.endl
        self.clearParams(keep_active=True)
    def clearStreams(self):
        for tf in self.transforms:
            tf.clearStreams()
        super().clearStreams()
    def activateStream(self, stream_tag):
        self.active_stream = self.map_streams[stream_tag]
        for tf in self.transforms:
            tf.activateStream(stream_tag)
    def openStream(self,
            data=None,
            tag=None,
            parent_tag=None,
            slice=None,
            slice_ax2=None,
            verbose=VERBOSE):
        if tag is None: tag = "S%d" % (len(self.map_streams))
        if verbose: print("Open stream '%s'" % tag)
        super().openStream(data=data, tag=tag, parent_tag=parent_tag,
            slice=slice, slice_ax2=slice_ax2, verbose=verbose)
        for t in self.transforms:
            t.openStream(data=data, tag=tag, parent_tag=parent_tag,
                slice=slice, slice_ax2=slice_ax2, verbose=verbose)
            if data is not None and hasattr(t, "_feed"): t.feed(data)
        return self.inputStream()
    def inputStream(self):
        return self.transforms[0].stream()
    def get(self, addr):
        tf, field = addr.split(".")
        if field.startswith("_"):
            return self[tf].params().get(field[1:])
        else:
            return self[tf].stream().get(field)
    def resolveArgs(self):
        for tf in self.transforms: tf.resolveArgs()
    def resolveOutputs(self):
        res = {}
        for key, addr in self.outputs.items():
            tf, k = tuple(addr.split("."))
            res[key] = self.map_transforms[tf].stream().get(k)
        return res
    # Hyperfit
    def hyperUpdate(self, updates, verbose=False):
        for addr, val in updates.items():
            tf_tag, arg_name = addr.split(".")
            if verbose:
                print("    Setting {0:15s}.{1:10s} = {2}".format(
                    tf_tag, arg_name, val))
            self[tf_tag].args[arg_name] = val
    def hyperEval(self, 
            stream, 
            updates, 
            split_args,
            accu_args, 
            target,
            target_ref,
            verbose=VERBOSE):
        self.hyperUpdate(updates, verbose=verbose)
        if verbose:
            log << "    Hash changed:" << log.flush
            for tf in self.transforms:
                if tf.hashChanged():
                    log << tf.tag << log.flush
            log << log.endl
        self.precompute(stream)
        accu = Accumulator(**accu_args)
        for substream_train, substream_test in stream.split(**split_args):
            self.fit(substream_train)
            out = self.map(substream_test)
            accu.append("test", out[target], self.get(target_ref))
        metric, metric_std = accu.evaluate("test")
        return metric
    def hyperfit(self, stream, log=None, **kwargs):
        if self.hyper is None:
            raise ValueError("<Module.hyperfit>: Hyper configuration is missing")
        if log: log << "Hyperfit on stream" << stream.tag << log.endl
        updates = self.hyper.optimize(self, stream, log=log, **kwargs)
        self.hyperUpdate(updates)
        return self.fit(stream)
    # Fit, map, precompute
    def fit(self, stream, endpoint=None, verbose=VERBOSE):
        if verbose: print("Fit '%s'" % stream.tag)
        self.activateStream(stream.tag)
        if endpoint is None:
            sweep = self.transforms
        else: 
            sweep = list(filter(
                lambda tf: tf.tag in self[endpoint].deps, self.transforms))
            sweep.append(self[endpoint])
        for tidx, t in enumerate(sweep):
            if hasattr(t, "_fit"):
                if verbose: print(" "*tidx, "Fit", t.tag, "using stream", stream.tag)
                t.fit(stream.tag, verbose=verbose)
            else:
                if verbose: log << " ".join([" "*tidx, "Map", t.tag,
                    "using stream", stream.tag]) << log.flush
                t.map(stream.tag, verbose=verbose)
                if verbose: log << log.endl
        return
    def map(self, stream, verbose=VERBOSE):
        if verbose: print("Map '%s'" % stream.tag)
        self.activateStream(stream.tag)
        for tidx, t in enumerate(self.transforms):
            if verbose: log << " ".join([" "*tidx, "Map", t.tag,
                "using stream", stream.tag]) << log.flush
            t.map(stream.tag, verbose=verbose)
            if verbose: log << log.endl
        return self.resolveOutputs()
    def precompute(self, stream, verbose=VERBOSE):
        precomps = list(filter(lambda tf: tf.precompute, self.transforms))
        precomps_deps = set(
            [ p.tag for p in precomps ] + \
            [ d for p in precomps for d in p.deps ])
        if verbose: print("Precompute '%s'" % stream.tag)
        self.activateStream(stream.tag)
        for tidx, tf in enumerate(filter(
                lambda tf: tf.tag in precomps_deps, self.transforms)):
            if hasattr(tf, "_fit"):
                if verbose: log << " ".join([" "*tidx, "Fit (precompute)",
                    tf.tag, "using stream", stream.tag]) << log.flush
                tf.fit(stream.tag, verbose=verbose)
            else:
                if verbose: log << " ".join([" "*tidx, "Map (precompute)",
                    tf.tag, "using stream", stream.tag]) << log.flush
                tf.map(stream.tag, verbose=verbose)
                if verbose: log << log.endl
        return self.inputStream()
    def __str__(self):
        return "Module='%s'" % self.tag + \
            "\n  "+"\n  ".join([ str(t) for t in self.transforms ])

class sopen(object):
    def __init__(self, module, data):
        self.module = module
        self.data = data
    def __enter__(self):
        return self.module.open(self.data)
    def __exit__(self, *args):
        self.module.close()

