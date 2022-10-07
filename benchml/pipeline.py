import abc
import copy
import enum
import hashlib
import inspect
import json
import time

import numpy as np

from benchml.accumulator import Accumulator
from benchml.logger import log
from benchml.splits import Split

VERBOSE = False


def force_json(data):
    if isinstance(data, np.ndarray):
        return str(data)
    else:
        return data


def generate_hash_id(data):
    data_md5 = hashlib.md5(json.dumps(data, sort_keys=True, default=force_json).encode("utf-8"))
    return data_md5.hexdigest()


def deps_from_inputs(inps):
    # { x1: a.x, x2: [ b.y, c.z ], ... } -> { a, b, c, ... }
    return set([dep for path, inp in inps.items() for dep in deps_from_input(inp)])


def deps_from_input(inp):
    # Case 1:   a.x -> [ a ]
    # Case 2: [ a.x, b.y, ... ] -> [ a, b, ... ]
    if type(inp) is str:
        yield inp.split(".")[0]
    elif type(inp) is list:
        for item in inp:
            yield item.split(".")[0]


def log_stage_info_message(cur_log, tidx, stage, transform, stream):
    msg_data = [" " * tidx, stage, transform.tag, "using stream", stream.tag]
    msg = " ".join(msg_data)
    cur_log << msg << cur_log.flush


class StreamHandle(object):
    def __init__(self, module):
        self.module = module
        self.map_stream = {}
        self.map_active_stream = {}
        self.stream_tags = set()

    def __len__(self):
        return len(self.stream_tags)

    def clear(self):
        self.map_stream.clear()
        self.map_active_stream.clear()
        self.stream_tags.clear()

    def hasPartition(self, tf):
        return tf.tag in self.map_stream

    def createPartition(self, tag, *args, **kwargs):
        self.stream_tags.add(tag)
        return Stream(self, *args, tag=tag, **kwargs)

    def getStream(self, tf, stream_tag):
        if type(tf) is str:
            return self.map_stream[tf][stream_tag]
        else:
            return self.map_stream[tf.tag][stream_tag]

    def setStreamFor(self, tf, stream, set_active=True):
        if not self.hasPartition(tf):
            self.map_stream[tf.tag] = {}
        self.map_stream[tf.tag][stream.tag] = stream
        if set_active:
            self.setActive(tf, stream)

    def getActive(self, tf):
        return self.map_active_stream[tf.tag]

    def setActive(self, tf, stream):
        self.map_active_stream[tf.tag] = stream

    def activate(self, tf, stream_tag):
        active = self.map_stream[tf.tag][stream_tag]
        self.map_active_stream[tf.tag] = active
        return active

    def info(self, tag=None):
        tags = list(self.stream_tags) if tag is None else [tag]
        for tag in tags:
            print("Stream tag:", tag)
            for tf, stream in self.map_stream.items():
                stream[tag].info()


class Stream(object):
    def __init__(self, handle, tag, tf, data=None, parent=None, slice=None, slice_ax2=None):
        self.handle = handle
        self.tag = tag
        self.tf = tf
        self.storage = {"version": None}
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

    def __getitem__(self, addr):
        return self.resolve(addr)

    def version(self, hash):
        self.storage["version"] = hash

    def has(self, key):
        return key in self.storage

    def get(self, key):
        try:
            return self.storage[key]
        except KeyError:
            raise KeyError("No such field '%s' in tf '%s'" % (key, self.tf.tag))

    def items(self):
        return self.storage.items()

    def keys(self):
        return self.storage.keys()

    def put(self, key, value, force=False):
        if not force and key not in self.tf.allow_stream:
            raise ValueError("Stream '%s' not allowed in transform '%s'" % (key, self.tf.tag))
        self.storage[key] = value

    def resolve(self, addr):
        tf, field = addr.split(".")
        return self.handle.getStream(tf, self.tag).get(field)

    def select(self, tf):
        return self.handle.getStream(tf, self.tag)

    def info(self):
        print("  ", self.tf.tag)
        for key, val in self.storage.items():
            print("      %-20s = %10s" % (key, str(type(val))))

    def split(self, **kwargs):
        self.split_iterator = Split(self, **kwargs)
        for info, train, test in self.split_iterator:
            s1 = self.tf.module.openStream(
                handle=self.handle,
                tag=self.tag + ".train",
                data=None,
                parent_tag=self.tag,
                slice=train,
                slice_ax2=None,
            )
            s2 = self.tf.module.openStream(
                handle=self.handle,
                tag=self.tag + ".test",
                data=None,
                parent_tag=self.tag,
                slice=test,
                slice_ax2=train,
            )
            yield s1, s2

    def sliceStorage(self, verbose=VERBOSE):
        if self.parent.has("version") and self.parent.get("version") is not None:
            self.put("version", self.parent.get("version"), force=True)
            for key in self.parent.tf.stream_copy:
                if not self.parent.has(key):
                    continue
                if VERBOSE:
                    print("    COPY  ", self.tf.tag, self.tag, "<-", self.parent.tag, key)
                self.put(key, self.parent.get(key))
            for key in self.parent.tf.stream_samples:
                if not self.parent.has(key):
                    continue
                if VERBOSE:
                    print("    SLICE ", self.tf.tag, self.tag, "<-", self.parent.tag, key)
                self.put(key, self.parent.get(key)[self.slice])
            for key in self.parent.tf.stream_kernel:
                if not self.parent.has(key):
                    continue
                if VERBOSE:
                    print("    KERN  ", self.tf.tag, self.tag, "<-", self.parent.tag, key)
                self.put(key, self.parent.get(key)[self.slice][:, self.slice_ax2])
            for key in self.parent.tf.stream_self_kernel:
                if not self.parent.has(key):
                    continue
                if VERBOSE:
                    print("    SELF  ", self.tf.tag, self.tag, "<-", self.parent.tag, key)
                self.put(key, self.parent.get(key)[self.slice][:, self.slice])


class Params(object):
    def __init__(self, tag, tf):
        self.tag = tag
        self.tf = tf
        self.storage = {"version": None}

    def version(self, hash):
        self.storage["version"] = hash

    def has(self, key):
        return key in self.storage

    def get(self, key):
        return self.storage[key]

    def items(self):
        return self.storage.items()

    def keys(self):
        return list(sorted(self.storage.keys()))

    def clone(self):
        return copy.deepcopy(self)

    def put(self, key, value, force=False):
        if not force and key not in self.tf.allow_params:
            raise ValueError("Param '%s' not allowed in transform '%s'" % (key, self.tf.tag))
        self.storage[key] = value


class TransformBase(object):
    """Transforms constitute the nodes of a pipeline. Each transforms
    implements at least one of three methods, through which it acts on the data
    stream: map, fit, and feed.

    Args:
        tag (str, optional): Address of the transform via which its outputs can be referenced.
            Default: self.__class__.__name__
        args (dict, optional): Dictionary of transform parameters.
            Default: {}
        inputs (dict, optional): Dictionary specifying input streams
            Default: {}
    """

    default_args = {}
    req_args = tuple()
    req_inputs = tuple()
    precompute = False
    allow_stream = {}
    allow_params = {}
    stream_copy = tuple()
    stream_samples = tuple()
    stream_kernel = tuple()
    stream_self_kernel = tuple()
    help_args = {}
    help_inputs = {}
    help_stream = {}
    help_params = {}

    def check_available(self, *args, **kwargs):
        return True

    def __init__(self, **kwargs):
        self.tag = kwargs.pop("tag", self.__class__.__name__)
        self.module = None
        self._is_setup = False
        self._freeze = False
        self._deployed = False
        if "precompute" in kwargs:
            self.precompute = kwargs["precompute"]
        # Default args, inputs, outputs
        self.args = copy.deepcopy(self.default_args)
        self.args.update(kwargs.pop("args", {}))
        self.args_links = self.parseArgsLinks()
        self.inputs = kwargs["inputs"] if "inputs" in kwargs else {}
        self.outputs = kwargs["outputs"] if "outputs" in kwargs else {}
        # Detached?
        detached = kwargs["detached"] if "detached" in kwargs else False
        self.detached = True if self.inputs is None else detached
        # Param sets
        self.map_params = {}
        self.active_params = None
        # Dependency and dependency hash
        self.deps = None
        self.hash_self = None
        self.hash_self_prev = None
        self.hash_deps = None
        self.hash_total = None
        self.hash_prev = None

    def detach(self, **args):
        return Standalone(self.__class__, **args)

    def attach(self, module):
        self.checkRequire()
        self.module = module

    def ready(self):
        return self._is_setup

    def freeze(self, freeze=True):
        self._freeze = True

    def frozen(self):
        return self._freeze

    def deploy(self, set_deployed=True):
        if set_deployed:
            self.resolveArgs()
            self.hashState()
            self._deployed = True
        else:
            self._deployed = False

    # PARAMS & STREAMS
    def params(self):
        return self.active_params

    def openParams(self, params_tag):
        params = Params(tag=params_tag, tf=self)
        self.map_params[params_tag] = params
        self.active_params = params
        return params

    def activateParams(self, params_tag):
        self.active_params = self.map_params[params_tag]

    def clearParams(self, keep_active=True):
        self.map_params.clear()
        if keep_active and self.active_params is not None:
            self.map_params[self.active_params.tag] = self.active_params
        else:
            self.active_params = None

    def openStream(self, handle, data, tag, parent_tag, slice, slice_ax2, verbose=VERBOSE):
        if VERBOSE:
            print("  trafo=%-15s: create stream=%-15s  [parent:%s]" % (self.tag, tag, parent_tag))
        parent = handle.getStream(self, parent_tag) if parent_tag is not None else None
        stream = handle.createPartition(
            tag=tag, data=data, parent=parent, slice=slice, slice_ax2=slice_ax2, tf=self
        )
        handle.setStreamFor(self, stream, set_active=True)
        return stream

    # DEPENDENCIES
    def clearDependencies(self):
        self.deps = None

    def updateDependencies(self):
        if self.deps is not None:
            return self.deps
        if self.inputs is None:
            self.deps = {}
            return self.deps
        deps = deps_from_inputs(self.inputs)
        if self.tag in deps:
            deps.remove(self.tag)
        deps_parents = set()
        for dep in deps:
            deps_parents = deps_parents.union(self.module[dep].updateDependencies())
        deps = deps.union(deps_parents)
        if self.tag in deps:
            deps.remove(self.tag)
        self.deps = deps
        return deps

    def updateInputs(self, ext):
        if self.inputs is None:
            return
        self.inputs.update(ext)

    def getHash(self):
        return self.hash_total

    def hashChanged(self):
        return self.hash_prev != self.hash_total

    def hashSelfChanged(self):
        return self.hash_self_prev != self.hash_self

    def hashState(self):
        if self._deployed:
            return
        self.hash_prev = self.hash_total
        self.hash_self_prev = self.hash_self
        self.hash_self = generate_hash_id(self.args)
        self.hash_deps = "".join([self.module[d].getHash() for d in sorted(list(self.deps))])
        self.hash_total = generate_hash_id(self.hash_self + self.hash_deps)

    # RESOLVE & CHECK ARGS & INPUTS
    def checkRequire(self):
        self.requireArgs(*self.__class__.req_args)
        self.requireInputs(*self.__class__.req_inputs)

    def requireArgs(self, *args):
        for arg in args:
            if arg not in self.args:
                raise KeyError(
                    "Missing argument: <%s> requires '%s'" % (self.__class__.__name__, arg)
                )

    def requireInputs(self, *inputs):
        if self.detached:
            return
        for inp in inputs:
            if inp not in self.inputs:
                raise KeyError("Missing input: <%s> requires '%s'" % (self.__class__.__name__, inp))

    def parseArgsLinks(self):
        args_links = {
            key: link
            for key, link in self.args.items()
            if type(link) is str and link.startswith("@")
        }
        for key, val in self.args.items():
            if (
                type(val) is list
                and len(val) > 0
                and type(val[0]) is str
                and val[0].startswith("@")
            ):
                args_links[key] = val
        self.args_links = args_links
        return args_links

    def resolveArgs(self):
        if self._deployed:
            return
        # Read "@tf_tag.field_name" -> module["tf_tag"].args["field_name"]
        for key, val in self.args_links.items():
            if type(val) is str and val.startswith("@"):
                tf, field = val[1:].split(".")
                self.args[key] = self.module[tf].args[field]
            elif type(val) is list:
                self.args[key] = [
                    self.module[v[0]].args[v[1]] for v in map(lambda v: v[1:].split("."), val)
                ]

    def resolveInputs(self, stream):
        res = {}
        for key, addr in self.inputs.items():
            if type(addr) is str:
                tf_k = addr.strip().split(".")
                if len(tf_k) > 1:
                    tf, k = tuple(tf_k)
                else:
                    tf, k = tf_k[0], None
                if k is None:
                    res[key] = self.module.map_transforms[tf]
                elif k.startswith("_"):
                    res[key] = self.module.map_transforms[tf].params().get(k[1:])
                else:
                    res[key] = stream.resolve(addr)
            elif type(addr) is list:
                res[key] = [stream.resolve(item) for item in addr]
            else:
                res[key] = addr
        return res

    # SETUP, FEED, MAP, FIT
    def setup(self):
        if not self.hashSelfChanged() and self._is_setup:
            return
        if VERBOSE and self._is_setup and self.hashSelfChanged():
            log << "[%s#H]" % self.tag << log.flush
        self._setup()
        self._is_setup = True

    def _setup(self):
        return

    # LOGGING
    def __str__(self):
        info = "%s <- %s" % (self.tag, str(self.inputs))
        info += "\n    State:    " + str(self.getHash())
        info += "\n    Precomp.: " + str(self.precompute)
        info += "\n    Depends:"
        if self.deps is not None:
            for dep in self.deps:
                info += " '%s'" % dep
        if len(self.map_params):
            info += "\n    Params: "
            for params_tag, params in self.map_params.items():
                info += " '%s'" % params_tag
        if self.params() is not None:
            info += "\n      Params:  ["
            for key, val in self.params().items():
                info += " '%s'" % key
            info += "]"
        return info

    def showHelpMessage(self):
        log << log.mb << " Node %-30s [tag='%s']" % (self.__class__.__name__, self.tag) << log.endl

        def truncate_path(f):
            p = f.split("/")
            i = p.index("benchml")
            return "/".join(p[i:])

        log << "  Implemented in %s" % truncate_path(inspect.getfile(self.__class__)) << log.endl
        for arg in self.args:
            req = arg in self.req_args
            info = (
                self.help_args[arg]
                if arg in self.help_args
                else [
                    type(self.args[arg]).__name__ if arg in self.args else "?",  # type info
                    "",  # help mssg
                    [],
                ]
            )  # allowed vals
            val = self.args[arg] if arg in self.args else "?"
            if type(info[2]) is str and info[2].startswith("lambda"):
                allow = eval(info[2])(self)
            else:
                allow = info[2]
            (
                log
                << "  Arg %-30s  val=%-10s type=%-10s req=%s %s"
                % (
                    "'%s'" % arg,
                    str(self.args[arg]) if type(val) is not list else "[...]",
                    str(info[0]),
                    str(req),
                    "help=%s" % info[1] if info[1] != "" else "",
                )
                << log.endl
            )
            if type(val) is list:
                log << "   val=[ " << log.flush
                for idx, v in enumerate(val):
                    log << "%s" % str(v) << log.flush
                    if (idx + 1) != len(val) and (idx + 1) % 5 == 0:
                        log << log.endl
                log << " ]" << log.endl
            if type(allow) is list and len(allow) > 0:
                log << log.my << "   allow=[" << log.endl
                for idx, a in enumerate(allow):
                    log << log.my << "    %-15s" % str(a) << log.flush
                    if (idx + 1) != len(allow) and (idx + 1) % 5 == 0:
                        log << log.endl
                log << log.my << " ]" << log.endl


class InputTransform(TransformBase):
    @abc.abstractmethod
    def _feed(self, data, stream):
        """Mandatory method for Input Transforms."""
        return

    def feed(self, stream, data, verbose=VERBOSE):
        self.resolveArgs()
        self.hashState()
        if self.detached:
            stream.version(self.getHash())
            return
        self.setup()
        self._feed(data, stream)
        stream.version(self.getHash())

    def map(self, inputs, stream):
        """A mock method."""
        return


class Transform(TransformBase):
    @abc.abstractmethod
    def _map(self, inputs, stream):
        """Mandatory method for descriptor and trainable Transforms."""
        return

    def map(self, stream, verbose=VERBOSE):
        stream = stream.handle.activate(self, stream.tag)
        self.resolveArgs()
        self.hashState()
        if self.detached:
            stream.version(self.getHash())
            return
        inputs = self.resolveInputs(stream)
        if self.precompute and stream.get("version") == self.getHash():
            if verbose:
                log << "[ hash matches, use cache ]" << log.flush
        else:
            self.setup()
            self._map(inputs, stream)
            self.hashState()
            stream.version(self.getHash())


class FitTransform(Transform):
    @abc.abstractmethod
    def _fit(self, inputs, stream, params):
        """Mandatory method for Trainable/Fittable Transforms."""
        return

    def fit(self, stream, verbose=VERBOSE):
        if self._freeze:
            if verbose:
                log << "[302->Map]" << log.flush
            return self.map(stream, verbose=verbose)
        stream = stream.handle.activate(self, stream.tag)
        self.resolveArgs()
        self.hashState()
        if self.detached:
            stream.version(self.getHash())
            return
        inputs = self.resolveInputs(stream)
        if self.precompute and stream.get("version") == self.getHash():
            if verbose:
                log << "[ hash matches, use cache ]" << log.flush
        else:
            params = self.openParams(stream.tag)
            self.setup()
            self._fit(inputs, stream, params)
            self.hashState()
            self.params().version(self.getHash())
            stream.version(self.getHash())


class ModelCategory(enum.Enum):
    classification = "Classification"
    regression = "Regression"
    na = "'Not Specified'"


class Module(TransformBase):
    """A module encapsulates an ML pipeline. It consists of a sequence of
    interdependent transforms.

    Args:
        tag (str): Module name (optional)
        broadcast (dict): Inputs to be broadcast to all nodes (optional)
        transforms (list): List of BenchML transforms
        hyper (object): Hyper-optimization controller
    """

    categories = ModelCategory
    docstring_template = """{doc_header}
{summary}

Model Category: {model_category.value}

{extended_summary}

{pipeline_graph}

{transforms}
"""
    def_doc = dict(
        summary="'No summary'",
        extended_summary="'No extended summary'",
        model_category=categories.na,
    )

    def __init__(
        self, tag="module", broadcast=None, transforms=None, hyper=None, doc=None, **kwargs
    ):
        super().__init__(tag=tag, **kwargs)
        if broadcast is None:
            broadcast = {}
        if transforms is None:
            transforms = []
        self.broadcast = broadcast
        self.transforms = []
        self.map_transforms = {}
        self.hyper = hyper
        for t in transforms:
            if hasattr(t, "is_macro"):
                for sub in t:
                    self.append(sub)
            else:
                self.append(t)
        self.updateDependencies()
        # Automated documentation ('docstring') section
        self._provided_doc = doc
        if doc is None:
            doc = self.def_doc.copy()
        else:
            doc = {**self.def_doc, **doc}
        self._autodoc = {}
        self._autodoc.update(dict(doc_header=self.make_doc_header()))
        self._autodoc.update(dict(graph=self.create_mermaid_graph(self, self.transforms)))
        self._autodoc.update(
            dict(
                transforms=self.make_transforms_section(),
                pipeline_graph=self.make_pipeline_section(),
            )
        )
        self.__doc__ = self.docstring_template.format(tag=self.tag, **doc, **self._autodoc)

    def make_doc_header(self):
        return "\n".join([self.tag, "=" * len(self.tag)])

    def make_transforms_section(self):
        header = "Transforms\n----------"
        body = "{name} : {type}\n    {description}\n"  # "Info:\n    {info}\n"
        t_docs = [body.format(**self.document_transform(t)) for t in self.transforms]
        return "\n".join([header, *t_docs])

    def make_pipeline_section(self):
        header = "Pipeline Graph\n--------------\n.. mermaid::\n"
        graph = self._autodoc.get("graph") or self.create_mermaid_graph(self, self.transforms)
        return "\n    ".join([header, *graph, "\n"])

    @staticmethod
    def create_mermaid_graph(model, transforms):
        g_header = """flowchart TD"""
        graph = [g_header]
        i = "    "  # Indent, for readability
        en_t = {t.tag: (idx, t) for idx, t in enumerate(transforms)}

        def name(t):
            return type(t).__name__

        def add_line(source_tag):
            source_ind, source_t = en_t[source_tag]
            graph.append(
                (
                    f"{i}{source_ind}"
                    f"[{name(source_t)}: '{source_tag}']"
                    f" -- {input_name} --> "
                    f"{idx}[{name(t)}: '{tag}']"
                )
            )

        graph.append(f"subgraph Model [{model.tag}]")
        for tag, (idx, t) in en_t.items():
            for input_name, input_source in t.inputs.items():
                if input_source is None:
                    # ReduceTypedMatrix has optional 'T' input
                    continue
                if type(input_source) is list:
                    for input_source_1 in input_source:
                        source_tag = input_source_1.split(".")[0]
                        add_line(source_tag)
                else:
                    source_tag = input_source.split(".")[0]
                    add_line(source_tag)
        graph.append("end")
        hyper_name = type(model.hyper).__name__
        graph.append(f"subgraph Hyper [{hyper_name}]")
        for hyper_field in model.hyper.getFields():
            dest_tag, dest_param = hyper_field.split(".")
            dest_ind, dest_t = en_t[dest_tag]
            graph.append(
                (
                    f"{i}HyperParams --- "
                    f"{dest_param} ---> {dest_ind}"
                    f"[{name(dest_t)}: '{dest_tag}']"
                )
            )
        graph.append("end")
        graph.append("subgraph Outputs [Outputs]")
        for idx, (o_name, o_label) in enumerate(model.outputs.items()):
            source_tag, input_name = o_label.split(".")
            source_ind, source_t = en_t[source_tag]
            graph.append(
                (
                    f"{i}{source_ind}"
                    f"[{name(source_t)}: '{source_tag}']"
                    f" -- {input_name} --> "
                    f"{o_name}[{o_name}]"
                )
            )
        graph.append("end")
        return graph

    @staticmethod
    def document_transform(t):
        name = type(t).__name__
        tt = str([c.__name__ for c in t.__class__.__bases__])
        info = str(t)
        if t.__doc__ is not None:
            desc = t.__doc__.split("\n")[0]
        else:
            desc = "'No description'"
        return dict(name=name, type=tt, description=desc, info=info)

    # Status
    def check_available(self, *args, **kwargs):
        return np.array([tf.check_available(*args, **kwargs) for tf in self.transforms]).all()

    def freeze(self, *tfs, freeze=True):
        for tf in tfs:
            self[tf].freeze(freeze=freeze)

    def unfreeze(self, *tfs):
        self.freeze(*tfs, freeze=False)

    def deploy(self, set_deployed=True):
        super().deploy(set_deployed)
        for tf in self.transforms:
            tf.deploy(set_deployed)

    # Transforms
    def __getitem__(self, tag):
        return self.map_transforms[tag]

    def append(self, transform):
        transform.updateInputs(self.broadcast)
        self.transforms.append(transform)
        if transform.tag in self.map_transforms:
            raise ValueError(f"Transformation with name '{transform.tag}' already exists")
        self.map_transforms[transform.tag] = transform
        transform.attach(self)
        return self

    def replace(self, tag, tf):
        tf.attach(self)
        self.map_transforms[tag] = tf
        self.transforms = list(map(lambda t: tf if (t.tag == tag) else t, self.transforms))
        self.updateDependencies()

    def reconnect(self, inputs):
        for destination, target in inputs.items():
            tf, input_field = destination.split(".")
            self[tf].inputs.update({input_field: target})
        self.updateDependencies()

    def hashState(self):
        for tf in self.transforms:
            tf.hashState()

    def updateDependencies(self):
        self.clearDependencies()
        for tf in self.transforms:
            tf.updateDependencies()

    def clearDependencies(self):
        for tf in self.transforms:
            tf.clearDependencies()

    # Params & Streams
    def clearParams(self, keep_active=True):
        for tf in self.transforms:
            tf.clearParams(keep_active=keep_active)

    def open(self, data=None, **kwargs):
        return self.openStream(handle=None, data=data, **kwargs)

    def close(self, stream, clear_stream=True, check=True):
        if clear_stream:
            stream.handle.clear()
        param_streams = {
            tf.params().tag
            for tf in self.transforms
            if (tf.params() is not None and not tf.detached)
        }
        if check and len(param_streams) > 1:
            (
                log << log.mr << "WARNING Model parametrized using more "
                "than one stream (did you perhaps use .precompute?)" << log.endl
            )
        self.clearParams(keep_active=True)

    def openStream(
        self,
        handle=None,
        data=None,
        tag=None,
        parent_tag=None,
        slice=None,
        slice_ax2=None,
        verbose=VERBOSE,
    ):
        if verbose:
            print("Open stream '%s'" % tag)
        if handle is None:
            handle = StreamHandle(module=self)
        if tag is None:
            tag = "S%d" % (len(handle))
        super().openStream(
            handle=handle,
            data=data,
            tag=tag,
            parent_tag=parent_tag,
            slice=slice,
            slice_ax2=slice_ax2,
            verbose=verbose,
        )
        for t in self.transforms:
            stream = t.openStream(
                handle=handle,
                data=data,
                tag=tag,
                parent_tag=parent_tag,
                slice=slice,
                slice_ax2=slice_ax2,
                verbose=verbose,
            )
            if data is not None and isinstance(t, InputTransform):
                if verbose:
                    log << " Feed '%s'" % t.tag << log.flush
                t.feed(stream, data, verbose=verbose)
                if verbose:
                    log << log.endl
        return handle.getActive(self.transforms[0])

    def activateStream(self, stream):
        stream.handle.activate(self, stream.tag)
        for tf in self.transforms:
            stream.handle.activate(tf, stream.tag)

    # Precompute, map, fit
    def precompute(self, stream, verbose=VERBOSE):
        """Precomputes the outputs of those transforms that are declared with
        precompute=True.

        The precomputation (necessarily) includes all upstream nodes in
        the dependency chain.
        """
        precomps = list(filter(lambda tf: tf.precompute, self.transforms))
        precomps_deps = set([p.tag for p in precomps] + [d for p in precomps for d in p.deps])
        if verbose:
            print("Precompute '%s'" % stream.tag)
        self.activateStream(stream)
        for tidx, tf in enumerate(filter(lambda tf: tf.tag in precomps_deps, self.transforms)):
            if isinstance(tf, InputTransform):
                if verbose:
                    log_stage_info_message(log, tidx, "Input (precompute)", tf, stream)
                continue
            do_fit = isinstance(tf, FitTransform)
            method_naming = {True: "Fit", False: "Map"}
            if verbose:
                log_stage_info_message(
                    log, tidx, method_naming[do_fit] + " (precompute)", tf, stream
                )
            t0 = time.time()
            if do_fit:
                tf.fit(stream, verbose=verbose)
            else:
                tf.map(stream, verbose=verbose)
            t1 = time.time()
            if verbose:
                log << "[%1.4f]" % (t1 - t0) << log.flush
                log << log.endl
        return stream

    def map(self, stream, endpoint=None, verbose=VERBOSE):
        if not isinstance(stream, (Stream,)):
            with sopen(self, stream) as stream:
                return self.map(stream)
        if verbose:
            print("Map '%s'" % stream.tag)
        self.activateStream(stream)
        sweep = self.filter(endpoint=endpoint)
        for tidx, t in enumerate(sweep):

            if isinstance(t, InputTransform):
                if verbose:
                    log_stage_info_message(log, tidx, "Input", t, stream)
                continue
            if verbose:
                log_stage_info_message(log, tidx, "Map", t, stream)
            t0 = time.time()
            t.map(stream, verbose=verbose)
            t1 = time.time()
            if verbose:
                log << "[%1.4f]" % (t1 - t0) << log.flush
                log << log.endl
        return self.resolveOutputs(stream)

    def fit(self, stream, endpoint=None, verbose=VERBOSE):
        if verbose:
            print("Fit '%s'" % stream.tag)
        self.activateStream(stream)
        sweep = self.filter(endpoint=endpoint)
        for tidx, t in enumerate(sweep):
            if isinstance(t, InputTransform):
                if verbose:
                    log_stage_info_message(log, tidx, "Input", t, stream)
                continue
            do_fit = isinstance(t, FitTransform)
            method_naming = {True: "Fit", False: "Map"}
            if verbose:
                log_stage_info_message(log, tidx, method_naming[do_fit], t, stream)
            t0 = time.time()
            if do_fit:
                t.fit(stream, verbose=verbose)
            else:
                t.map(stream, verbose=verbose)
            t1 = time.time()
            if verbose:
                log << "[%1.4f]" % (t1 - t0) << log.endl
        return

    def filter(self, endpoint):
        if endpoint is None:
            return self.transforms
        deps = set()
        if type(endpoint) is list:
            deps = deps.union(*([self[e].deps for e in endpoint]))
            deps = deps.union(set(endpoint))
        else:
            deps = deps.union(self[endpoint].deps)
            deps.add(endpoint)
        shortlist = list(filter(lambda tf: tf.tag in deps, self.transforms))
        return shortlist

    def resolveArgs(self):
        for tf in self.transforms:
            tf.resolveArgs()

    def resolveOutputs(self, stream):
        res = {}
        for key, addr in self.outputs.items():
            tf, k = tuple(addr.split("."))
            res[key] = stream.resolve(addr)
        return res

    # Hyperfit
    def hyperfit(self, stream, log=None, verbose=False, **kwargs):
        if self.hyper is None:
            raise ValueError("<Module.hyperfit>: Hyper configuration is missing")
        if log:
            log << "Hyperfit" << self.tag << "on stream" << stream.tag << log.endl
        updates = self.hyper.optimize(self, stream, log=log, **kwargs)
        self.hyperUpdate(updates)
        return self.fit(stream, verbose=verbose)

    def hyperEval(self, stream, updates, split_args, accu_args, target, target_ref, verbose=None):
        if verbose is None:
            verbose = VERBOSE
        self.hyperUpdate(updates, verbose=verbose)
        if verbose:
            log << "    Hash changed:" << log.flush
            for tf in self.transforms:
                if tf.hashChanged():
                    log << tf.tag << log.flush
            log << log.endl
        self.precompute(stream, verbose=verbose)
        accu = Accumulator(**accu_args)
        for substream_train, substream_test in stream.split(**split_args):
            self.fit(substream_train, verbose=verbose)
            out = self.map(substream_test)
            accu.append("test", out[target], substream_test.resolve(target_ref))
        metric, metric_std = accu.evaluate("test")
        return metric

    def hyperUpdate(self, updates, verbose=False, check_existing=False):
        for addr, val in updates.items():
            tf_tag, arg_name = addr.split(".")
            if verbose:
                print("    Setting {0:15s}.{1:10s} = {2}".format(tf_tag, arg_name, val))
            if check_existing and arg_name not in self[tf_tag].args:
                raise KeyError("Non-existing field '%s' in '%s' args" % (arg_name, tf_tag))
            self[tf_tag].args[arg_name] = val

    # Info & logging
    def compileArgs(self):
        args = {}
        for tf in self.transforms:
            for key, v in tf.args.items():
                args["%s.%s" % (tf.tag, key)] = v
        return args

    def compileInputs(self):
        inputs = {}
        for tf in self.transforms:
            for key, v in tf.inputs.items():
                inputs["%s.%s" % (tf.tag, key)] = v
        return inputs

    def compileStream(self, typestr_only=False):
        raise NotImplementedError("Uses old streaming model")

    def __str__(self):
        return "Module='%s'" % self.tag + "\n  " + "\n  ".join([str(t) for t in self.transforms])

    def showHelpMessage(self):
        avail = self.check_available()
        (
            log
            << (log.mg if avail else log.mr)
            << f"Help message for module '{self.tag}'"
            << log.endl
        )
        for tf in self.transforms:
            tf.showHelpMessage()
        if self.hyper is not None:
            (
                log
                << log.pp
                << f" Hyperparameter optimization: {self.hyper.__class__.__name__}"
                << log.endl
            )
            for h in self.hyper.hypers:
                log << log.pp << "  " << h.instr << log.endl
        return


class Macro(object):
    req_inputs = tuple()
    req_args = tuple()
    transforms = []
    is_macro = True

    def __init__(self, tag=None, args=None, inputs=None, **kwargs):
        if args is None:
            args = {}
        if inputs is None:
            inputs = {}
        self.tag = tag if tag is not None else self.__class__.__name__
        self.args = args
        self.inputs = inputs
        self.transforms = copy.deepcopy(self.__class__.transforms)
        for req in self.req_inputs:
            if req not in self.inputs:
                raise KeyError("Macro '%s' requires input '%s'" % (self.__class__.__name__, req))
        for req in self.req_args:
            if req not in self.args:
                raise KeyError("Macro '%s' requires arg '%s'" % (self.__class__.__name__, req))

    def __iter__(self):
        for tf_args in self.transforms:
            tf_class = tf_args.pop("class")
            init = copy.deepcopy(tf_args)
            init_args = init["args"]
            init_tag = init["tag"] if "tag" in init else tf_class.__name__
            # Update args
            for key in init_args:
                init_args[key] = self.args.pop(".".join([init_tag, key]), init_args[key])
            tf = tf_class(**init)
            # Link inputs to parent module
            for key, addr in tf.inputs.items():
                tf.inputs[key] = tf.inputs[key].format(self=self.tag + "/")
                tf.inputs[key] = self.inputs.pop(".".join([tf.tag, key]), tf.inputs[key])
            tf.parseArgsLinks()
            tf.tag = self.tag + "/" + tf.tag
            yield tf


class Standalone(object):
    def __init__(self, TransformClass, **kwargs):
        self.base = TransformClass(detached=True, args=kwargs)
        self.base.setup()

    def fit(self, **kwargs):
        stream = Stream(handle=None, tag="", tf=self.base)
        params = self.base.openParams("")
        return self.base._fit(kwargs, stream, params)

    def map(self, **kwargs):
        stream = Stream(handle=None, tag="", tf=self.base)
        self.base._map(kwargs, stream)
        stream.storage.pop("version")
        return stream.storage

    def params(self):
        return self.base.params()


class sopen(object):
    def __init__(self, module, data, verbose=False):
        self.module = module
        self.data = data
        self.verbose = verbose
        self.root = None

    def __enter__(self):
        self.root = self.module.open(self.data, verbose=self.verbose)
        return self.root

    def __exit__(self, *args):
        self.module.close(self.root)


stream = sopen


class hupdate(object):
    def __init__(self, module, updates, verbose=False):
        self.module = module
        self.updates = updates
        self.verbose = verbose
        self.backup = {}
        for field, val in self.updates.items():
            tf, arg = field.split(".")
            self.backup[field] = self.module[tf].args[arg]

    def __enter__(self):
        self.module.hyperUpdate(self.updates)

    def __exit__(self, *args):
        self.module.hyperUpdate(self.backup)
