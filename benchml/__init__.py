"""BenchML.

.. moduleauthor:: Carl Poelking <cp605@cam.ac.uk>
"""

from benchml import analysis, benchmark, data, models, pipeline, splits, transforms, utils
from benchml.accumulator import Accumulator
from benchml.data import load_dataset
from benchml.logger import log
from benchml.pipeline import hupdate, sopen, stream
from benchml.readwrite import load, read, save, write
from benchml.splits import Split
from benchml.transforms import Transform

__version__ = "0.3.1"
__all__ = [
    "analysis",
    "benchmark",
    "data",
    "models",
    "pipeline",
    "splits",
    "transforms",
    "utils",
    "Accumulator",
    "load_dataset",
    "log",
    "hupdate",
    "sopen",
    "stream",
    "load",
    "read",
    "save",
    "write",
    "Split",
    "Transform",
]
