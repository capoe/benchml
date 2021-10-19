"""BenchML __init__.py

.. moduleauthor:: Carl Poelking <cp605@cam.ac.uk>

"""

from . import analysis, benchmark, data, models, pipeline, splits, transforms, utils
from .accumulator import Accumulator
from .data import load_dataset
from .logger import log
from .pipeline import hupdate, sopen, stream
from .readwrite import load, read, save, write
from .splits import Split
from .transforms import Transform

__version__ = "0.1.2"
