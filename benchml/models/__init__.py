from . import mod_basic
from . import mod_dscribe

collections = {}
for register in [ 
        mod_basic.register_all,
        mod_dscribe.register_all
      ]:
    collections.update(register())

import numpy as np
from ..logger import log

def list_all(verbose=True):
    log << "Model collections in register:" << log.endl
    for group, collection in sorted(collections.items()):
        models = collection()
        avail = np.array([ m.check_available() for m in models ]).all()
        log << (log.mg if avail else log.mb) \
            << "  Collection %-20s [%d models," % (group, len(models)) \
            << "available=%s]" % str(avail) << log.endl
        if verbose:
            for model in models:
                log << "  - %-25s" % model.tag \
                    << log.endl

def compile(groups, **kwargs):
    selected = [ model \
        for group in groups \
            for model in collections[group](**kwargs) ]
    return selected

