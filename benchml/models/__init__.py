import re

import numpy as np

from benchml.logger import log
from benchml.models import (
    mod_basic,
    mod_bench,
    mod_bench_class,
    mod_bench_xtal,
    mod_dscribe,
    mod_logd,
    mod_logd_ai,
    mod_xy,
)

collections = {}
for register in [
    mod_basic.register_all,
    mod_dscribe.register_all,
    mod_bench.register_all,
    mod_bench_xtal.register_all,
    mod_bench_class.register_all,
    mod_logd.register_all,
    mod_logd_ai.register_all,
    mod_xy.register_all,
]:
    collections.update(register())


def list_all(verbose=True):
    log << "Model collections in register:" << log.endl
    for group, collection in sorted(collections.items()):
        models = collection()
        avail = np.array([m.check_available() for m in models]).all()
        (
            log
            << (log.mg if avail else log.mb)
            << "  Collection %-20s [%d models," % (group, len(models))
            << "available=%s]" % str(avail)
            << log.endl
        )
        if verbose:
            for model in models:
                log << "  - %-25s" % model.tag << log.endl


def compile(groups, **kwargs):
    selected = [model for group in groups for model in collections[group](**kwargs)]
    return selected


def get(re):
    if isinstance(re, str):
        re = [re]
    return compile_and_filter(filter_collections=[".*"], filter_models=re)


def compile_and_filter(filter_collections=[".*"], filter_models=[".*"], verbose=False):
    if verbose:
        log << "Compile & filter models" << log.endl
    filter_models = [re.compile(f) for f in filter_models]
    filter_collections = [re.compile(c) for c in filter_collections]
    check_added = set()
    filtered_models = []
    for group, collection in sorted(collections.items()):
        if not np.array([f.match(group) for f in filter_collections]).any():
            continue
        models = collection()
        for m in models:
            avail = m.check_available()
            matches = np.array([f.match(m.tag) for f in filter_models]).any()
            if matches and not avail:
                (
                    log
                    << log.mr
                    << " - Exclude '%s/%s' (requested but not available)" % (group, m.tag)
                    << log.endl
                )
            elif matches:
                if m.tag not in check_added:
                    log << " - Added '%s/%s'" % (group, m.tag) << log.endl
                    check_added.add(m.tag)
                    filtered_models.append(m)
                else:
                    log << log.mr << " - '%s/%s' skipped (duplicate)" % (group, m.tag) << log.endl
            elif not avail:
                if verbose:
                    (
                        log
                        << log.my
                        << " - Exclude '%s/%s' (not available)" % (group, m.tag)
                        << log.endl
                    )
    return filtered_models
