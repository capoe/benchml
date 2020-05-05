#! /usr/bin/env python
import json
import os
import benchml
from benchml.transforms import *
from benchml.logger import log, Mock
path = os.path.dirname(__file__)

class TestMorgan(Mock):
    def run(create=False):
        args = Mock()
        args.seed = 971231
        args.data_folder = os.path.join(path, "..", "test_data")
        args.filter = "none"
        args.groups = "morgan"
        args.verbose = False
        args.output = os.path.join(path, "test_ref.json" if create else "test.json")
        benchml.splits.synchronize(args.seed)
        data = benchml.data.compile(
            root=args.data_folder,
            filter_fct=benchml.filters[args.filter])
        models = benchml.models.compile(args.groups.split())
        bench = benchml.benchmark.evaluate(
            data, models, log, verbose=args.verbose)
        json.dump(bench, open(args.output, "w"), indent=1, sort_keys=True)
    def validate():
        diff = log >> log.catch >> 'diff %s %s' % (
            os.path.join(path, "test.json"), os.path.join(path, "test_ref.json"))
        if diff.strip() != "": return False
        return True

