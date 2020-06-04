import json
import os
import benchml
from ..transforms import *
from ..logger import log, Mock

class TestMock(Mock):
    group = None
    seed = 971231
    path = os.path.dirname(__file__)
    data_dir = ("..", "test_data")
    def __init__(self):
        pass
    def run(self, create=False):
        args = Mock()
        args.seed = 971231
        args.data_folder = os.path.join(self.path, *self.data_dir)
        args.filter = "none"
        args.groups = self.group
        args.verbose = False
        args.output = os.path.join(self.path, "test_ref.json" if create else "test.json")
        benchml.splits.synchronize(args.seed)
        data = benchml.data.compile(
            root=args.data_folder,
            filter_fct=benchml.filters[args.filter])
        models = benchml.models.compile(args.groups.split())
        bench = benchml.benchmark.evaluate(
            data, models, log, verbose=args.verbose)
        json.dump(bench, open(args.output, "w"), indent=1, sort_keys=True)
        return self
    def validate(self):
        diff = log >> log.catch >> 'diff %s %s' % (
            os.path.join(self.path, "test.json"), os.path.join(self.path, "test_ref.json"))
        if diff.strip() != "": return False
        return True

