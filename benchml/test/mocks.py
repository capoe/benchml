import json
import os
import benchml
from ..transforms import *
from ..logger import log, Mock

def assert_equal(x, y, tol):
    if np.abs(x-y) < tol: return True
    return False
        
class TestMock(Mock):
    group = None
    seed = 971231
    path = os.path.dirname(__file__)
    data_dir = ("..", "test_data")
    tol = 1e-6
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
        log << "Validate" << __class__ << log.endl
        try:
            ref = json.load(open(os.path.join(self.path, "test_ref.json")))
        except IOError:
            log << log.mr << "Reference output for" << __class__ \
                << "missing. Abort validation." << log.endl
            return False
        out = json.load(open(os.path.join(self.path, "test.json")))
        success = True
        for key, record_ref in sorted(ref.items()):
            log << "  %20s...%20s" % (key[0:20], key[-20:]) << log.flush
            record_out = out[key]
            for field, val in sorted(record_ref.items()):
                check = assert_equal(val, record_out[field], self.tol)
                success = success and check
                if check: log << "+" << log.flush
            log << log.endl
        return True

