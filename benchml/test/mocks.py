import json
import os
import benchml
from ..transforms import *
from ..logger import log, Mock

def assert_equal(x, y, tol):
    if np.abs(x-y) < tol: return True
    return False

def assert_array_equal(x, y, tol):
    if np.max(np.abs(x-y)) < tol: return True
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
        log << "Validate" << self.__class__.__name__ << log.endl
        try:
            ref = json.load(open(os.path.join(self.path, "test_ref.json")))
        except IOError:
            log << log.mr << "Reference output for" << self.__class__.__name__ \
                << "missing. Abort validation." << log.endl
            return False
        out = json.load(open(os.path.join(self.path, "test.json")))
        success = True
        for data, data_ref in zip(out, ref):
            log << "  Dataset=" << data["dataset"] << "model=" << data["model"] << log.endl
            for split in data["splits"]:
                log << "   -" << split << log.flush
                Y1 = data["output"][split]
                Y2 = data_ref["output"][split]
                for y1, y2 in zip(Y1, Y2):
                    y1 = np.array(y1["pred"])
                    y2 = np.array(y2["pred"])
                    check = assert_array_equal(y1, y2, self.tol)
                    success = success and check
                    if check: log << "+" << log.flush
                log << log.endl
        return success

