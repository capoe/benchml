import json
import os

import numpy as np

import benchml as bml
from benchml.logger import Mock, log


def assert_array_equal(x, y, tol):
    if np.max(np.abs(x - y)) < tol:
        return True
    return False


class TestMock(Mock):
    model_regex = None
    seed = 971231
    path = os.path.dirname(__file__)
    data_dir = ("..", "test_data")
    tol = 1e-6

    def __init__(self):
        pass

    def getArgs(self, create):
        args = Mock()
        args.seed = 971231
        args.data_folder = os.path.join(self.path, *self.data_dir)
        args.filter = "none"
        args.model_regex = self.model_regex
        args.verbose = False
        args.output = os.path.join(self.path, "test_ref.json" if create else "test.json")
        return args

    def run(self, create=False):
        filters_map = {"none": None}
        args = self.getArgs(create=create)
        bml.splits.synchronize(args.seed)
        data = bml.data.BenchmarkData(
            root=args.data_folder, 
            filter_fct=filters_map.get(args.filter, None)
        )
        models = bml.models.compile_and_filter(filter_models=self.model_regex)
        bench = bml.benchmark.evaluate(
            data=data, 
            models=models, 
            log=log, 
            verbose=args.verbose, 
            detailed=True
        )
        json.dump(bench, open(args.output, "w"), indent=1, sort_keys=True)
        return self

    def validate(self):
        log << "Validate" << self.__class__.__name__ << log.endl
        try:
            ref = json.load(open(os.path.join(self.path, "test_ref.json")))
        except IOError:
            (
                log
                << log.mr
                << "Reference output for"
                << self.__class__.__name__
                << "missing. Abort validation."
                << log.endl
            )
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
                    if check:
                        log << "+" << log.flush
                    else:
                        log << "-" << log.flush
                log << log.endl
        return success

