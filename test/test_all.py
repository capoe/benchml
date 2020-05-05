#! /usr/bin/env python
from test_morgan import *
import benchml
import sys
import inspect
import optparse
# NOTE Independent random seeds used in
# - Split
# - BayesianHyper
np.random.seed(712311)

def get_bases_recursive(obj):
    bases = list(obj.__bases__)
    sub = []
    for b in bases:
        sub = sub + get_bases_recursive(b)
    bases = bases + sub
    return bases

def get_all_mocks(verbose=False):
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj):
            if Mock in get_bases_recursive(obj):
                yield obj

if __name__ == "__main__":
    benchml.readwrite.configure(use_ase=False)
    parser = optparse.OptionParser()
    parser.add_option("-c", "--create", action="store_true", dest="create", 
        default=False, help="Create reference")
    args, _ = parser.parse_args()
    for mock in get_all_mocks():
        log << "Test <%s> start" % mock.__name__ << log.endl
        mock.run(create=args.create)
        success = mock.validate()
        if not success:
            log << log.mr << "Test <%s> failed" % mock.__name__ << log.endl
        else:
            log << log.mg << "Test <%s> success" % mock.__name__  << log.endl

