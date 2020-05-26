#! /usr/bin/env python
from test_morgan import *
from test_asap import *
from test_dscribe import *
import re
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
    parser = optparse.OptionParser()
    parser.add_option("-c", "--create", action="store_true", dest="create", 
        default=False, help="Create reference")
    parser.add_option("-l", "--list", action="store_true", dest="list",
        default=False, help="List all tests")
    parser.add_option("-f", "--filter", default=".*", dest="filter", 
        help="Filter regular expression")
    parser.add_option("-a", "--ase", action="store_true", dest="ase",
        default=False, help="Use ASE parser")
    args, _ = parser.parse_args()
    benchml.readwrite.configure(use_ase=args.ase)
    for mock in get_all_mocks():
        run = bool(re.match(re.compile(args.filter), mock.__name__))
        colour = log.mg
        if not run: colour = log.mb
        log << colour << "[%s] Test <%s>" % ("Run" if run else "---", mock.__name__) << log.endl
        if args.list or not run: continue
        mock.run(create=args.create)
        success = mock.validate()
        if not success:
            log << log.mr << "Test <%s> failed" % mock.__name__ << log.endl
        else:
            log << log.mg << "Test <%s> success" % mock.__name__  << log.endl

