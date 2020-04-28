#! /usr/bin/env python
import benchml
import optparse
import json
from benchml.transforms import *
log = benchml.log
benchml.readwrite.configure(use_ase=False)

def main(args):
    # Load datasets (as iterator)
    benchml.splits.synchronize(args.seed)
    data = benchml.data.compile(
        root=args.data_folder,
        filter_fct=benchml.filters[args.filter])
    # Compile models
    models = benchml.models.compile(args.groups.split())
    # Evaluate
    bench = benchml.benchmark.evaluate(
        data, models, log, verbose=args.verbose)
    json.dump(bench, open(args.output, "w"), indent=1, sort_keys=True)

if __name__ == "__main__":
    parser = optparse.OptionParser()
    parser.add_option("-d", "--data_folder", dest="data_folder", default="./data",
        help="Dataset folder", metavar="D")
    parser.add_option("-g", "--groups", dest="groups", default="null",
        help="Model groups", metavar="G")
    parser.add_option("-f", "--filter", dest="filter", default="none", 
        help="Dataset filter", metavar="F")
    parser.add_option("-o", "--output", dest="output", default="bench.json", 
        help="Output benchmark json file", metavar="J")
    parser.add_option("-s", "--seed", dest="seed", default=0, 
        help="Global random seed", metavar="S")
    parser.add_option("-v", "--verbose", action="store_true", dest="verbose", default=False,
        help="Set verbose output")
    parser.add_option("-l", "--list_transforms", action="store_true", dest="list_transforms", 
        default=False, help="List available transforms and quit")
    args, _ = parser.parse_args()
    if args.list_transforms:
        benchml.transforms.list_all(verbose=args.verbose)
        log.okquit()
    main(args)

