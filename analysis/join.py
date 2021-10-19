#! /usr/bin/env python
import glob
import gzip
import json

import benchml as bml

bml.log.Connect()
bml.log.AddArg("dataset", str)
args = bml.log.Parse()

js = sorted(glob.glob("%s*.json" % args.dataset))
data = []
for j in js:
    data.extend(json.load(open(j)))

with gzip.GzipFile("benchmark_%s.json.gz" % args.dataset, "w") as fout:
    fout.write(json.dumps(data, indent=1, sort_keys=True).encode("utf-8"))
