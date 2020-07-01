from ..logger import log
from ..pipeline import Transform
import numpy as np
import os

def get_smiles(c):
    return c.info["smiles"] if "smiles" in c.info else c.info["SMILES"]

class CxCalcTransform(Transform):
    default_args = {
        "cxcalc": "/software/chemaxon/MarvinBeans/bin/cxcalc",
        "cmd": "logp --method consensus",
        "tmpdir": "tmp",
        "reshape_as_matrix": False
    }
    allow_stream = {"X",}
    stream_samples = ("X",)
    precompute = True
    def _setup(self, *args, **kwargs):
        if not os.path.isfile(self.args["cxcalc"]):
            raise IOError("Invalid cxcalc path")
    def _map(self, inputs):
        configs = inputs["configs"]
        smiles = [ '"%s"' % get_smiles(c) for c in configs ]
        log >> 'mkdir -p %s' % self.args["tmpdir"]
        compiled = '{cxcalc} {cmd} {smiles}'.format(
            cxcalc=self.args["cxcalc"],
            cmd=self.args["cmd"],
            smiles=" ".join(smiles))
        res = log >> log.catch >> compiled
        res = res.split("\n")
        header = res.pop(0)
        assert header.startswith('id')
        X = np.array([ float(r.split()[1]) for r in res ])
        if self.args["reshape_as_matrix"]:
            X = X.reshape((-1,1))
        assert X.shape[0] == len(configs)
        self.stream().put("X", X)

