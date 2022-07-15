import os
from shutil import which

import numpy as np

from benchml.logger import log
from benchml.pipeline import Transform


def get_smiles(c):
    return c.info["smiles"] if "smiles" in c.info else c.info["SMILES"]


class CxCalcTransform(Transform):
    default_args = {
        "cxcalc": "cxcalc",
        "cmd": "logp --method consensus",
        "tmpdir": "tmp",
        "reshape_as_matrix": False,
        "batch_size": 100,  # To avoid a line buffer overflow in tmux ... :/
    }
    allow_stream = {
        "X",
    }
    stream_samples = ("X",)
    precompute = True

    def _setup(self, *args, **kwargs):
        given_path = self.args.get("cxcalc")
        if given_path:
            cxcalc_path = which(given_path)
        else:
            cxcalc_path = None
        if cxcalc_path is None:
            msg = f"Did not find path of Chemaxon Calculator 'cxcalc'; given path: {given_path}"
            raise IOError(msg)
        if os.environ.get("CHEMAXON_LICENSE_URL") is None:
            msg_l = "CHEMAXON_LICENSE_URL environment variable is not set, cxcalc will likely fail."
            if os.environ.get("CHEMAXON_HOME") is None:
                msg_h = "CHEMAXON_HOME environment variable is not set."
                log << UserWarning(msg_h) << log.endl
            log << UserWarning(msg_l) << log.endl

    def _map(self, inputs, stream):
        configs = inputs["configs"]
        X_collect = []
        smiles = ['"%s"' % get_smiles(c) for c in configs]
        log >> "mkdir -p %s" % self.args["tmpdir"]
        batch_size = self.args["batch_size"]
        n_batches = len(smiles) // batch_size + (1 if len(smiles) % batch_size != 0 else 0)
        for batch in range(n_batches):
            log << "Batch" << (batch + 1) << "/" << n_batches << log.endl
            smiles_batch = smiles[batch * batch_size : (batch + 1) * batch_size]
            compiled = "{cxcalc} {cmd} {smiles}".format(
                cxcalc=self.args["cxcalc"], cmd=self.args["cmd"], smiles=" ".join(smiles_batch)
            )
            try:
                res = log >> log.catch >> compiled
            except KeyboardInterrupt:
                break
            res = res.split("\n")
            header = res.pop(0)
            assert header.startswith("id")
            X = np.array([float(r.split()[1]) for r in res])
            X_collect.append(X)
        X = np.concatenate(X_collect)
        if self.args["reshape_as_matrix"]:
            X = X.reshape((-1, 1))
        assert X.shape[0] == len(configs)
        stream.put("X", X)
