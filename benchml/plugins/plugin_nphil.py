from ..pipeline import Transform
from ..logger import log
from .plugin_check import *
import numpy as np

class NonlinearFeatureFilter(Transform):
    default_args = {
        "uops": ["el|sr2"],
        "bops": ["+-:*"],
        "unit_min_exp": 0.5,
        "unit_max_exp": 3.0,
        "correlation_measure": "moment",
        "rank_coeff": 0.5
    }
    req_inputs = {"X","Y","meta"}
    allow_stream = {"X",}
    allow_params = {"fgraph","ranked","variables"}
    def check_available():
        return check_nphil_available(__class__)
    def _fit(self, inputs, stream, params):
        X = np.copy(inputs["X"])
        Y = np.copy(inputs["Y"])
        if len(Y.shape) > 1 and Y.shape[1] > 1:
            raise NotImplementedError(
                "NonlinearFeatureFilter requires single-column"+\
                " vector Y, but got multi-column matrix")
        variables = inputs["meta"]["variables"]
        fgraph = nphil.generate.generate_graph(
            variables=variables,
            uop_list=self.args["uops"],
            bop_list=self.args["bops"],
            unit_min_exp=self.args["unit_min_exp"],
            unit_max_exp=self.args["unit_max_exp"],
            correlation_measure=self.args["correlation_measure"],
            rank_coeff=self.args["rank_coeff"])
        covs = np.zeros((len(fgraph),), dtype=X.dtype)
        X_out = np.zeros((len(X),len(fgraph)), dtype=X.dtype)
        Y = Y.reshape((-1,1))
        fgraph.applyAndCorrelate(
            X,
            Y,
            X_out, # <- Careful, modified=z-scored in place
            covs,
            Y.shape[0],
            Y.shape[1])
        order = np.argsort(np.abs(covs))[::-1]
        params.put("fgraph", fgraph)
        params.put("ranked", order)
        params.put("variables", variables)
        self._map(inputs, stream)
    def _map(self, inputs, stream):
        X = np.copy(inputs["X"])
        fgraph = self.params().get("fgraph")
        ranked = self.params().get("ranked")
        X_out = np.zeros((len(X),len(fgraph)), dtype=X.dtype)
        fgraph.apply(X, X_out, len(X))
        stream.put("X", X_out[:,ranked[0]].reshape((-1,1)))
        
