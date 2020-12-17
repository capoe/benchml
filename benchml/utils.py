import numpy as np
import copy
import json
from .logger import log

try_smiles_key = [
    "smiles",
    "SMILES",
    "canonical_smiles",
    "CANONICAL_SMILES"
]

def get_smiles(config):
    for key in try_smiles_key:
        if key in config.info: break
    return config.info[key]

class LineExpansion(object):
    def __init__(self, interval, periodic, n_bins, sigma, type):
        self.x0 = interval[0]
        self.x1 = interval[1]
        self.dx = self.x1-self.x0
        self.periodic = periodic
        self.n_bins = n_bins
        self.type = type
        self.epsilon = 1e-10
        self.sigma = sigma
        self.setup()
    def setup(self):
        if self.periodic:
            self.res = self.dx/self.n_bins
            self.rbf_centers = np.linspace(self.x0, self.x1-self.res, self.n_bins)
        else:
            self.res = self.dx/(self.n_bins-1)
            self.rbf_centers = np.linspace(self.x0, self.x1, self.n_bins)
    def wrap(self, vals):
        if not self.periodic: 
            return vals
        else:
            return vals - ((vals+0.5*self.res)/(0.5*self.dx)).astype('int')*self.dx
    def expand(self, vals):
        vals_wrapped = self.wrap(vals)
        vals_expand = np.subtract.outer(vals_wrapped, self.rbf_centers)
        if self.type == "heaviside":
            vals_expand = np.heaviside(-np.abs(vals_expand)+1e-10+0.5*self.res, 0.0)
        elif self.type == "gaussian":
            vals_expand = np.exp(-0.5*vals_expand**2/self.sigma**2)
        else: raise ValueError(self.type)
        vals_expand = (vals_expand.T/(np.sum(vals_expand, axis=1)+self.epsilon)).T
        return vals_expand 

