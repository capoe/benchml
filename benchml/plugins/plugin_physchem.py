from ..pipeline import Transform, Macro
from ..utils import get_smiles, LineExpansion
import numpy as np
from .plugin_check import *

class PhyschemXtal(Transform):
    default_args = {
        "bins": 10,
        "sigma_fac": 1.0}
    allow_stream = ("X",)
    stream_samples = ("X",)
    req_inputs = ("configs",)
    precompute = True
    fct_type = { 
        "discrete": "heaviside", 
        "continuous": "gaussian" }
    def _setup(self, *args, **kwargs):
        self.atom_data, self.atom_data_types = get_atomic_data()
        X = np.array([ x for e, x in self.atom_data.items() ])
        self.n_atom_features = X.shape[1]
        self.n_types = len(self.atom_data)
        self.xmin = np.min(X, axis=0)
        self.xmax = np.max(X, axis=0)
        self.xstd = np.std(X, axis=0)
        self.bins = [ int(self.xmax[0] - self.xmin[0] + 1.5) ] + \
            [ self.args["bins"] for i in range(self.n_atom_features-1) ]
        self.xbasis = [ \
            LineExpansion(
                interval=[ self.xmin[i], self.xmax[i] ],
                periodic=False,
                n_bins=self.bins[i],
                sigma=self.args["sigma_fac"]*self.xstd[i],
                type=self.fct_type[self.atom_data_types[i]]) \
            for i in range(0, self.n_atom_features) ]
    def _map(self, inputs, stream):
        X = [ self.mapSingleConfig(c) for c in inputs["configs"] ]
        X = np.array(X)
        stream.put("X", X)
    def mapSingleConfig(self, config):
        cell = config.get_cell()
        if cell is None:
            raise RuntimeError("ERROR Using PhyschemXtal for non-periodic system")
        volume = np.cross(cell[0], cell[1]).dot(cell[2])
        x = np.array([ self.atom_data[s] for s in config.symbols ])
        x_spectrum = []
        for i in range(self.n_atom_features):
            xi = x[:,i] 
            xi_spectrum = self.xbasis[i].expand(xi)
            xi_spectrum_red = np.sum(xi_spectrum, axis=0)
            xi_spectrum_number = xi_spectrum_red/np.sum(xi_spectrum_red)
            xi_spectrum_volume = xi_spectrum_red/volume
            x_spectrum.append(xi_spectrum_number)
            x_spectrum.append(xi_spectrum_volume)
        return np.concatenate(x_spectrum)

def get_atomic_data():
    # Source: https://github.com/rouyang2017/SISSO/blob/master/utilities/atom_features
    #         (curated from https://www.webelements.com)
    # Columns:   Z       IE           XP        Rc        XAR
    return {
        "H":  [  1,      13.59,       2.20,     0.31,     2.20 ], 
        "Li": [  3,      5.391,       0.98,     1.28,     0.97 ],  
        "Na": [ 11,      5.138,       0.93,     1.66,     1.01 ],  
        "K":  [ 19,      4.340,       0.82,     2.03,     0.91 ],  
        "Rb": [ 37,      4.176,       0.82,     2.20,     0.89 ],  
        "Cs": [ 55,      3.893,       0.79,     2.44,     0.86 ],  
        "Fr": [ 87,      4.073,       0.70,     2.60,     0.86 ],  
        "Be": [  4,      9.322,       1.57,     0.96,     1.47 ],  
        "Mg": [ 12,      7.645,       1.31,     1.41,     1.23 ],  
        "Ca": [ 20,      6.112,       1.00,     1.76,     1.04 ],  
        "Sr": [ 38,      5.695,       0.95,     1.95,     0.99 ],  
        "Ba": [ 56,      5.212,       0.89,     2.15,     0.97 ],  
        "Ra": [ 88,      5.279,       0.97,     2.21,     0.97 ],  
        "Sc": [ 21,      6.561,       1.36,     1.70,     1.20 ],  
        "Y":  [ 39,      6.218,       1.22,     1.90,     1.11 ],  
        "Lu": [ 71,      5.425,       1.27,     1.87,     1.14 ],  
        "Ti": [ 22,      6.827,       1.54,     1.60,     1.32 ],  
        "Zr": [ 40,      6.634,       1.33,     1.75,     1.22 ],  
        "Hf": [ 72,      6.824,       1.30,     1.75,     1.23 ],  
        "V":  [ 23,      6.746,       1.63,     1.53,     1.45 ],  
        "Nb": [ 41,      6.758,       1.60,     1.64,     1.23 ],  
        "Ta": [ 73,      7.887,       1.50,     1.70,     1.33 ],  
        "Cr": [ 24,      6.766,       1.66,     1.39,     1.56 ],  
        "Mo": [ 42,      7.092,       2.16,     1.54,     1.30 ],  
        "W":  [ 74,      7.980,       2.36,     1.62,     1.40 ],  
        "Mn": [ 25,      7.434,       1.55,     1.39,     1.60 ],  
        "Tc": [ 43,      7.275,       1.90,     1.47,     1.36 ],  
        "Re": [ 75,      7.877,       1.90,     1.51,     1.46 ],  
        "Fe": [ 26,      7.902,       1.83,     1.32,     1.64 ],  
        "Ru": [ 44,      7.361,       2.22,     1.42,     1.42 ],  
        "Os": [ 76,      8.706,       2.20,     1.44,     1.52 ],  
        "Co": [ 27,      7.880,       1.88,     1.26,     1.70 ],  
        "Rh": [ 45,      7.459,       2.28,     1.42,     1.45 ],  
        "Ir": [ 77,      9.121,       2.20,     1.41,     1.55 ],  
        "Ni": [ 28,      7.639,       1.91,     1.24,     1.75 ],  
        "Pd": [ 46,      8.337,       2.20,     1.39,     1.35 ],  
        "Pt": [ 78,      9.017,       2.28,     1.36,     1.44 ],  
        "Cu": [ 29,      7.726,       1.90,     1.32,     1.75 ],  
        "Ag": [ 47,      7.576,       1.93,     1.45,     1.42 ],  
        "Au": [ 79,      9.225,       2.54,     1.36,     1.42 ],  
        "Zn": [ 30,      9.394,       1.65,     1.22,     1.66 ],  
        "Cd": [ 48,      8.994,       1.69,     1.44,     1.46 ],  
        "Hg": [ 80,      10.43,       2.00,     1.32,     1.44 ],  
        "B":  [  5,      8.297,       2.04,     0.84,     2.01 ],  
        "Al": [ 13,      5.985,       1.61,     1.21,     1.47 ],  
        "Ga": [ 31,      5.998,       1.81,     1.22,     1.82 ],  
        "In": [ 49,      5.786,       1.78,     1.42,     1.49 ],  
        "Tl": [ 81,      6.108,       1.62,     1.45,     1.44 ],  
        "C":  [  6,      11.26,       2.55,     0.76,     2.50 ],  
        "Si": [ 14,      8.151,       1.90,     1.11,     1.74 ],  
        "Ge": [ 32,      7.897,       2.01,     1.20,     2.02 ],  
        "Sn": [ 50,      7.344,       1.96,     1.39,     1.72 ],  
        "Pb": [ 82,      7.416,       2.33,     1.46,     1.55 ],  
        "N":  [  7,      14.53,       3.04,     0.71,     3.07 ],  
        "P":  [ 15,      10.48,       2.19,     1.07,     2.06 ],  
        "As": [ 33,      9.814,       2.18,     1.19,     2.20 ],  
        "Sb": [ 51,      8.643,       2.05,     1.39,     1.82 ],  
        "Bi": [ 83,      7.286,       2.02,     1.48,     1.67 ],  
        "O":  [  8,      13.61,       3.44,     0.66,     3.50 ],  
        "S":  [ 16,      10.36,       2.58,     1.05,     2.44 ],  
        "Se": [ 34,      9.752,       2.55,     1.20,     2.48 ],  
        "Te": [ 52,      9.009,       2.10,     1.38,     2.01 ],  
        "Po": [ 84,      8.416,       2.00,     1.40,     1.76 ],  
        "F":  [  9,      17.42,       3.98,     0.57,     4.10 ],  
        "Cl": [ 17,      12.96,       3.16,     1.02,     2.83 ],  
        "Br": [ 35,      11.81,       2.96,     1.20,     2.74 ],  
        "I":  [ 53,      10.45,       2.66,     1.39,     2.21 ],  
        "At": [ 85,      9.317,       2.20,     1.50,     1.90 ],  
        "La": [ 57,      5.577,       1.10,     2.07,     1.08 ],  
        "Ce": [ 58,      5.538,       1.12,     2.04,     1.08 ],  
        "Pr": [ 59,      5.461,       1.13,     2.03,     1.07 ],  
        "Nd": [ 60,      5.525,       1.14,     2.01,     1.07 ],  
        "Pm": [ 61,      5.597,       1.15,     1.99,     1.07 ],  
        "Sm": [ 62,      5.643,       1.17,     1.98,     1.07 ],  
        "Eu": [ 63,      5.670,       1.18,     1.98,     1.01 ],  
        "Gd": [ 64,      6.150,       1.20,     1.96,     1.11 ],  
        "Tb": [ 65,      5.864,       1.21,     1.94,     1.10 ],  
        "Dy": [ 66,      5.938,       1.22,     1.92,     1.10 ],  
        "Ho": [ 67,      6.021,       1.23,     1.92,     1.10 ],  
        "Er": [ 68,      6.107,       1.24,     1.89,     1.11 ],  
        "Tm": [ 69,      6.184,       1.25,     1.90,     1.11 ],  
        "Yb": [ 70,      6.253,       1.25,     1.87,     1.06 ],  
        "Ac": [ 89,      5.172,       1.10,     2.15,     1.00 ],  
        "Th": [ 90,      6.083,       1.30,     2.06,     1.11 ],  
        "Pa": [ 91,      5.887,       1.50,     2.00,     1.14 ],  
        "U":  [ 92,      6.193,       1.38,     1.96,     1.22 ],  
        "Np": [ 93,      6.265,       1.36,     1.90,     1.22 ],  
        "Pu": [ 94,      6.059,       1.28,     1.87,     1.22 ],  
        "Am": [ 95,      5.991,       1.30,     1.80,     1.20 ],  
        "Cm": [ 96,      6.022,       1.30,     1.69,     1.20 ],  
        "Bk": [ 97,      6.198,       1.30,     1.69,     1.20 ],  
        "Cf": [ 98,      6.282,       1.30,     1.69,     1.20 ],  
        "Es": [ 99,      6.368,       1.30,     1.69,     1.20 ]  
    }, [ "discrete", "continuous", "continuous", "continuous", "continuous" ]

class PhyschemUser(Transform):
    req_inputs = {"configs",}
    req_args = {"fields",}
    default_args = {
        "fields": [],
    }
    allow_stream = ("X",)
    stream_samples = ("X",)
    def _map(self, inputs, stream):
        X = []
        try:
            for config in inputs["configs"]:
                x = [ float(config.info[f]) for f in self.args["fields"] ]
                X.append(x)
        except KeyError as err:
            raise KeyError("PhyschemUser node expects missing custom field %s" % err)
        X = np.array(X)
        stream.put("X", X)

class Physchem2D(Transform):
    allow_stream = ("X",)
    stream_samples = ("X",)
    req_inputs = ("configs",)
    precompute = True
    help_args = {
        "select": [ 
            "list(str)", 
            "choice of descriptors", 
            "lambda t: [ d[0] for d in t.descriptors ]" ]
    }
    default_args = {
        "select_predef": "extended",
        "select": None
    }
    predefined = {
        "basic": [
            "tpsa",       
            "mollogp",
        ],
        "core": [
            "molwt",      
            "n_hacc",     
            "n_hdon",     
            "tpsa",       
            "mollogp",
        ],
        "logp": [
            "tpsa",       
            "mollogp",
            "slogp01",
            "slogp02",
            "slogp03",
            "slogp04",
            "slogp05",
            "slogp06",
            "slogp07",
            "slogp08",
            "slogp09",
            "slogp10",
            "slogp11",
            "slogp12"
        ],
        "extended": [
            "molwt",      
            "n_hacc",     
            "n_hdon",     
            "tpsa",       
            "mollogp",    
            "molmr",      
            "molwtheavy", 
            "n_heavy",    
            "n_nhoh",     
            "n_no",       
            "n_hetero",   
            "n_rotbond",  
            "n_valel",    
            "n_aromaring",
            "n_satring",  
            "n_aliphring",
            "n_ring"
            "slogp01",
            "slogp02",
            "slogp03",
            "slogp04",
            "slogp05",
            "slogp06",
            "slogp07",
            "slogp08",
            "slogp09",
            "slogp10",
            "slogp11",
            "slogp12",
        ]
    }
    descriptors_active = []
    descriptors = [] if (rdesc is None) else [
        # Core
        ["molwt",         "rdesc.ExactMolWt"],
        ["n_hacc",        "rdesc.NumHAcceptors"],
        ["n_hdon",        "rdesc.NumHDonors"],
        ["tpsa",          "rdesc.TPSA"],
        ["mollogp",       "rdesc.MolLogP"],
        # Extended
        ["molmr",         "rdesc.MolMR"],
        ["molwtheavy",    "rdesc.HeavyAtomMolWt"],
        ["n_heavy",       "rdesc.HeavyAtomCount"],
        ["n_nhoh",        "rdesc.NHOHCount"],
        ["n_no",          "rdesc.NOCount"],
        ["n_hetero",      "rdesc.NumHeteroatoms"],
        ["n_rotbond",     "rdesc.NumRotatableBonds"],
        ["n_valel",       "rdesc.NumValenceElectrons"],
        ["n_aromaring",   "rdesc.NumAromaticRings"],
        ["n_satring",     "rdesc.NumSaturatedRings"],
        ["n_aliphring",   "rdesc.NumAliphaticRings"],
        ["n_ring",        "rdesc.RingCount"],
        # Complexity
        ["balabanj",      "rdesc.BalabanJ"],
        ["bertzct",       "rdesc.BertzCT"],
        ["ipc",           "rdesc.Ipc"],
        ["hallkieralpha", "rdesc.HallKierAlpha"],
        ["kappa1",        "rdesc.Kappa1"],
        ["kappa2",        "rdesc.Kappa2"],
        ["kappa3",        "rdesc.Kappa3"],
        ["chi0",          "rdesc.Chi0"],
        ["chi1",          "rdesc.Chi1"],
        ["chi0n",         "rdesc.Chi0n"],
        ["chi1n",         "rdesc.Chi1n"],
        ["chi2n",         "rdesc.Chi2n"],
        ["chi3n",         "rdesc.Chi3n"],
        ["chi4n",         "rdesc.Chi4n"],
        ["chi0v",         "rdesc.Chi0v"],
        ["chi1v",         "rdesc.Chi1v"],
        ["chi2v",         "rdesc.Chi2v"],
        ["chi3v",         "rdesc.Chi3v"],
        ["chi4v",         "rdesc.Chi4v"],
        # Surface
        ["slogp01",       "rdesc.SlogP_VSA1"],
        ["slogp02",       "rdesc.SlogP_VSA2"],
        ["slogp03",       "rdesc.SlogP_VSA3"],
        ["slogp04",       "rdesc.SlogP_VSA4"],
        ["slogp05",       "rdesc.SlogP_VSA5"],
        ["slogp06",       "rdesc.SlogP_VSA6"],
        ["slogp07",       "rdesc.SlogP_VSA7"],
        ["slogp08",       "rdesc.SlogP_VSA8"],
        ["slogp09",       "rdesc.SlogP_VSA9"],
        ["slogp10",       "rdesc.SlogP_VSA10"],
        ["slogp11",       "rdesc.SlogP_VSA11"],
        ["slogp12",       "rdesc.SlogP_VSA12"],
        ["smr01",         "rdesc.SMR_VSA1"],
        ["smr02",         "rdesc.SMR_VSA2"],
        ["smr03",         "rdesc.SMR_VSA3"],
        ["smr04",         "rdesc.SMR_VSA4"],
        ["smr05",         "rdesc.SMR_VSA5"],
        ["smr06",         "rdesc.SMR_VSA6"],
        ["smr07",         "rdesc.SMR_VSA7"],
        ["smr08",         "rdesc.SMR_VSA8"],
        ["smr09",         "rdesc.SMR_VSA9"],
        ["smr10",         "rdesc.SMR_VSA10"],
        ["peoe01",        "rdesc.PEOE_VSA1"],
        ["peoe02",        "rdesc.PEOE_VSA2"],
        ["peoe03",        "rdesc.PEOE_VSA3"],
        ["peoe04",        "rdesc.PEOE_VSA4"],
        ["peoe05",        "rdesc.PEOE_VSA5"],
        ["peoe06",        "rdesc.PEOE_VSA6"],
        ["peoe07",        "rdesc.PEOE_VSA7"],
        ["peoe08",        "rdesc.PEOE_VSA8"],
        ["peoe09",        "rdesc.PEOE_VSA9"],
        ["peoe10",        "rdesc.PEOE_VSA10"],
        ["peoe11",        "rdesc.PEOE_VSA11"],
        ["peoe12",        "rdesc.PEOE_VSA12"],
        ["peoe13",        "rdesc.PEOE_VSA13"],
        ["peoe14",        "rdesc.PEOE_VSA14"],
    ]
    def _setup(self, *args, **kwargs):
        if self.args["select"] is not None:
            select = set(self.args["select"])
        else:
            select = set(self.predefined[
                self.args["select_predef"]]) # Invalid arg in Physchem2D.select_set?
        self.descriptors_active = list(filter(
            lambda d: d[0] in select,
            self.descriptors))
        if len(self.descriptors_active) < 1:
            raise ValueError("Empty or invalid descriptor list in Physchem2D.select")
    def _map(self, inputs, stream):
        configs = inputs["configs"]
        smiles = [ get_smiles(c) for c in configs ]
        mols = [ rchem.MolFromSmiles(s) for s in smiles ]
        X = []
        for mol in mols:
            x = [ 1.*eval(d[1])(mol) for d in self.descriptors_active ]
            X.append(x)
        X = np.array(X)
        stream.put("X", X)

