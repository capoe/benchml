from ..pipeline import Transform, Macro
from ..utils import get_smiles
import numpy as np
from .plugin_check import *

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

