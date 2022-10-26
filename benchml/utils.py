import os
import subprocess
import time

import numpy as np

from benchml import readwrite
from benchml.logger import log

try:
    import rdkit.Chem as chem
except ImportError:
    chem = None


class LineExpansion:
    def __init__(self, interval, periodic, n_bins, sigma, type):
        self.x0 = interval[0]
        self.x1 = interval[1]
        self.dx = self.x1 - self.x0
        self.periodic = periodic
        self.n_bins = n_bins
        self.type = type
        self.epsilon = 1e-10
        self.sigma = sigma
        self.setup()
        self.res = None
        self.rbf_centers = None

    def setup(self):
        if self.periodic:
            self.res = self.dx / self.n_bins
            self.rbf_centers = np.linspace(self.x0, self.x1 - self.res, self.n_bins)
        else:
            self.res = self.dx / (self.n_bins - 1)
            self.rbf_centers = np.linspace(self.x0, self.x1, self.n_bins)

    def wrap(self, vals):
        if not self.periodic:
            return vals
        else:
            return vals - ((vals + 0.5 * self.res) / (0.5 * self.dx)).astype("int") * self.dx

    def expand(self, vals):
        vals_wrapped = self.wrap(vals)
        vals_expand = np.subtract.outer(vals_wrapped, self.rbf_centers)
        if self.type == "heaviside":
            vals_expand = np.heaviside(-np.abs(vals_expand) + 1e-10 + 0.5 * self.res, 0.0)
        elif self.type == "gaussian":
            vals_expand = np.exp(-0.5 * vals_expand**2 / self.sigma**2)
        else:
            raise ValueError(self.type)
        vals_expand = (vals_expand.T / (np.sum(vals_expand, axis=1) + self.epsilon)).T
        return vals_expand


class StagedTimer:
    def __init__(self):
        self.stages = []
        self.times = {}
        self.current_idx = -1
        self.t_in = None
        self.t_out = None

    def time(self, stage):
        self.stages.append(stage)
        self.current_idx += 1
        return self

    def __enter__(self):
        self.t_in = time.time()
        return self.stages[self.current_idx]

    def __exit__(self, *args, **kwargs):
        self.t_out = time.time()
        self.times[self.stages[self.current_idx]] = self.t_out - self.t_in

    def report(self, log):
        (
            log
            << "    "
            << " ".join(list(map(lambda s: "dt(%s)=%1.4fs" % (s, self.times[s]), self.stages)))
            << log.endl
        )


class OneHot:
    def __init__(self, cats):
        self.cats = cats
        self.index = {c: i for i, c in enumerate(cats)}

    def dim(self):
        return len(self.cats)

    def map(self, cats):
        x = np.zeros((len(cats), len(self.cats)))
        for i, c in enumerate(cats):
            x[i, self.index[c]] = 1
        return x


try_smiles_key = ["smiles", "SMILES", "canonical_smiles", "CANONICAL_SMILES"]


def get_smiles_key(dict_to_check, verbose=False):
    keys = []
    for key in try_smiles_key:
        if key in dict_to_check:
            keys.append(key)
    if len(keys) >= 1:
        if len(keys) > 1 and verbose:
            msg = f"WARNING: several 'smiles' fields are found, chosen {keys[0]}"
            log << msg << log.endl << log.flush
        return keys[0]
    else:
        raise ValueError("No 'smiles' field was found")


def get_smiles(config):
    return config.info[get_smiles_key(config.info)]


def smiles_to_extxyz(
    smiles,
    gen3d=False,
    throw_error=False,
    tmpfolder="tmp",
    corina="/path/to/corina",
    babel="/path/to/babel",
):
    if not gen3d:
        return smiles_to_pseudo_extxyz(smiles).pop()
    log >> "mkdir -p %s" % tmpfolder
    log >> "rm -f %s/tmp.*" % tmpfolder
    with open("%s/tmp.smi" % tmpfolder, "w") as f:
        f.write("%s\n" % smiles)
    log >> "%s -d wh -i t=smiles %s/tmp.smi %s/tmp.sdf" % (corina, tmpfolder, tmpfolder)
    log >> "%s -isdf %s/tmp.sdf -oxyz %s/tmp.xyz" % (babel, tmpfolder, tmpfolder)
    if not os.path.isfile("%s/tmp.xyz" % tmpfolder):
        if throw_error:
            raise RuntimeError("smiles -> xyz failed for '%s'" % smiles)
        return None
    else:
        config = readwrite.read("%s/tmp.xyz" % tmpfolder).pop()
        return config


def smiles_to_pseudo_extxyz(smiles):
    configs = []
    for idx, smi in enumerate(smiles):
        try:
            mol = chem.MolFromSmiles(smi)  # pylint: disable=E1101
            mol = chem.AddHs(mol)  # pylint: disable=E1101
        except Exception:
            print(f"Smiles problem in idx {idx} ,smiles string {smi}")
            configs.append(None)
            continue
        symbols = [a.GetSymbol() for a in mol.GetAtoms()]
        pos = np.zeros((len(symbols), 3))
        config = readwrite.ExtendedXyz(pos=pos, symbols=symbols)
        config.info["lmat"] = 1.0 * chem.GetAdjacencyMatrix(mol)  # pylint: disable=E1101
        configs.append(config)
    return configs


def dataframe_to_extxyz(
    data,
    smiles_from=None,
    tmpfolder="tmp",
    gen3d=False,
    corina="/path/to/corina",
    babel="/path/to/babel",
):
    if chem is None:
        raise ImportError("csv_to_extxyz requires rdkit")
    if smiles_from is None:
        row = next(data.iterrows())[1]
        smiles_from = get_smiles_key(row, True)
    errors = []
    configs = []
    log.debug = False
    try:
        for r, row in data.iterrows():
            log << log.back << "Convert row" << r << log.flush
            config = smiles_to_extxyz(
                smiles=[row[smiles_from]],
                gen3d=gen3d,
                tmpfolder=tmpfolder,
                corina=corina,
                babel=babel,
            )
            if config is None:
                errors.append(row)
            else:
                config.info.update(row)
                configs.append(config)
        log << log.endl
    except KeyboardInterrupt:
        return []
    return configs, errors


def git_hash():
    moduledir = os.path.dirname(__file__)
    command = ["git", "rev-parse", "--short", "HEAD"]
    o = subprocess.run(command, cwd=moduledir, capture_output=True)
    try:
        o.check_returncode()
        git_hash = o.stdout.decode().strip()
    except subprocess.CalledProcessError:
        git_hash = "git_hash_not_available"
    return git_hash
