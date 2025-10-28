from benchml.logger import Mock

try:
    import nphil

    nphil.silent(True)
except ImportError:
    nphil = None


def check_nphil_available(obj, require=False):
    if nphil is None:
        if require:
            raise ImportError("%s requires nphil" % obj.__name__)
        return False
    return True


try:
    import gylm
except ImportError:
    gylm = Mock()
    gylm.SoapGtoCalculator = None
    gylm.GylmCalculator = None
    gylm.smooth_match = None
    gylm.evaluate_power = None


def check_gylmxx_available(obj, require=False):
    if gylm.GylmCalculator is None:
        if require:
            raise ImportError("%s requires gylmxx" % obj.__name__)
        return False
    return True


try:
    import asaplib.data
    import asaplib.hypers
except ImportError:
    asaplib = Mock()
    asaplib.hypers = None
    asaplib.data = None


def check_asap_available(obj, require=False):
    if asaplib.hypers is None:
        if require:
            raise ImportError("%s requires asaplib" % obj.__name__)
        return False
    return True


try:
    import dscribe
    import dscribe.descriptors as dd
except ImportError:
    dscribe = None
    dd = Mock()
    dd.CoulombMatrix = None
    dd.SineMatrix = None
    dd.EwaldSumMatrix = None
    dd.ACSF = None
    dd.SOAP = None
    dd.MBTR = None
    dd.LMBTR = None


def check_dscribe_available(obj, require=False):
    if dscribe is None:
        if require:
            raise ImportError("'%s' requires dscribe" % obj.__class__.__name__)
        return False
    return True


try:
    import rdkit.Chem as rchem
    from rdkit.Chem import AllChem as achem, Descriptors as rdesc
    from rdkit.Chem import rdFingerprintGenerator
except ImportError:
    rchem = None
    achem = None
    rdesc = None
    rdFingerprintGenerator = None


def check_rdkit_available(obj, require=False):
    if rchem is None or achem is None:
        if require:
            raise ImportError("'%s' requires rdkit" % obj.__class__.__name__)
        return False
    return True


try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = Mock()
    nn.Module = Mock


def check_torch_available(obj, require=False):
    if torch is None:
        if require:
            raise ImportError("'%s' requires rdkit" % obj.__class__.__name__)
        return False
    return True
