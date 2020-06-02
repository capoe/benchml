from ..pipeline import Transform, Macro
from ..logger import Mock
import numpy as np
try:
    import asaplib
    import asaplib.data
except ImportError:
    asaplib = Mock()
    asaplib.data = None

def check_asap_available(obj, require=False):
    if asaplib.data is None:
        if require: raise ImportError("'%s' requires asap" % obj.__class__.__name__)
        return False
    return True

class AsapTransform(Transform):
    def check_available():
        return check_asap_available(AsapTransform)

class AsapXyz(AsapTransform):
    req_inputs = ("configs",)
    def _fit(self, inputs):
        asapxyz = asaplib.data.ASAPXYZ(
            frames=inputs["configs"], 
            periodic=inputs["meta"]["periodic"] \
                if "periodic" in inputs["meta"] else False)
        soap_js = {'soap1': {'type': 'SOAP',
                       'cutoff': 2.0,
                      'n': 2, 'l': 2,
                      'atom_gaussian_width': 0.2,
                      'rbf': 'gto', 'crossover': False}}

        acsf_js = {'acsf1': {'type': 'ACSF',
                            'cutoff': 2.0,
                            'g2_params': [[1, 1], [1, 2], [1, 3]],
                            'g4_params': [[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]]}}

        k2_js = { 'lmbtr-k2': {'type': 'LMBTR_K2',
             'k2':{
            "geometry": {"function": "distance"},
            "grid": {"min": 0, "max": 2, "n": 10, "sigma": 0.1},
            "weighting": {"function": "exponential", "scale": 0.5, "cutoff": 1e-3}},
             'periodic': False,
             'normalization': "l2_each"}}

        kernel_js = {}
        kernel_js['k1'] = {'kernel_type': 'moment_average',  
                                  'zeta': 2,
                                  'element_wise': False}
        kernel_js['k2'] = {'kernel_type': 'sum',  
                                  'element_wise': True}

        desc_spec_js = {'test_cm': {'type': "CM"},
                        'test_soap':{'atomic_descriptor':  soap_js, 'kernel_function': kernel_js},
                        'test_acsf': {'atomic_descriptor':  acsf_js, 'kernel_function': kernel_js},
                        'test_k2': {'atomic_descriptor':  k2_js, 'kernel_function': kernel_js}}

        # compute the descripitors
        print(inputs["meta"])
        asapxyz.compute_global_descriptors(desc_spec_js, sbs=[], keep_atomic=inputs["meta"]["per_atom"])

        # Design matrix
        # self.stream().put("X", asapxyz.get_descriptor("CM")) # <--
        # Kernel matrix
        # self.stream().put("K", asapxyz.get_kernel())

        # write selected descriptors
        asapxyz.write_computed_descriptors(prefix,['test_cm', 'test_soap'],[0])
        # write all
        asapxyz.write(prefix)
        asapxyz.save_state(tag)



        return
    def _map(self, inputs):
        return

