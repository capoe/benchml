"""
model classfications for molecular datasets
"""

import matplotlib.cm as cm

color_dict = { 'acsf': cm.tab10(0),
               'soap': cm.tab10(1),
               'cm': cm.tab10(2),
               'gylm': cm.tab10(3),
               'mbtr': cm.tab10(4),
               'pdf': cm.tab10(5),
               'physchem': cm.tab10(6),
               'esm': cm.tab10(7),
               'ecfp': cm.tab10(8)
             }


def base2model(base_list, mode_list):
    return [ a+'_'+m for a in base_list for m in mode_list]

def addprefix(base_list, prefix):
    return [ prefix+'_'+a for a in base_list]

bmol = {}

bmol_acsf_base = [ 'bmol_acsf_longrange', 
                   'bmol_acsf_minimal', 
                   'bmol_acsf_smart']
bmol['acsf'] = base2model(base2model(bmol_acsf_base, ['ext','int']), ['rr','krr'])

bmol_cm_base = ['bmol_cm_eigenspectrum', 'bmol_cm_sorted_l2' ]
bmol['cm'] = base2model(bmol_cm_base, ['rr','krr'])


bmol_ecfp_base = ['bmol_ecfp4', 'bmol_ecfp6' ]
bmol['ecfp'] = base2model(bmol_ecfp_base, ['rr','krr'])


bmol_gylm_base = [ 'bmol_gylm_minimal', 
                   'bmol_gylm_standard']
bmol['gylm'] = base2model(base2model(bmol_gylm_base, ['ext','int']), ['rr','krr'])


bmol_mbtr_base = [ 'bmol_mbtr']
bmol['mbtr'] = base2model(base2model(bmol_mbtr_base, ['ext','int']), ['rr','krr'])


bmol_pdf_base = ['bmol_pdf_gylm_minimal', 
                 'bmol_pdf_gylm_standard', 
                 'bmol_pdf_soap_minimal',
                 'bmol_pdf_soap_standard']
bmol['pdf'] = base2model(bmol_pdf_base, ['rr','krr'])

bmol_physchem_base = ['bmol_physchem_basic', 
                      'bmol_physchem_core', 
                      'bmol_physchem_extended',
                      'bmol_physchem_logp']
bmol['physchem'] = base2model(bmol_physchem_base, ['rr','rf'])

bmol_soap_base = [ 'bmol_soap_longrange', 
                   'bmol_soap_minimal', 
                   'bmol_soap_smart']
bmol['soap'] = base2model(base2model(base2model(bmol_soap_base,['cross','nocross']), ['ext','int']), ['rr','krr'])

"""
model classification for bulk datasets
"""

bxtal = {}

bxtal_acsf_base = ['bxtal_acsf_longrange', 
                   'bxtal_acsf_minimal', 
                   'bxtal_acsf_smart']
bxtal['acsf'] = base2model(base2model(bxtal_acsf_base, ['ext','int']), ['rr','krr'])

bxtal_esm_base = ['bxtal_esm_eigenspectrum', 'bxtal_esm_sorted_l2' ]
bxtal['esm'] = base2model(bxtal_esm_base, ['rr','krr'])



bxtal_gylm_base = ['bxtal_gylm_minimal', 
                   'bxtal_gylm_standard']
bxtal['gylm'] = base2model(base2model(bxtal_gylm_base, ['ext','int']), ['rr','krr'])


bxtal_mbtr_base = [ 'bxtal_mbtr']
bxtal['mbtr'] = base2model(base2model(bxtal_mbtr_base, ['ext','int']), ['rr','krr'])


bxtal_pdf_base = ['bxtal_pdf_gylm_minimal', 
                  'bxtal_pdf_gylm_standard', 
                  'bxtal_pdf_soap_minimal',
                  'bxtal_pdf_soap_standard']
bxtal['pdf'] = base2model(bxtal_pdf_base, ['rr','krr'])

bxtal_physchem_base = ['bxtal_physchem_s05', 'bxtal_physchem_s10', 'bxtal_physchem_s20']
bxtal['physchem'] = base2model(bxtal_physchem_base, ['rr','rf'])

bxtal_soap_base = ['bxtal_soap_longrange', 
                   'bxtal_soap_minimal', 
                   'bxtal_soap_smart']
bxtal['soap'] = base2model(base2model(base2model(bxtal_soap_base,['cross','nocross']), ['ext','int']), ['rr','krr'])
