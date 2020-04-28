
class PeriodicTable(object):
    element_elneg = [ 0.0, 
        2.3, 4.16, 
        0.912, 1.576, 2.051, 2.544, 3.066, 3.610, 4.193, 4.789,
        0.869, 1.293, 1.613, 1.916, 2.253, 2.589, 2.869, 3.242,
        0.734, 1.034, 1.19, 1.38, 1.53, 1.65, 1.75, 1.80, 1.84, 1.88, 1.85, 1.59, 1.756, 1.994, 2.211, 2.434, 2.689, 2.966,
        0.706, 0.963, 1.12, 1.32, 1.41, 1.47, 1.51, 1.54, 1.56, 1.59, 1.87, 1.52, 1.656, 1.824, 1.984, 2.158, 2.359, 2.582,
        0.659, 0.881,    
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # Lanthanides     
        1.16, 1.34, 1.47, 1.60, 1.65, 1.68, 1.72, 1.92, 1.76, 1.789, 1.854, 2.01,  2.19,  2.39,  2.60,
        0.67,  0.90,    
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # Actinides
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ] # Allen scale
    element_names = ["?",
       "H","He",
       "Li","Be","B","C","N","O","F","Ne",
       "Na","Mg","Al","Si","P","S","Cl","Ar",
       "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge", "As","Se","Br","Kr",
       "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn", "Sb","Te","I","Xe",
       "Cs","Ba",
       "La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
       "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn",
       "Fr","Ra",
       "Ac","Th","Pa","U","Np","Pu","Am","Cm", "Bk","Cf","Es","Fm","Md","No","Lr",
       "Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Uub", "Uut","Uuq","Uup","Uuh" ]
    element_mass = [-1., 
        1.00794, 4.00260, 6.941, 9.012187, 10.811, 12.0107, 14.00674, 15.9994, 18.99840, 20.1797, 22.98977,  
        24.3050, 26.98154, 28.0855, 30.97376, 32.066, 35.4527, 39.948, 39.0983, 40.078, 44.95591, 47.867,      
        50.9415, 51.9961, 54.93805, 55.845, 58.93320, 58.6934, 63.546, 65.39, 69.723, 72.61, 74.92160, 78.96,  
        79.904, 83.80, 85.4678, 87.62, 88.90585, 91.224, 92.90638, 95.94, 98.0, 101.07, 102.90550, 106.42,     
        107.8682, 112.411, 114.818, 118.710, 121.760, 127.60, 126.90447, 131.29, 132.90545, 137.327, 138.9055, 
        140.116, 140.90765, 144.24, 145.0, 150.36, 151.964, 157.25, 158.92534, 162.50, 164.93032, 167.26,      
        168.93421, 173.04, 174.967, 178.49, 180.9479, 183.84, 186.207, 190.23, 192.217, 195.078, 196.96655,    
        200.59, 204.3833, 207.2, 208.98038, 209.0, 210.0, 222.0, 223.0, 226.0, 227.0, 232.0381, 231.03588,     
        238.0289, 237.0, 244.0, 243.0, 247.0, 247.0, 251.0, 252.0, 257.0, 258.0, 259.0, 262.0, 261.0, 262.0,   
        263.0, 264.0, 265.0, 268.0, 271.0, 272.0, 285.0, 284.0, 289.0, 288.0, 292.0 ]
    element_covrad = [-1.,
        0.320,0.310,1.630,0.900,0.820,0.770,0.750,0.730,0.720,0.710,1.540,1.360,1.180,1.110,1.060,1.020,     
        0.990,0.980,2.030,1.740,1.440,1.320,1.220,1.180,1.170,1.170,1.160,1.150,1.170,1.250,1.260,1.220,1.200, 
        1.160,1.140,1.120,2.160,1.910,1.620,1.450,1.340,1.300,1.270,1.250,1.250,1.280,1.340,1.480,1.440,1.410, 
        1.400,1.360,1.330,1.310,2.350,1.980,1.690,1.650,1.650,1.840,1.630,1.620,1.850,1.610,1.590,1.590,1.580, 
        1.570,1.560,2.000,1.560,1.440,1.340,1.300,1.280,1.260,1.270,1.300,1.340,1.490,1.480,1.470,1.460,1.460, 
        2.000,2.000,2.000,2.000,2.000,1.650,2.000,1.420,2.000,2.000,2.000,2.000,2.000,2.000,2.000,2.000,2.000, 
        2.000,2.000,2.000,2.000,2.000,2.000,2.000,2.000,2.000,2.000,2.000,2.000,2.000,2.000,2.000 ]
    element_valence = [-1.,
          1,-1, 1, 2, 3, 4, 3, 2, 1,-1, 1, 2, 3, 4, 3, 2,     
          1,-1, 1, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,     
         -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,     
         -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,     
         -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,     
         -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,     
         -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,     
         -1,-1,-1,-1]
    def __init__(self):
        self.elements_by_z = {}
        self.elements_by_name = {}
    def __getitem__(self, key):
        if type(key) == str:
            return self.elements_by_name[key]
        elif type(key) == int:
            return self.elements_by_z[key]
        else:
            raise RuntimeError("Invalid element identifier '%s'" % key)
    def setup(self):
        element_z = [ i for i in range(len(self.element_names)) ]
        assert len(self.element_names) == len(self.element_mass) == \
            len(self.element_covrad) == len(self.element_valence)
        for z, name, mass, covrad, elneg, valence in zip(
                element_z, 
                self.element_names, 
                self.element_mass, 
                self.element_covrad, 
                self.element_elneg, 
                self.element_valence):
            name = name.strip()
            self.addElement(z, name, mass, covrad, elneg, valence)
        return self
    def addElement(self, z, name, mass, covrad, elneg, valence):
        elem = AtomicElement(z, name, mass, covrad, elneg, valence)
        self.elements_by_z[z] = elem
        self.elements_by_name[name] = elem
        return
    def getPropertyDict(self, key, convert = lambda v: v):
        props = {}
        for name, elem in self.elements_by_name.items():
            props[name] = convert(elem[key])
        return props

class AtomicElement(object):
    def __init__(self, z, name, mass, covrad, elneg, valence):
        self.z = z
        self.name = name
        self.mass = mass
        self.covrad = covrad
        self.elneg = elneg
        self.valence = valence
        self.property_dict = { 
            'z' : z, 
            'covrad' : covrad, 
            'name': name, 
            'mass': mass, 
            'elneg': elneg, 
            'valence': valence }
    def __getitem__(self, key):
        return self.property_dict[key]

lookup = PeriodicTable().setup()

