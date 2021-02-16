import numpy as np
import cantera as ct
from collections import namedtuple

State = namedtuple('State', ['frac', 'frac_t', 'rho', 'T', 'p'])
State.__new__.__defaults__ = (None, None, None)

'''
fid = open('temp_JICF.dat.sz.out','rb')
data = np.fromfile(fid,dtype='single')
data = np.reshape(data,[1280,896,576],order='C')
data_slice = data[:,:,288]
fid = open('temp_slice2.dat','wb')
data_slice.tofile(fid)
'''

def calc_stoic(gas, ox, fu):
    # Get species weight factors
    alpha = np.empty_like(gas.molecular_weights)
    for i, s in enumerate(gas.species_names):
        alpha[i] = (  2.0*gas.n_atoms(s, 'C')
                    + 0.5*gas.n_atoms(s, 'H')
                    - 1.0*gas.n_atoms(s, 'O'))
    alpha /= gas.molecular_weights

    # Oxidiser
    assert isinstance(ox, State)
    if ox.frac_t == 'mole':
        gas.X = ox.frac
    elif ox.frac_t == 'mass':
        gas.Y = ox.frac
    else:
        raise ValueError("Oxidiser fraction must be 'mass' of 'mole'")
    Y_ox = gas.Y

    # Fuel
    assert isinstance(fu, State)
    if fu.frac_t == 'mole':
        gas.X = fu.frac
    elif fu.frac_t == 'mass':
        gas.Y = fu.frac
    else:
        raise ValueError("Fuel fraction must be 'mass' of 'mole'")
    Y_fu = gas.Y

    # Get coupling function beta in fuel and oxidiser stream
    beta0 = np.dot(alpha, Y_ox)
    beta1 = np.dot(alpha, Y_fu)

    # Get stoichiometric mixture fraction
    Zst = (0.0 - beta0)/(beta1 - beta0)

    return Zst, beta0*1000.0, beta1*1000.0, alpha*1000.0, Y_ox


gas = ct.Solution('GRI_Red29.xml')
fu = State({'O2': 2.226056e-01, 'N2': 7.327526e-01, 'CH4': 4.464178e-02}, 'mass')
ox = State({'H2': 1.235707e-06, 'H': 3.633121e-08, 'O': 1.172828e-05, 'O2': 7.766151e-02, \
            'OH': 2.700374e-04, 'H2O': 8.199540e-02, 'HO2': 5.280324e-07, 'H2O2': 3.006126e-08, \
            'CH3': 2.562851e-25, 'CH4': 2.300946e-25, 'CO': 3.127669e-05, 'CO2': 1.002935e-01, \
            'CH2O': 2.438678e-16, 'CH3OH': 1.354947e-24, 'C2H2': 4.377876e-33,'C2H4': 2.620127e-39, \
            'C2H6':   8.051122e-47, 'CH2CO':  1.186053e-28, 'NH2':  1.230390e-13, 'NH3':  8.595266e-13, \
            'NO':  1.489855e-03, 'NO2':  3.606896e-06,  'N2O':  8.713956e-08, 'HCN':  2.408213e-17, \
            'HOCN':  7.119852e-17, 'HNCO':  1.062905e-13, 'NCO':  1.537852e-15, 'N2-1':  0.0, 'N2':   0.7382412},\
            'mass')
print(gas.species_names)
Zst, beta0, beta1, alpha, Y_ox  = calc_stoic(gas,ox,fu)

print(beta0, beta1)
fid = open('yspecies_JICF.dat','rb')
Yspecies = np.fromfile(fid,dtype='single')
fid.close()
Yspecies = np.reshape(Yspecies,[1280, 896, 576, 29],order='C')
#Yspecies = Yspecies[:,:,323,:]
#fid = open('N2_spec.dat','wb')
#Yspecies[:,:,288,28].tofile(fid)
#fid.close()
print(Yspecies.dtype, alpha.shape)
beta = np.dot(Yspecies, alpha.astype('float32'))
print(beta.shape, beta.dtype)
mixfrac = (beta-np.float32(beta0))/np.float32(beta1-beta0)
mixfrac[mixfrac<0]=0
mixfrac[mixfrac>1]=1.0
print(np.max(mixfrac))
fid = open('mixture_fraction_fortran2.dat','wb')
mixfrac.T.tofile(fid)
fid.close()
