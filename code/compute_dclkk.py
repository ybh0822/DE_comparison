import argparse
import math
import sys, platform, os
from numpy.linalg import inv
from scipy.integrate import quad
# from matplotlib import pyplot as plt
import numpy as np
#uncomment this if you are running remotely and want to keep in synch with repo changes
#if platform.system()!='Windows':
#    !cd $HOME/git/camb; git pull github master; git log -1
print('Using CAMB installed at '+ os.path.realpath(os.path.join(os.getcwd(),'..')))
sys.path.insert(0,os.path.realpath(os.path.join(os.getcwd(),'..')))
import camb
from camb import model, initialpower
from multiprocessing import Pool

# 1. Get noise curve
# 0 = CMBS4
# 1 = AdvACT

# noisetype = 0

# 2. Select redshift bins
# 0 = auto
# 1 = manual

bintype = 1

if bintype == 0:

    # Auto zlist
    zmax = 7.0
    dzbin = 2.0

    N = int(np.ceil(zmax/dzbin))
    zrange = np.zeros(N+1)

    zlist = np.arange(0, N*dzbin+dzbin, dzbin)

elif bintype == 1:

    #Manual zlist
    zlist = np.array([0., 0.2, 0.4, 0.6, 0.8, 1., 1.2, 1.4, 1.6, 1.8, 2., 2.3, 2.6, 3., 3.5, 4., 7., 100])
    N = len(zlist) - 1
    
    
# 3. Set ellmin for Cl_gg
ellmin_gg = 0

# 4. Set survey area
survarea = 18000

# 5. Linear bias model
# 0 = LSST science book
# 1 = Marcel
# 2 = Weinberg

bias = 1

def biasg(zs):
    if bias == 0:
        btemp = 0.95/np.sqrt(PK2.P(zs, k1, grid=False)/PK2.P(0, k1, grid=False))
    elif bias == 1:
        btemp = 1 + zs
    elif bias == 2:
        btemp = 1 + 0.84*zs
    return btemp

# 6. Set labels
labels = ["z = 0 - 0.5", "z = 0.5 - 1", "z = 1 - 2", "z = 2 - 3", "z = 3 - 4", "z = 4 - 7", "z = 7 - 100"]

# 7. fsky
fsky = 0.5

# 8. dndz type
dndztype = 2
# 9. Integration pts
acc_npts = 300002



    
print 'zrange: ', zlist
print 'Number of bins: ', N

print 'ell_min for Cl_gg: ', ellmin_gg

print 'Survey area: ', survarea, 'deg^2'

if bias == 0:
    print 'b(z) = 0.95/Growth(z)'
elif bias == 1:
    print 'b(z) = 1 + z'
elif bias == 2:
    print 'b(z) = 1 + 0.84z'
    
def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

print 'Number of auto-spectra: ', N + 1
print 'Number of cross-spectra: ', nCr(N, 2) + N

print 'fsky = ', fsky


#dndz

if dndztype == 0:
    z0 = 0.3

    Ntotal = 40.
    def pz_func(zs):
        return 1/(2*z0)*((zs/z0)**2)*np.exp(-zs/z0)

    ndensity = np.ones(N)*np.inf

if dndztype == 1:
    z0 = 0.3

    Ntotal = 40.
    def pz_func(zs):
        return 1/(2*z0)*((zs/z0)**2)*np.exp(-zs/z0)

    ndensity = []

    for i in range(N):
        aa = Ntotal*quad(pz_func, zlist[i], zlist[i+1])[0]
        ndensity.append(aa)

    print 'n_gal in each bin (gal/arcmin^2): ', np.round(ndensity, 3)
    print 'total n_gal in all bins:', sum(ndensity)

    ndensity = np.array(ndensity)
    print 'N_gal in each bin (in billion): ', np.round(3600*survarea*ndensity/1.e9, 3)
    print 'N_total: ', np.round(sum(3600*survarea*ndensity)/1.e9, 3), 'billion gals'

if dndztype == 2:

    import csv

    z_Marcel = []
    dndz_Marcel = []

    with open('dndz_LSST_i27_SN5_3y.csv') as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            z_Marcel.append(row[0])
            dndz_Marcel.append(row[1])

    from scipy.interpolate import interp1d

    z_Marcel = np.array(z_Marcel, dtype = np.float128)
    dndz_Marcel = np.array(dndz_Marcel, dtype = np.float128)

    z_Marcel[-1] = 1100.

    z_Marcel = np.append(0, z_Marcel)
    dndz_Marcel = np.append(0, dndz_Marcel)

    f_dndz = interp1d(z_Marcel, dndz_Marcel, kind='linear')

    ndensity = []

    for i in range(N):
        aa = quad(f_dndz, zlist[i], zlist[i+1])[0]
        ndensity.append(aa)
        
    print 'n_gal in each bin (gal/arcmin^2): ', np.round(ndensity, 3)
    print 'total n_gal in all bins:', sum(ndensity)

    ndensity = np.array(ndensity)
    print 'N_gal in each bin (in billion): ', np.round(3600*survarea*ndensity/1.e9, 3)
    print 'N_total: ', np.round(sum(3600*survarea*ndensity)/1.e9, 3), 'billion gals'


def getCl(l):
    
    global chis
    global zs
    global dzs
    global pars
    global K_k
    global om_tot
    global w_DE
    global w1
    global kmax
    global PK_kk

    k=(l+0.5)/chis
    w1[:]=1
    w1[k<1e-4]=0
    w1[k>=kmax]=0
    cl_kk = np.dot(dzs, 1/(2.99792*(10**5))*w1*K_k*K_k*PK_kk.P(zs, k, grid=False)*(pars.H0*np.sqrt(om_tot*(1+zs)**3 + 4.2*(10**(-5))/((pars.H0/100.)**2)*(1+zs)**4 + (1-om_tot)*(1+zs)**(3*(1+w_DE)) ))/(chis**2))
    return cl_kk


def Cl_kk(theta, acc_npts):
    
    global chis
    global zs
    global dzs
    global pars
    global K_k
    global om_tot
    global w_DE
    global w1
    global kmax
    global PK_kk
    
    H0, ombh2, omch2, ns, As, mnu, w_DE, tau, wa = theta

    #Set up a new set of parameters for CAMB
    pars = camb.CAMBparams()
    #This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
    pars.set_accuracy(AccuracyBoost=3.0, lSampleBoost=3.0, lAccuracyBoost=3.0)
    pars.set_cosmology(H0=H0, cosmomc_theta = None, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=0, tau=tau, nnu=3.046, standard_neutrino_neff=3.046, num_massive_neutrinos = 1, neutrino_hierarchy='degenerate')
    pars.InitPower.set_params(ns=ns, r=0, As=As)
    pars.set_for_lmax(3000, lens_potential_accuracy=3, max_eta_k=30*3000)
    
    pars.set_dark_energy(w=w_DE,wa=wa,dark_energy_model='ppf') 

    #calculate results for these parameters
    results = camb.get_results(pars)
    
    om_tot = ( ombh2+omch2+mnu/94.07*(3.046/3.0)**0.75 )/((pars.H0/100)**2)


    nz = acc_npts #number of steps to use for the radial/redshift integration
    kmax=10  #kmax to use
    # #First set up parameters as usual
    # pars = camb.CAMBparams()
    # pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
    # pars.InitPower.set_params(ns=0.965)

    #For Limber result, want integration over \chi (comoving radial distance), from 0 to chi_*.
    #so get background results to find chistar, set up arrage in chi, and calculate corresponding redshifts
    results= camb.get_background(pars)
    chistar = results.conformal_time(0)- results.tau_maxvis
    chis = np.linspace(0,chistar,nz)
    zs=results.redshift_at_comoving_radial_distance(chis)
    #Calculate array of delta_chi, and drop first and last points where things go singular
    dchis = (chis[2:]-chis[:-2])/2
    dzs = (zs[2:]-zs[:-2])/2
    chis = chis[1:-1]
    zs = zs[1:-1]

    #Get the matter power spectrum interpolation object (based on RectBivariateSpline). 
    #Here for lensing we want the power spectrum of the Weyl potential.
    PK_kk = camb.get_matter_power_interpolator(pars, nonlinear=False, 
        hubble_units=False, k_hunit=False, kmax=kmax, var1=model.Transfer_tot, var2=model.Transfer_tot,
        zmax=zs[-1])
        

    ls = np.arange(2,2001+1, dtype=np.float64)

    w1 = np.ones(chis.shape)
    const = 3*om_tot*(pars.H0**2)/2/(pars.H0*np.sqrt(om_tot*(1+zs)**3 + 4.2*(10**(-5))/((pars.H0/100.)**2)*(1+zs)**4 + (1-om_tot)*(1+zs)**(3*(1+w_DE)) ))/(2.99792*(10**5))
    win = ((chistar-chis)/(chistar))
    K_k = const*(1+zs)*win*chis

    pool = Pool(processes = 32) 
    cl_kk = []
    cl_kk = pool.map(getCl, ls)
    
    pool.close() 
    pool.join()
    
    pars.set_dark_energy() 
    return cl_kk
    

def Gradient_kk(theta0, h, parindex, acc_npts):
    # Get partial derivatives
    ndim = len(theta0)
    df_dtheta = []
    # for i in range(ndim):
    dh = np.zeros_like(h)
    i = int(parindex)
    dh[i] = h[i]/2.
    f = Cl_kk(theta0 - dh, acc_npts)
    f_h = Cl_kk(theta0 + dh, acc_npts)
    df_dtheta.append( (np.array(f_h)-np.array(f))/h[i] )
    df_dtheta = np.array(df_dtheta)
    return df_dtheta

# Set fiducial parameters from Planck 2015 Cosmological Parameter paper
H0 = 67.0
ombh2  = 0.022
omch2  = 0.1194
ns     = 0.96
As     = 2.2e-9
mnu    = 0.06
w_DE   = -1
tau    = 0.06
wa     = 0

theta = np.array([H0, ombh2, omch2, ns, As, mnu, w_DE, tau, wa])

ls = np.arange(2,2001+1, dtype=np.float64)

h = np.array([0.5, 0.0008, 0.0030, 0.01, 0.1e-9, 0.02, 0.05, 0.02, 0.05])




def compute_power(ns):

    h = np.array([0.5, 0.0008, 0.0030, 0.01, 0.1e-9, 0.02, 0.05, 0.02, 0.05])
    dClkk_dpar = Gradient_kk(theta, h, ns.pindex, acc_npts)

    if ns.pindex == 4:
        dClkk_dpar = dClkk_dpar/10**9

    # print np.shape(np.array(dClkk_dpar))
    # print np.shape(np.array(dClkk_dpar).flatten())

    DataOut = np.column_stack((np.array(dClkk_dpar).flatten()))
    np.savetxt('/global/homes/y/ybh0822/TomoDelens/r300_ppf_N17_lin/13test%d.dat'%(ns.pindex), DataOut)

if __name__ == '__main__':

    desc = 'compute the power spectrum of the BOSS DR12 combined sample'
    parser = argparse.ArgumentParser(description=desc)

    h = 'the mock sample number'
    parser.add_argument('pindex', type=int, choices=list(range(0, 10)), help=h)

    ns = parser.parse_args()
    compute_power(ns)





