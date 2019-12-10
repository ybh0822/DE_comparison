
# coding: utf-8

# In[1]:


import math
import sys, platform, os
from numpy.linalg import inv
from scipy.integrate import quad
from matplotlib import pyplot as plt
import numpy as np
#uncomment this if you are running remotely and want to keep in synch with repo changes
#if platform.system()!='Windows':
#    !cd $HOME/git/camb; git pull github master; git log -1
print('Using CAMB installed at '+ os.path.realpath(os.path.join(os.getcwd(),'..')))
sys.path.insert(0,os.path.realpath(os.path.join(os.getcwd(),'..')))
import camb
from camb import model, initialpower
from orphics.stats import FisherMatrix

font = {'size' : 14, 'family' : 'serif', 'serif' : 'cm'}
plt.rc('font', **font)
plt.rcParams['text.usetex'] = True

params = {'legend.fontsize': 12}
plt.rcParams.update(params)

kmax_cut = 0.3*0.6751
kmin_cut = 0.0*0.6751


w0_kgTE = []
wa_kgTE = []
mnu_kgTE = []
FoM_kgTE = []

w0_kgTE_mnufixed = []
wa_kgTE_mnufixed = []
FoM_kgTE_mnufixed = []

w0_DESI = []
wa_DESI = []
mnu_DESI = []
FoM_DESI = []

w0_DESI_mnufixed = []
wa_DESI_mnufixed = []
FoM_DESI_mnufixed = []

Pw0_kgTE = []
Pwa_kgTE = []
Pmnu_kgTE = []
PFoM_kgTE = []

Pw0_kgTE_mnufixed = []
Pwa_kgTE_mnufixed = []
PFoM_kgTE_mnufixed = []

Pw0_DESI = []
Pwa_DESI = []
Pmnu_DESI = []
PFoM_DESI = []

Pw0_DESI_mnufixed = []
Pwa_DESI_mnufixed = []
PFoM_DESI_mnufixed = []


params = []
params.append('H0')
params.append('ombh2')
params.append('omch2')
params.append('ns')
params.append('As')
params.append('mnu')
params.append('w')
params.append('tau')
params.append('wa')

# FisherS4 = np.loadtxt('FullS4Fisher_S4_CMB_nT1.4_nP2.12.txt')
FisherS4 = np.loadtxt('ppf_S4_nophi.txt')
FisherS4 = FisherMatrix( FisherS4, params )

FullPlanck1 = np.loadtxt('ppf_nophi_LowEllPlanck.txt')
FullPlanck1 = FisherMatrix( FullPlanck1, params )
FullPlanck2 = np.loadtxt('ppf_nophi_HighEllPlanck_fsky_0.65.txt')
FullPlanck2 = FisherMatrix( FullPlanck2, params )

FisherP_addS4 = np.loadtxt('ppf_nophi_HighEllPlanck_fsky_0.25.txt')
FisherP_addS4 = FisherMatrix( FisherP_addS4, params )

FisherPS4 = FisherS4 + FisherP_addS4 + FullPlanck1


params = []
params.append('H0')
params.append('ombh2')
params.append('omch2')
params.append('ns')
params.append('As')
params.append('mnu')
params.append('w')
params.append('tau')
params.append('wa')
params.append('omk')

FisherBAO = np.loadtxt('Fisher_DESIBAO_omk_wa.txt')
FisherBAO = FisherMatrix( FisherBAO, params )

FisherBAO.delete('omk')



#Set up a new set of parameters for CAMB
pars = camb.CAMBparams()
#This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency

H0 = 67.51
ombh2  = 0.02226
omch2  = 0.1193
ns     = 0.9653
As     = 2.130e-9
mnu    = 0.06
w_DE   = -1
tau    = 0.063

# pars.set_accuracy(AccuracyBoost=3.0, lSampleBoost=3.0, lAccuracyBoost=3.0)
pars.set_cosmology(H0=H0, cosmomc_theta = None, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=0, tau=tau, nnu=3.046, standard_neutrino_neff=3.046, neutrino_hierarchy='normal')
pars.InitPower.set_params(ns=ns, r=0, As=As)
pars.set_for_lmax(2500, lens_potential_accuracy=2)

#calculate results for these parameters
results = camb.get_results(pars)
results= camb.get_background(pars)



# newnumdensity = np.array([1, 5, 10, 30, 60, 90, 120, 200, 500, 10000])
# newnumdensity = np.array([66])

# newnumdensity = np.array([1])

for iteration in range(16)[15:]:

    fsky = 0.4
    bintype = 1
    noisetype = 11
    dndztype = 2


    if bintype == 0:

        # Auto zlist
        zmax = 7.0
        dzbin = 2.0

        N = int(np.ceil(zmax/dzbin))
        zrange = np.zeros(N+1)

        zlist = np.arange(0, N*dzbin+dzbin, dzbin)

    elif bintype == 1:

        #Manual zlist
        zlist = np.array([0., 0.2, 0.4, 0.6, 0.8, 1., 1.2, 1.4, 1.6, 1.8, 2., 2.3, 2.6, 3., 3.5, 4., 7.])[0:iteration+2]
        # zlist = np.array([0, 0.5, 1., 2.])
        N = len(zlist) - 1
        
        
    # 3. Set ellmin for Cl_gg
    ellmin_gg = 0

    # 4. Set survey area
    survarea = 18000

    # 5. Linear bias model
    # 0 = LSST science book
    # 1 = Marcel
    # 2 = Weinberg

    # 6. Set labels
    labels = ["z = 0 - 0.5", "z = 0.5 - 1", "z = 1 - 2", "z = 2 - 3", "z = 3 - 4", "z = 4 - 7", "z = 7 - 100"]

    # 7. fsky

    # 8. dndz type

    #dndz

    if dndztype == 0:
        z0 = 0.3

        Ntotal = 40.
        def pz_func(zs):
            return 1/(2*z0)*((zs/z0)**2)*np.exp(-zs/z0)

        ndensity = np.ones(N)*np.inf

    if dndztype == 1:

        Ntotal = 30.
        def pz_func(zs):
            return (zs**1.2)*np.exp(-zs/0.5)/0.239794

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



    Y = np.loadtxt('test_cl3.dat')
    cl_kk = Y

    cl_kki = []
    for i in range(N):
        Y = np.loadtxt('test_cl0_%d.dat'%(i))
        cl_kki.append( Y )
        
    cl_kg = []
    for i in range(N):
        Y = np.loadtxt('test_cl1_%d.dat'%(i))
        cl_kg.append( Y )

    cl_gg = []
    for i in range(N):
        Y = np.loadtxt('test_cl2_%d.dat'%(i))
        cl_gg.append( Y )

    zcenter = (zlist[1:] + zlist[0:N])/2

    cl_gg = np.array(cl_gg)
    cl_kg = np.array(cl_kg)



    # new = np.zeros(cl_gg.shape)
    # for uq in range(N):
    #     lmax = int( kmax_cut*results.comoving_radial_distance(zcenter[uq]) )
    #     new[uq,:(lmax-1)] = cl_gg[uq,:(lmax-1)]
    #     # if lmax < 2000:
    #     #     new[uq,(lmax-1):] = 1./(ndensity[uq]/(8.462*10**-8))
    # cl_gg = new

    # new = np.zeros(cl_kg.shape)
    # for uq in range(N):
    #     lmax = int( kmax_cut*results.comoving_radial_distance(zcenter[uq]) )
    #     new[uq,:(lmax-1)] = cl_kg[uq,:(lmax-1)]
    # cl_kg = new

    # for i in range(N):
    #     if newnumdensity[iteration] == 10000:
    #         cl_gg[i] = cl_gg[i] - 1./(ndensity[i]/(8.462*10**-8))
    #     else:
    #         cl_gg[i] = cl_gg[i] - 1./(ndensity[i]/(8.462*10**-8)) + 1./(newnumdensity[iteration]/sum(ndensity)*ndensity[i]/(8.462*10**-8))



    ls = np.arange(2,2001+1, dtype=np.float64)
    #CMB NOISE
    if noisetype == 0:
        print 'Zero noise'
        Nl_kk = np.zeros(ls.shape)
    elif noisetype == 1:
        print 'CMBS4 noise'
        import pickle
        noise = pickle.load(open('lensNoisePowerAdvACT.pkl'))
        Nl_kk = noise[1][0:2000]*cl_kk[1200-2]/noise[1][1200-2]
    elif noisetype == 2:
        print 'AdvACT noise'
        import pickle
        noise = pickle.load(open('lensNoisePowerAdvACT.pkl'))
        Nl_kk = noise[1][0:2000]
    elif noisetype == 3:
        print 'SO test full'
        Y = np.loadtxt("kappa_noise_SOtest_full.dat")
        Nl_kk = Y[:,1]
    elif noisetype == 4:
        print 'SO test half'
        Y = np.loadtxt("kappa_noise_SOtest_half.dat")
        Nl_kk = Y[:,1]
    elif noisetype == 11:
        print 'S4 manual'
        Y = np.loadtxt('CMBS4_MV_lensnoise_1muK.dat')
        Nl_kk = Y[0:2000,1]
    elif noisetype == 12:
        print 'S3 manual'
        Y = np.loadtxt('CMBS3_MV_lensnoise_5muK.dat')
        Nl_kk = Y[0:2000,1]
    elif noisetype == 13:
        print 'S4 manual ver 2'
        Y = np.loadtxt('CMBS4_MV_lensnoise_1muK_test.dat')
        Nl_kk = Y[0:2000,1]
    elif noisetype == 17:
        print '7muK survey'
        Y = np.loadtxt('CMBS3_MV_lensnoise_7muK_test.dat')
        Nl_kk = Y[0:2000,1]
    elif noisetype == 20:
        print 'SO v3'
        Y = np.loadtxt("xcorr_mv_L_Nlkk_SENS2_16000.txt")
        lss = Y[:,0]
        Nl_kk = Y[:,1]

        from scipy.interpolate import interp1d
        Nl_kk_int = interp1d(lss, Nl_kk, kind='linear')

        Nl_kk = np.append(np.zeros(13), Nl_kk_int(ls[13:]))

    elif noisetype == 30:
        print 'S4 working group'
        nlkk = np.loadtxt('CMB_S4_noise_coadd.txt')
        Nl_kk = nlkk[:,0]*(nlkk[:,0]+1)*nlkk[:,3]/4
        Nl_kk = Nl_kk[0:2000]

    elif noisetype == 50:
        print 'Blake proposal - AdvACT noise'
        import pickle
        noise = pickle.load(open('noisePowerREDO9.0uK.pkl'))
        Nl_kk = noise[1][0:2000]
        
    # plt.figure(figsize = (10, 7))
    # plt.loglog(ls, cl_kk, 'r--', label = '$C_l^{\kappa \kappa}$')
    # plt.loglog(ls, Nl_kk, 'b--', label = 'Noise')
    # # plt.loglog(ls, Nl_kk2, 'y:', label = 'Half Noise')
    # plt.loglog(ls, Nl_kk+cl_kk, 'g-', label = '$C_l^{\kappa \kappa}$ + $N_l^{\kappa \kappa}$')
    # plt.xlim(2, 2000)
    # plt.legend()
    # plt.show()

    #Add noise
    cl_kk = cl_kk + Nl_kk


    # Set fiducial parameters from Planck 2015 Cosmological Parameter paper
    cosmomc_theta = .010409
    ombh2  = 0.02226
    omch2  = 0.1193
    ns     = 0.9653
    As     = 2.130e-9
    mnu    = 0.06
    w_DE   = -1
    tau    = 0.063
    wa     = 0

    theta = np.array([cosmomc_theta, ombh2, omch2, ns, As, mnu, w_DE, tau, wa])



    # In[2]:



    ndim = len(theta)
    dClkk_dpar = []
    for i in range(ndim):
        Y = np.loadtxt('13test%d.dat'%(i))
        dClkk_dpar.append( Y )
    dClkk_dpar = np.array(dClkk_dpar)


    # In[3]:


    # tdClkg_dpar = []
    # for i in range(ndim):
    #     temp = []
    #     for j in range(N):
    #         Y = np.loadtxt('test1_%d_%d.dat'%(j, i))
    #         temp.append( Y )
    #     tdClkg_dpar.append( temp )
    # tdClkg_dpar = np.array(tdClkg_dpar)


    # # In[4]:


    # tdClgg_dpar = []
    # for i in range(ndim):
    #     temp = []
    #     for j in range(N):
    #         Y = np.loadtxt('test2_%d_%d.dat'%(j, i))
    #         temp.append( Y )
    #     tdClgg_dpar.append( temp )
    # tdClgg_dpar = np.array(tdClgg_dpar)

    
    # zcenter = (zlist[1:] + zlist[0:N])/2

    # dClgg_dpar = np.zeros(tdClgg_dpar.shape)
    # for uq in range(N):
    #     lmax = int( kmax_cut*results.comoving_radial_distance(zcenter[uq]) )
    #     dClgg_dpar[:,uq,:(lmax-1)] = tdClgg_dpar[:,uq,:(lmax-1)]


    # dClkg_dpar = np.zeros(tdClkg_dpar.shape)
    # for uq in range(N):
    #     lmax = int( kmax_cut*results.comoving_radial_distance(zcenter[uq]) )
    #     dClkg_dpar[:,uq,:(lmax-1)] = tdClkg_dpar[:,uq,:(lmax-1)]



    dClkg_dpar = []
    for i in range(ndim):
        temp = []
        for j in range(N):
            Y = np.loadtxt('test1_%d_%d.dat'%(j, i))
            temp.append( Y )
        dClkg_dpar.append( temp )
    dClkg_dpar = np.array(dClkg_dpar)


    # In[4]:


    dClgg_dpar = []
    for i in range(ndim):
        temp = []
        for j in range(N):
            Y = np.loadtxt('test2_%d_%d.dat'%(j, i))
            temp.append( Y )
        dClgg_dpar.append( temp )
    dClgg_dpar = np.array(dClgg_dpar)

    

    # dClkk_dpar = np.zeros(tdClkk_dpar.shape)
    # for uq in range(len(zcenter)):
    #     lmax = int( kmax_cut*results.comoving_radial_distance(zcenter[uq]) )
    #     dClkk_dpar[:,uq,:(lmax-1)] = tdClkk_dpar[:,uq,:(lmax-1)]

    # In[5]:


    dClkk_dpar[7,:] = np.zeros(dClkk_dpar[7,:].shape)
    dClkg_dpar[7,:,:] = np.zeros(dClkg_dpar[7,:,:].shape)
    dClgg_dpar[7,:,:] = np.zeros(dClkg_dpar[7,:,:].shape)



    print np.shape(dClkk_dpar)
    print np.shape(dClkg_dpar)
    print np.shape(dClgg_dpar)


    # In[ ]:


    # In[6]:


    uy = 4
    dClkk_dpar[uy] = dClkk_dpar[uy]*(10**9)
    dClkg_dpar[uy] = dClkg_dpar[uy]*(10**9)
    dClgg_dpar[uy] = dClgg_dpar[uy]*(10**9)


    # In[7]:


    def nCr(n,r):
        f = math.factorial
        return f(n) / f(r) / f(n-r)

    if N == 1:
        Ngg = 0
    else:
        Ngg = nCr(N,2)
    MatSize = Ngg
    indMat = [np.zeros([2,2]) for i in range(MatSize)]

    print Ngg




    lmaxlist = []
    for uq in range(N):
        lmaxlist.append( int( kmax_cut*results.comoving_radial_distance(zcenter[uq]) ) )
    lmaxlist = np.array(lmaxlist)
    lmaxlist = np.array([0, 169, 320, 454, 572, 676, 768, 851, 925, 992, 1053, 1135, 1206, 1290, 1378, 1453])

    lminlist = []
    for uq in range(N):
        lminlist.append( int( kmin_cut*results.comoving_radial_distance(zcenter[uq]) ) )
    lminlist = np.array(lminlist)

    print lminlist
    print lmaxlist



    invCovlist = []
    Covlist = []

    lmin = 50
    Ngen = len(zlist) - 1
    lmax = 2000
    npar = 2*Ngen+9

    FisherM = np.zeros([npar,npar])


    for u in range(len(ls))[lmin-2:lmax-1]:
        # print u
        # Ngen = len(zlist)-1

        NN = sum(ls[u]*np.ones(lmaxlist[0:Ngen].shape) < lmaxlist[0:Ngen])
        skip = Ngen-NN

        NN = sum(ls[u]*np.ones(lminlist[0:Ngen].shape) > lminlist[0:Ngen])
        incb = NN

        N = sum(np.array(ls[u]*np.ones(lminlist[0:Ngen].shape) > lminlist[0:Ngen]) & np.array(ls[u]*np.ones(lmaxlist[0:Ngen].shape) < lmaxlist[0:Ngen]))

        if N == 1 or N == 0:
            Ngg = 0
        else:
            Ngg = nCr(N,2)

        Cov = np.zeros([N+1+N+Ngg,N+1+N+Ngg], dtype = np.float64)

        if ls[u] == 2 or ls[u] == 1500:
            print (Ngen, N, Ngg, skip)


        offset = skip
        # N = Ngen-offset-incb


        MatSize = Ngg
        indMat = [np.zeros([2,2]) for i in range(MatSize)]

        n = 1
        m = 1
        a = 1+offset
        b = 2+offset
        while n <= Ngg:
            indMat[n-1][1][0] = a
            indMat[n-1][1][1] = b
            if b < N+offset:
                b = b + 1
            else:
                a = a + 1
                b = a + 1
            n = n + 1
            m = m + 1

        for i in range(N+1+N+Ngg):
            if i == 0:
                Cov[0,i] = (2*cl_kk[u]**2)/(2.*ls[u]+1)
            if i > 0 and i < N+1:
                Cov[0,i] = (2*cl_kk[u]*cl_kg[int(i)-1+skip][u])/(2.*ls[u]+1)
            if i > N and i < 2*N+1:
                Cov[0,i] = (2*cl_kg[int(i)-1-N+skip][u]**2)/(2.*ls[u]+1)
            if i > 2*N:
                ind = i-2*N-1
                if indMat[ind][0][0] == 0 and indMat[ind][1][0] == 0:
                    v1 = cl_kk[u]
                elif indMat[ind][0][0] == 0 and indMat[ind][1][0] != 0:
                    v1 = cl_kg[int(indMat[ind][1][0])-1][u]
                elif indMat[ind][0][0] != 0 and indMat[ind][1][0] == 0:
                    v1 = cl_kg[int(indMat[ind][0][0])-1][u]
                elif indMat[ind][0][0] != 0 and indMat[ind][1][0] == indMat[ind][0][0]:
                    v1 = cl_gg[int(indMat[ind][0][0])-1][u]
                else:
    #                 if ls[u] < ellmin_gg:
                    v1 = 0
                if indMat[ind][0][1] == 0 and indMat[ind][1][1] == 0:
                    v2 = cl_kk[u]
                elif indMat[ind][0][1] == 0 and indMat[ind][1][1] != 0:
                    v2 = cl_kg[int(indMat[ind][1][1])-1][u]
                elif indMat[ind][0][1] != 0 and indMat[ind][1][1] == 0:
                    v2 = cl_kg[int(indMat[ind][0][1])-1][u]
                elif indMat[ind][0][1] != 0 and indMat[ind][1][1] == indMat[ind][0][1]:
                    v2 = cl_gg[int(indMat[ind][0][1])-1][u]
                else:
    #                 if ls[u] < ellmin_gg:
                    v2 = 0
                
                if indMat[ind][0][0] == 0 and indMat[ind][1][1] == 0:
                    v3 = cl_kk[u]
                elif indMat[ind][0][0] == 0 and indMat[ind][1][1] != 0:
                    v3 = cl_kg[int(indMat[ind][1][1])-1][u]
                elif indMat[ind][0][0] != 0 and indMat[ind][1][1] == 0:
                    v3 = cl_kg[int(indMat[ind][0][0])-1][u]
                elif indMat[ind][0][0] != 0 and indMat[ind][1][1] == indMat[ind][0][0]:
                    v3 = cl_gg[int(indMat[ind][0][0])-1][u]
                else:
    #                 if ls[u] < ellmin_gg:
                    v3 = 0
                
                if indMat[ind][0][1] == 0 and indMat[ind][1][0] == 0:
                    v4 = cl_kk[u]
                elif indMat[ind][0][1] == 0 and indMat[ind][1][0] != 0:
                    v4 = cl_kg[int(indMat[ind][1][0])-1][u]
                elif indMat[ind][0][1] != 0 and indMat[ind][1][0] == 0:
                    v4 = cl_kg[int(indMat[ind][0][1])-1][u]
                elif indMat[ind][0][1] != 0 and indMat[ind][1][0] == indMat[ind][0][1]:
                    v4 = cl_gg[int(indMat[ind][0][1])-1][u]
                else:
    #                 if ls[u] < ellmin_gg:
                    v4 = 0
                
                Cov[0, i] = (v1*v2+v3*v4)/(2.*ls[u]+1)
                
            if i != 0:
                Cov[i, 0] = Cov[0, i]
            
            j = 1
            while i > 0 and i <= N and j <= N:
                if j == i:
                    Cov[i, i] = (cl_kk[u]*cl_gg[int(i)-1+skip][u]+cl_kg[int(i)-1+skip][u]**2)/(2.*ls[u]+1)
                    Cov[i+N, i] = (2*cl_gg[int(i)-1+skip][u]*cl_kg[int(i)-1+skip][u])/(2.*ls[u]+1)
                    Cov[i, i+N] = Cov[i+N, i]
                if j > i:
                    Cov[j, i] = (cl_kg[int(i)-1+skip][u]*cl_kg[int(j)-1+skip][u])/(2.*ls[u]+1)
                    Cov[i, j] = Cov[j, i]
                j = j + 1

            j = 2*N+1
            while i > 0 and i <= N and j <= 2*N+Ngg:
                ind = j-2*N-1
                if indMat[ind][1][0] == i:
                    Cov[i, j] = (cl_kg[int(indMat[ind][1][1])-1][u]*cl_gg[int(i)-1+skip][u])/(2.*ls[u]+1)
                    Cov[j, i] = Cov[i, j]
                if indMat[ind][1][1] == i:
                    Cov[i, j] = (cl_kg[int(indMat[ind][1][0])-1][u]*cl_gg[int(i)-1+skip][u])/(2.*ls[u]+1)
                    Cov[j, i] = Cov[i, j]
                j = j + 1
            
            if i > N and i < 2*N+1:
                Cov[i, i] = (2*cl_gg[int(i-N)-1+skip][u]**2)/(2.*ls[u]+1)
            if i > 2*N:
                ind = i-2*N-1
                Cov[i, i] = (cl_gg[int(indMat[ind][1][0])-1][u]*cl_gg[int(indMat[ind][1][1])-1][u])/(2.*ls[u]+1)

        Cov = Cov/fsky

        # print (u, np.shape(Cov), N)

        for ij in range(N+1):
            Cov = np.delete(Cov, 0, 0)
            Cov = np.delete(Cov, 0, 1)

        invCov = inv(Cov)

        # print (u, skip, incb, N)


        npar = 2*N+8
        for i in range(npar):
            vec1 = np.zeros(N+Ngg)
            if i < N:
                if ndensity[i] != 0:
                    vec1[i] = 2.*( cl_gg[i+skip][u] - 1./(ndensity[i+skip]/(8.462*10**-8)) )
                # else:
                #     vec1[i+N+1-skip] = 0
                # if ls[u] < ellmin_gg:
                #     vec1[i+N+1] = 0
            elif i >= N and i < 2*N:
                if ndensity[i-N] != 0:
                    vec1[i-N] = 2.*( cl_gg[i-N+skip][u] - 1./(ndensity[i-N+skip]/(8.462*10**-8)) )
                # else:
                #     vec1[i+N+1-Ngen-skip] = 0
                # if ls[u] < ellmin_gg:
                #     vec1[i+N+1-N] = 0
            else:
                for jj in range(N):
                    vec1[jj] = dClgg_dpar[i-2*N][jj+skip][u]

            for j in range(npar):
                vec2 = np.zeros(N+Ngg)
                if j < N:
                    if ndensity[j] != 0:
                        vec2[j] = 2.*( cl_gg[j+skip][u] - 1./(ndensity[j+skip]/(8.462*10**-8)) )
                    # else:
                    #     vec2[j+N+1-skip] = 0
                    # if ls[u] < ellmin_gg:
                    #     vec2[j+N+1] = 0
                elif j >= N and j < 2*N:
                    if ndensity[j-Ngen] != 0:
                        vec2[j-N] = 2.*( cl_gg[j-N+skip][u] - 1./(ndensity[j-N+skip]/(8.462*10**-8)) )
                    # else:
                    #     vec2[j+N+1-Ngen-skip] = 0
                    # if ls[u] < ellmin_gg:
                    #     vec2[j+N+1-N] = 0
                else:
                    for jj in range(N):
                        vec2[jj] = dClgg_dpar[j-2*N][jj+skip][u]
                # if i ==0:
                #     print (Ngg, np.shape(vec1), np.shape(invCovlist[u]))
                if i < N:
                    if j < N:
                        FisherM[i+skip,j+skip] += np.dot( np.dot(vec1, invCov), vec2 )
                    elif j >= N and j < 2*N:
                        FisherM[i+skip,(j-N)+Ngen+skip] += np.dot( np.dot(vec1, invCov), vec2 )
                    else:
                        FisherM[i+skip,2*Ngen+(j-2*N)] += np.dot( np.dot(vec1, invCov), vec2 )
                elif i >= N and i < 2*N:
                    if j < N:
                        FisherM[(i-N)+Ngen+skip,j+skip] += np.dot( np.dot(vec1, invCov), vec2 )
                    elif j >= N and j < 2*N:
                        FisherM[(i-N)+Ngen+skip,(j-N)+Ngen+skip] += np.dot( np.dot(vec1, invCov), vec2 )
                    else:
                        FisherM[(i-N)+Ngen+skip,2*Ngen+(j-2*N)] += np.dot( np.dot(vec1, invCov), vec2 )
                else:
                    if j < N:
                        FisherM[2*Ngen+(i-2*N),j+skip] += np.dot( np.dot(vec1, invCov), vec2 )
                    elif j >= N and j < 2*N:
                        FisherM[2*Ngen+(i-2*N),(j-N)+Ngen+skip] += np.dot( np.dot(vec1, invCov), vec2 )
                    else:
                        FisherM[2*Ngen+(i-2*N),2*Ngen+(j-2*N)] += np.dot( np.dot(vec1, invCov), vec2 )


    N = len(zlist)-1
    if ndensity[-1] == 0:

        print 'error'

    else:

        params = []

        for i in range(N):
            params.append('sig8_%s' %i)

        for i in range(N):
            params.append('b1_%s' %i)

        params.append('H0')
        params.append('ombh2')
        params.append('omch2')
        params.append('ns')
        params.append('As')
        params.append('mnu')
        params.append('w')
        params.append('tau')
        params.append('wa')

        FM_tot = FisherMatrix( FisherM, params )

        FM_tot = FM_tot + FisherS4 


        for i in range(N):
            FM_tot.delete('sig8_%s' %i)   

        Nmargb = sum(lmaxlist < lmin)
        for i in range(Nmargb):
            FM_tot.delete('b1_%s' %i) 

        # FM_tot.delete('omk')

        Cov_marg = FM_tot.marge_var_2param('w', 'wa')

        eigvec, eigval, u = np.linalg.svd(Cov_marg)

        semimaj = np.sqrt(eigval[0])
        semimin = np.sqrt(eigval[1])

        FoM = 1./(2.3*semimaj*semimin)

        # print 'check'
        # print semimaj
        # print np.sqrt(0.5*(Cov_marg[0,0]+Cov_marg[1,1])+np.sqrt( 0.25*(Cov_marg[0,0]-Cov_marg[1,1])**2 + Cov_marg[1,0]**2 ))
        # print 1./(2.3*semimaj*semimin)

        w0_kgTE.append( FM_tot.sigmas()['w'] )
        wa_kgTE.append( FM_tot.sigmas()['wa'] )
        mnu_kgTE.append( FM_tot.sigmas()['mnu']*1000 )
        FoM_kgTE.append( FoM )

        FM_tot.delete('mnu')

        Cov_marg = FM_tot.marge_var_2param('w', 'wa')

        eigvec, eigval, u = np.linalg.svd(Cov_marg)

        semimaj = np.sqrt(eigval[0])
        semimin = np.sqrt(eigval[1])

        FoM = 1./(2.3*semimaj*semimin)

        # print 'check'
        # print semimaj
        # print np.sqrt(0.5*(Cov_marg[0,0]+Cov_marg[1,1])+np.sqrt( 0.25*(Cov_marg[0,0]-Cov_marg[1,1])**2 + Cov_marg[1,0]**2 ))
        # print 1./(2.3*semimaj*semimin)

        w0_kgTE_mnufixed.append( FM_tot.sigmas()['w'] )
        wa_kgTE_mnufixed.append( FM_tot.sigmas()['wa'] )
        FoM_kgTE_mnufixed.append( FoM )

        print 'this'
        print FoM



    N = len(zlist)-1
    if ndensity[-1] == 0:

        print 'error'

    else:

        params = []

        for i in range(N):
            params.append('sig8_%s' %i)

        for i in range(N):
            params.append('b1_%s' %i)

        params.append('H0')
        params.append('ombh2')
        params.append('omch2')
        params.append('ns')
        params.append('As')
        params.append('mnu')
        params.append('w')
        params.append('tau')
        params.append('wa')

        FM_tot = FisherMatrix( FisherM, params )

        FM_tot = FM_tot + FisherS4 + FisherBAO


        for i in range(N):
            FM_tot.delete('sig8_%s' %i)   

        Nmargb = sum(lmaxlist < lmin)
        for i in range(Nmargb):
            FM_tot.delete('b1_%s' %i) 

        # FM_tot.delete('omk')

        Cov_marg = FM_tot.marge_var_2param('w', 'wa')

        eigvec, eigval, u = np.linalg.svd(Cov_marg)

        semimaj = np.sqrt(eigval[0])
        semimin = np.sqrt(eigval[1])

        FoM = 1./(2.3*semimaj*semimin)

        # print 'check'
        # print semimaj
        # print np.sqrt(0.5*(Cov_marg[0,0]+Cov_marg[1,1])+np.sqrt( 0.25*(Cov_marg[0,0]-Cov_marg[1,1])**2 + Cov_marg[1,0]**2 ))
        # print 1./(2.3*semimaj*semimin)

        w0_DESI.append( FM_tot.sigmas()['w'] )
        wa_DESI.append( FM_tot.sigmas()['wa'] )
        mnu_DESI.append( FM_tot.sigmas()['mnu']*1000 )
        FoM_DESI.append( FoM )

        FM_tot.delete('mnu')

        Cov_marg = FM_tot.marge_var_2param('w', 'wa')

        eigvec, eigval, u = np.linalg.svd(Cov_marg)

        semimaj = np.sqrt(eigval[0])
        semimin = np.sqrt(eigval[1])

        FoM = 1./(2.3*semimaj*semimin)

        # print 'check'
        # print semimaj
        # print np.sqrt(0.5*(Cov_marg[0,0]+Cov_marg[1,1])+np.sqrt( 0.25*(Cov_marg[0,0]-Cov_marg[1,1])**2 + Cov_marg[1,0]**2 ))
        # print 1./(2.3*semimaj*semimin)

        w0_DESI_mnufixed.append( FM_tot.sigmas()['w'] )
        wa_DESI_mnufixed.append( FM_tot.sigmas()['wa'] )
        FoM_DESI_mnufixed.append( FoM )


    N = len(zlist)-1
    if ndensity[-1] == 0:

        print 'error'

    else:

        params = []

        for i in range(N):
            params.append('sig8_%s' %i)

        for i in range(N):
            params.append('b1_%s' %i)

        params.append('H0')
        params.append('ombh2')
        params.append('omch2')
        params.append('ns')
        params.append('As')
        params.append('mnu')
        params.append('w')
        params.append('tau')
        params.append('wa')

        FM_tot = FisherMatrix( FisherM, params )

        FM_tot = FM_tot + FisherPS4 


        for i in range(N):
            FM_tot.delete('sig8_%s' %i)   

        Nmargb = sum(lmaxlist < lmin)
        for i in range(Nmargb):
            FM_tot.delete('b1_%s' %i) 

        # FM_tot.delete('omk')

        Cov_marg = FM_tot.marge_var_2param('w', 'wa')

        eigvec, eigval, u = np.linalg.svd(Cov_marg)

        semimaj = np.sqrt(eigval[0])
        semimin = np.sqrt(eigval[1])

        FoM = 1./(2.3*semimaj*semimin)

        # print 'check'
        # print semimaj
        # print np.sqrt(0.5*(Cov_marg[0,0]+Cov_marg[1,1])+np.sqrt( 0.25*(Cov_marg[0,0]-Cov_marg[1,1])**2 + Cov_marg[1,0]**2 ))
        # print 1./(2.3*semimaj*semimin)

        Pw0_kgTE.append( FM_tot.sigmas()['w'] )
        Pwa_kgTE.append( FM_tot.sigmas()['wa'] )
        Pmnu_kgTE.append( FM_tot.sigmas()['mnu']*1000 )
        PFoM_kgTE.append( FoM )

        FM_tot.delete('mnu')

        Cov_marg = FM_tot.marge_var_2param('w', 'wa')

        eigvec, eigval, u = np.linalg.svd(Cov_marg)

        semimaj = np.sqrt(eigval[0])
        semimin = np.sqrt(eigval[1])

        FoM = 1./(2.3*semimaj*semimin)

        # print 'check'
        # print semimaj
        # print np.sqrt(0.5*(Cov_marg[0,0]+Cov_marg[1,1])+np.sqrt( 0.25*(Cov_marg[0,0]-Cov_marg[1,1])**2 + Cov_marg[1,0]**2 ))
        # print 1./(2.3*semimaj*semimin)

        Pw0_kgTE_mnufixed.append( FM_tot.sigmas()['w'] )
        Pwa_kgTE_mnufixed.append( FM_tot.sigmas()['wa'] )
        PFoM_kgTE_mnufixed.append( FoM )



    N = len(zlist)-1
    if ndensity[-1] == 0:

        print 'error'

    else:

        params = []

        for i in range(N):
            params.append('sig8_%s' %i)

        for i in range(N):
            params.append('b1_%s' %i)

        params.append('H0')
        params.append('ombh2')
        params.append('omch2')
        params.append('ns')
        params.append('As')
        params.append('mnu')
        params.append('w')
        params.append('tau')
        params.append('wa')

        FM_tot = FisherMatrix( FisherM, params )

        FM_tot = FM_tot + FisherPS4 + FisherBAO


        for i in range(N):
            FM_tot.delete('sig8_%s' %i)   

        Nmargb = sum(lmaxlist < lmin)
        for i in range(Nmargb):
            FM_tot.delete('b1_%s' %i) 

        # FM_tot.delete('omk')

        Cov_marg = FM_tot.marge_var_2param('w', 'wa')

        eigvec, eigval, u = np.linalg.svd(Cov_marg)

        semimaj = np.sqrt(eigval[0])
        semimin = np.sqrt(eigval[1])

        FoM = 1./(2.3*semimaj*semimin)

        # print 'check'
        # print semimaj
        # print np.sqrt(0.5*(Cov_marg[0,0]+Cov_marg[1,1])+np.sqrt( 0.25*(Cov_marg[0,0]-Cov_marg[1,1])**2 + Cov_marg[1,0]**2 ))
        # print 1./(2.3*semimaj*semimin)

        Pw0_DESI.append( FM_tot.sigmas()['w'] )
        Pwa_DESI.append( FM_tot.sigmas()['wa'] )
        Pmnu_DESI.append( FM_tot.sigmas()['mnu']*1000 )
        PFoM_DESI.append( FoM )

        FM_tot.delete('mnu')

        Cov_marg = FM_tot.marge_var_2param('w', 'wa')

        eigvec, eigval, u = np.linalg.svd(Cov_marg)

        semimaj = np.sqrt(eigval[0])
        semimin = np.sqrt(eigval[1])

        FoM = 1./(2.3*semimaj*semimin)

        # print 'check'
        # print semimaj
        # print np.sqrt(0.5*(Cov_marg[0,0]+Cov_marg[1,1])+np.sqrt( 0.25*(Cov_marg[0,0]-Cov_marg[1,1])**2 + Cov_marg[1,0]**2 ))
        # print 1./(2.3*semimaj*semimin)

        Pw0_DESI_mnufixed.append( FM_tot.sigmas()['w'] )
        Pwa_DESI_mnufixed.append( FM_tot.sigmas()['wa'] )
        PFoM_DESI_mnufixed.append( FoM )


print FoM_kgTE


# DataOut = np.column_stack((w0_kgTE, wa_kgTE, mnu_kgTE, FoM_kgTE, w0_kgTE_mnufixed, wa_kgTE_mnufixed, FoM_kgTE_mnufixed))
# np.savetxt('kgTE_N16.dat', DataOut)

# DataOut = np.column_stack((w0_DESI, wa_DESI, mnu_DESI, FoM_DESI, w0_DESI_mnufixed, wa_DESI_mnufixed, FoM_DESI_mnufixed))
# np.savetxt('kgTE_DESI_N16.dat', DataOut)


# DataOut = np.column_stack((Pw0_kgTE, Pwa_kgTE, Pmnu_kgTE, PFoM_kgTE, Pw0_kgTE_mnufixed, Pwa_kgTE_mnufixed, PFoM_kgTE_mnufixed))
# np.savetxt('PkgTE_N16.dat', DataOut)

# DataOut = np.column_stack((Pw0_DESI, Pwa_DESI, Pmnu_DESI, PFoM_DESI, Pw0_DESI_mnufixed, Pwa_DESI_mnufixed, PFoM_DESI_mnufixed))
# np.savetxt('PkgTE_DESI_N16.dat', DataOut)

