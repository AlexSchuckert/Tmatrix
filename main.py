import numpy as np
from numba import njit
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import root_scalar
@njit #(parallel=True)
def eval_Pis(ks,omegas,Nmu, Nk,eta):
    qs=np.linspace(0,1,Nk) # zero temperature
    dq=(qs[1]-qs[0])
    mus=np.linspace(-1,1,Nmu)
    dmu=mus[1]-mus[0]
    #prefactor of bubble
    pref=(dq*dmu/(2*np.pi)**2)
    res=np.zeros((omegas.size,ks.size),dtype=np.complex128)
    for k in range(ks.size):
        res[:,k]-=1j*np.sqrt(omegas+1j*eta+1-ks[k]**2/2)/(8*np.sqrt(2)*np.pi)
        for q in qs[qs<1]:
            for mu in mus[mus<1]:
                res[:,k]-=pref*q**2/(omegas+1j*eta-2*q**2-ks[k]**2+2*q*ks[k]*mu+1)
    return res

def eval_T_dressed(E,k,Om,Delta, invkfa,PiR,PiI,interpbounds):
    Om_R=np.sqrt(Om**2+Delta**2)
    assert ((E-Om_R/2).any() > interpbounds[0]),"bla" #and (E-Om_R/2 < interpbounds[1]) and (E+Om_R/2 > interpbounds[0]) and (E+Om_R/2 < interpbounds[1]), "Evaluation outside bounds of interpolation."
    Tmat=np.ones((E.size,2,2),dtype=np.complex128)
    #for i in range(E.size):
    Tmat*=np.array([[Om_R+Delta,Om],[Om,Om_R-Delta]])
    bare=Om_R*invkfa/(4*np.pi)
    pl=(Om_R+Delta)*(PiR(E-Om_R/2,k)[:,0]+1j*PiI(E-Om_R/2,k)[:,0])
    mi=(Om_R-Delta)*(PiR(E+Om_R/2,k)[:,0]+1j*PiI(E+Om_R/2,k)[:,0])
    return Tmat/(bare-pl-mi)[:,np.newaxis,np.newaxis]
def eval_Sigma_dressed(omegas,invkfa,Om,Delta,PiR,PiI,Nk,interpbounds):
    res=np.zeros((omegas.size,2,2),dtype=np.complex128)
    qs=np.linspace(0,1,Nk) # zero temperature
    dq=qs[1]-qs[0]
    max=0
    for q in qs:
        if q<1: #not including 1 because Riemann sum doesn't have last point
            # Om_R = np.sqrt(Om ** 2 + Delta ** 2)
            # val=np.maximum(np.max(np.abs(omegas+q**2-1 - Om_R / 2)),np.max(np.abs(omegas+q**2-1 + Om_R / 2)))
            # if val>max:
            #     max=val
            res+=q**2*eval_T_dressed(omegas+q**2-1,q,Om,Delta,invkfa,PiR,PiI,interpbounds)
    # print(max)
    return res*(dq/(2*np.pi**2))
def calc_V(Omega, Delta):
    Omega_R=np.sqrt(Omega**2+Delta**2)
    V=np.array([[np.sqrt(Omega_R+Delta),np.sqrt(Omega_R-Delta)],
                [np.sqrt(Omega_R-Delta),-np.sqrt(Omega_R+Delta)]])
    return V/np.sqrt(2*Omega_R)


if __name__ == '__main__':


    # external E,k values.
    # For polaron, only need 0 to kf because of restriction in self energy
    ks = np.linspace(0, 1.01, 30)


    # parameters for notdressed eval
    cutoff = 5
    N = 200
    Nk = 1000 # not sure whether this should be same as in bubble
    eta = 0.05
    # Evaluate bubble
    # Nk = 1000
    # Nmu = 1000
    # eta = 0.05
    # Pi = 1/eval_Pis(ks, omegas, Nmu, Nk, eta)
    # Nk = 1000
    # Nmu = 2000
    # eta = 0.05
    # Pi1 = 1/eval_Pis(ks, omegas, Nmu, Nk, eta)
    #np.save('Pi1.npy',Pi1)
    #np.save('Pi.npy', Pi)
    # plt.subplot(411)
    # minPi = np.min(np.real(Pi))
    # maxPi = np.max(np.real(Pi))
    # plt.pcolormesh(ks,omegas, np.real(Pi),vmin=minPi,vmax=maxPi)
    # plt.colorbar()
    # plt.subplot(412)
    # plt.pcolormesh(ks, omegas,  np.real(Pi1),vmin=minPi,vmax=maxPi)
    # plt.colorbar()
    # plt.subplot(413)
    # #plot error of self energy integrand
    # plt.pcolormesh(ks, omegas, (ks**2)[np.newaxis,:] *np.abs((np.real(Pi) - np.real(Pi1))) ) #/ np.abs(np.real(Pi1))
    # plt.colorbar()
    # print(np.max(np.abs((np.real(Pi) - np.real(Pi1))) / np.abs(np.real(Pi1))), np.mean(np.abs(np.real(Pi) - np.real(Pi1)) / np.abs(np.real(Pi1))))
    # plt.subplot(414)
    # plt.pcolormesh(ks, omegas, (ks**2)[np.newaxis,:] * np.abs((np.imag(Pi) - np.imag(Pi1))) ) # np.abs(np.imag(Pi1))
    # plt.colorbar()
    # print(np.max(np.abs((np.imag(Pi) - np.imag(Pi1))) / np.abs(np.imag(Pi1))), np.mean(np.abs((np.imag(Pi) - np.imag(Pi1))) / np.abs(np.imag(Pi1))))
    # plt.show()
    Pi=np.load('Pi1.npy')
    # interpolate - the omegas and ks need to match Pi1
    omegas = np.linspace(-10, 10, 100)
    ks = np.linspace(0, 1.01, 30)
    interpbounds = [np.min(omegas), np.max(omegas)]
    interpre = RectBivariateSpline(omegas, ks, np.real(Pi))
    interpim = RectBivariateSpline(omegas, ks, np.imag(Pi))
    # Evaluate conversion matrix
    # parameters for dressed eval
    # Omega = 3.0
    # Delta = 0.0
    # invkfas = np.linspace(-0.5, 0.75, 50)
    # omegas = np.linspace(-10, 10, 200)
    # V=calc_V(Omega, Delta)
    ##########
    ## Evaluate spectral function in dressed and undressed basis as fct of kfa
    ##########
    # Specfunc11 = np.zeros((invkfas.size, omegas.size))
    # Specfuncmm = np.zeros((invkfas.size, omegas.size))
    # for invkfa in invkfas:
    #     Sigma_dressed=eval_Sigma_dressed(omegas, invkfa, Omega, Delta, interpre, interpim, Nk,interpbounds)
    #     Sigma = np.swapaxes(V.dot(Sigma_dressed.dot(V)),
    #                                0, 1)[:, 0, 0]
    #     Specfunc11[invkfa == invkfas, :] = -(1 / np.pi) * np.imag(1 / (omegas + 1j * eta - Sigma))
    #     Specfuncmm[invkfa == invkfas, :] = -(1 / np.pi) * np.imag(1 / (omegas + 1j * eta - Sigma_dressed[:,1,1]))
    # plt.figure(figsize=(7, 3), dpi=200)
    # plt.subplot(121)
    # plt.pcolormesh(invkfas, omegas, Specfuncmm.T, norm=LogNorm())
    # cb = plt.colorbar()
    # cb.set_label('Impurity spectral function')
    # plt.ylabel(r'frequency $\omega/EF$')
    # plt.xlabel(r'inverse scattering length $1/k_Fa$')
    # plt.title('$\Omega_0=3$')
    # plt.tight_layout(pad=0)
    # plt.subplot(122)
    # plt.pcolormesh(invkfas, omegas, Specfunc11.T, norm=LogNorm())
    # cb = plt.colorbar()
    # cb.set_label('Impurity spectral function')
    # plt.ylabel(r'frequency $\omega/EF$')
    # plt.xlabel(r'inverse scattering length $-1/k_Fa$')
    # plt.title('$\Omega_0=3$')
    # plt.tight_layout(pad=0)
    # plt.show()


    ##########
    ## Calculate polaron energy at unitarity
    #########
    # Plot
    # Omegas = np.logspace(-1, 1, 10)
    # plt_dets=np.linspace(-1, 1.0, 100)
    # # Delta=-0.6
    invkfa=0.0
    # roots_Omega0 = np.zeros((Omegas.size,plt_dets.size))
    # for Omega in Omegas:
    #
    #
    #
    #     # def func(x):
    #     #     Sigma_dressed = eval_Sigma_dressed(np.array([x]), invkfa, Omega, -x, interpre, interpim, Nk,interpbounds)
    #     #     return x - np.real(Sigma_dressed[:, 0,0])
    #     def func_undressed(x):
    #         V = calc_V(Omega, x)
    #         Sigma_dressed = eval_Sigma_dressed(np.array([x]), invkfa, Omega, x, interpre, interpim, Nk, interpbounds)
    #         Sigma = np.swapaxes(V.dot(Sigma_dressed.dot(V)),
    #                             0, 1)[:, 0, 0]
    #         return x - np.real(Sigma)
    #     roots_Omega0[Omega==Omegas,:]=np.array([func_undressed(plt_dets[i]) for i in range(plt_dets.size)])[:,0]
    # plt.plot(plt_dets,roots_Omega0.T)
    # plt.show()



    # Find root -> polaron energy
    #invkfa=0.5
    Omegas = np.logspace(-1, 1.1, 50)
    Ep=[]
    Ep_undressed = []
    #Omegas=np.logspace(0.0,5.0,100)
    #Delta=-0.6
    for Omega in Omegas:
        # def func(x):
        #     V = calc_V(Omega, -x)
        #     Sigma_dressed = eval_Sigma_dressed(np.array([x]), invkfa, Omega, -x, interpre, interpim, Nk,interpbounds)
        #     return x - np.real(Sigma_dressed[:, 0,0])
        def func_undressed(x):
            V = calc_V(Omega, x)
            Sigma_dressed = eval_Sigma_dressed(np.array([x]), invkfa, Omega, x, interpre, interpim, Nk,interpbounds)
            Sigma=np.swapaxes(V.dot(Sigma_dressed.dot(V)),
                                       0, 1)[:, 0, 0]
            return x - np.real(Sigma)
        #root = root_scalar(func, bracket=(-1,0.1)).root
        root_undr = root_scalar(func_undressed, bracket=(-1, 1.0)).root
        # Ep.append(root)
        Ep_undressed.append(root_undr)
    #plt.plot(Omegas, Ep, label='dressed')
    plt.plot(Omegas, -np.array(Ep_undressed), label='undressed')
    dat = np.loadtxt("../Data_Unitarity/Delta0.csv", delimiter=',')
    plt.errorbar(dat[:, 0], dat[:, 1], fmt='.', yerr=dat[:, 2])
    plt.xscale('log')
    plt.ylabel(r"Polaron energy $E_p/E_F$")
    plt.xlabel(r"$\Omega_0/E_F$")
    plt.show()
    # #####
    # ## Lifetimes
    # ####
    # lifetimes_undressed=[]
    # for i in range(Omegas.size):
    #     V = calc_V(Omegas[i], Ep_undressed[i])
    #     Sigma_dressed = eval_Sigma_dressed(np.array(Ep_undressed[i]), invkfa, Omegas[i], Ep_undressed[i], interpre, interpim, Nk,interpbounds)[0]
    #     Sigma = V.dot(Sigma_dressed.dot(V))[0, 0]
    #     lt=np.imag(Sigma)
    #     lifetimes_undressed.append(lt)
    # plt.plot(Omegas, -2*np.array(lifetimes_undressed))
    #
    # dat = np.loadtxt("../Data_Unitarity/tau_T2.csv", delimiter=',')
    # dat1 = np.loadtxt("../Data_Unitarity/OmegaR-Omega0.csv", delimiter=',')
    # # Assuming OmegaR/Omega_0=sqrt(Z) - this is not necessarily true, see (S52) in Adlong
    # plt.errorbar(dat[:, 0], dat[:, 1]/(dat1[:, 1])**2, fmt='.', yerr=dat[:, 2])
    #
    # plt.xlabel(r'Rabi frequency $\Omega_0/E_F$')
    # plt.ylabel(r'Imag. part of self energy -2Im($\Sigma$)/E_F')
    # plt.yscale('log')
    # plt.xscale('log')
    # plt.show()


    ########
    ## Quasiparticle residue ##
    ########




