import numpy as np
from numba import njit
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
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

def eval_T_dressed(E,k,Om,Delta, invkfa,PiR,PiI):
    Om_R=np.sqrt(Om**2+Delta**2)
    Tmat=np.ones((E.size,2,2),dtype=np.complex128)
    #for i in range(E.size):
    Tmat*=np.array([[Om_R+Delta,Om],[Om,Om_R-Delta]])
    bare=Om_R*invkfa/(4*np.pi)
    pl=(Om_R+Delta)*(PiR(E-Om_R/2,k)[:,0]+1j*PiI(E-Om_R/2,k)[:,0])
    mi=(Om_R-Delta)*(PiR(E+Om_R/2,k)[:,0]+1j*PiI(E+Om_R/2,k)[:,0])
    return Tmat/(bare-pl-mi)[:,np.newaxis,np.newaxis]
def eval_Sigma_dressed(omegas,invkfa,Om,Delta,PiR,PiI,Nk):
    res=np.zeros((omegas.size,2,2),dtype=np.complex128)
    qs=np.linspace(0,1,Nk) # zero temperature
    dq=qs[1]-qs[0]
    for q in qs:
        if q<1: #not including 1 because Riemann sum doesn't have last point
            res+=q**2*eval_T_dressed(omegas+q**2-1,q,Om,Delta,invkfa,PiR,PiI)
    return res*(dq/(2*np.pi**2))
def calc_V(Omega, Delta):
    Omega_R=np.sqrt(Omega**2+Delta**2)
    V=np.array([[np.sqrt(Omega_R+Delta),np.sqrt(Omega_R-Delta)],
                [np.sqrt(Omega_R-Delta),-np.sqrt(Omega_R+Delta)]])
    return V/np.sqrt(2*Omega_R)

if __name__ == '__main__':
    omegas = np.linspace(-3, 2, 100)
    invkfas = np.linspace(-6, 4, 200)
    # parameters for dressed eval
    Omega = 0.01
    Delta = 0.0
    Nk = 200
    Nmu=200
    eta=0.05
    #momenta for bubble evaluation
    ks = np.linspace(0, 5, 30)
    # parameters for notdressed eval
    cutoff = 5
    N = 200
    # Evaluate bubble
    Pi=eval_Pis(ks, omegas, Nmu, Nk, eta)
    # interpolate
    interpre = RectBivariateSpline(omegas, ks, np.real(Pi))
    interpim = RectBivariateSpline(omegas, ks, np.imag(Pi))
    # Evaluate conversion matrix
    V=calc_V(Omega, Delta)
    res = np.zeros((invkfas.size, omegas.size))
    for invkfa in invkfas:
        Sigmadressed = np.swapaxes(V.dot(eval_Sigma_dressed(omegas, invkfa, Omega, Delta, interpre, interpim, Nk).dot(V)),
                                   0, 1)[:, 0, 0]
        res[invkfa == invkfas, :] = -(1 / np.pi) * np.imag(1 / (omegas + 1j * eta - Sigmadressed))
    plt.figure(figsize=(7, 3), dpi=200)
    plt.pcolormesh( invkfas,omegas, res.T, norm=LogNorm())
    cb = plt.colorbar()
    cb.set_label('Impurity spectral function')
    plt.ylabel(r'frequency $\omega/EF$')
    plt.xlabel(r'inverse scattering length $-1/k_Fa$')
    plt.title('$\Omega_0=3$')
    plt.tight_layout(pad=0)
    plt.show()

