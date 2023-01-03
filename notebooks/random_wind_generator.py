## -*- coding: utf-8 -*-
#!/usr/bin/env python3
u"""General informations"""
__author__ = "Filippo Gatti"
__copyright__ = "Copyright 2022, CentraleSupélec (LMPS UMR CNRS 9026)"
__credits__ = ["Filippo Gatti"]
__license__ = "GPL"
__version__ = "1.0.1"
__Maintainer__ = "Filippo Gatti"
__email__ = "filippo.gatti@centralesupelec.fr"
__status__ = "Beta"

# Load fundamental modules
import numpy as np
import pandas as pd
from numpy.random import default_rng
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.colors import LogNorm
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

# Compute the transfer function using the Fast Fourier Transform (use np.fft.rfft - https://numpy.org/doc/stable/reference/generated/numpy.fft.fft.html)
def next_power_of_2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()

def beta_i(t):
    nfft = next_power_of_2(vtm.size)
    dfq = 1.0/(nfft*dt)
    vfq = np.linspace(0.0,(nfft/2+1)*dfq,num=nfft//2+1)
    vwo = 2.0*np.pi*vfq
    
    # random amplitude (Normal distribution)
    a = np.array(rng.standard_normal(vfq.size,dtype=np.float64))
    b = np.array(rng.standard_normal(vfq.size,dtype=np.float64))
    rho = np.abs(a+1j*b)

    # random phase (Uniform distribution)
    phi = rng.uniform(0.0,1.0,vfq.size).reshape(-1,1)

    return 2.0*rho*np.sqrt(4.0*Sv(vfq)*dfq)@np.cos(-2.0*np.pi*phi+np.tensordot(vwo,t,axes = 0))

def beta(t,Ns):
    all_beta=np.empty((Ns,vtm.size))
    for i in range(Ns):
        all_beta[i,:]=beta_i(t)
    return all_beta


if __name__ == '__main__':
    # time vector
    dt = 1.0e-2  # time step
    dur = 60.0  # duration
    vtm = np.arange(0.0, dur, dt)  # time vector
    ntm = vtm.size  # number of time steps
    nfft = next_power_of_2(vtm.size)
    dfq = 1.0/(nfft*dt)
    vfq = np.linspace(0.0, (nfft/2+1)*dfq, num=nfft//2+1)

    # Wind load characteristics PSD
    muV = 1.0e5/3600.0  # mean velocity at 10m height: 100 km/h
    kappa = 0.002  # constant
    Lcar = 1200.0  # constant

    Sv = lambda f: 4.0*kappa*muV*Lcar/(2.0+(f*Lcar/muV)**2)**(5/6)

    # Hint: Use the numpy random number generator https://numpy.org/doc/stable/reference/random/index.html
    rng = default_rng()

    # Random wind realizations
    Ns = 10  # number of wind samples
    beta_realizations = beta(vtm,Ns)
    Ebeta = np.mean(beta_realizations,0)
    
    # Beta FS
    beta_fs = np.fft.rfft(beta_realizations, nfft, axis=-1)*dt  # real fft
    
    
    # velocity realizations
    v_realizations = muV+beta_realizations
     
    # empirical velocity average
    Ev = np.mean(v_realizations)
    v_fs = np.fft.rfft(v_realizations-Ev, nfft, axis=-1)*dt  # real fft
    Sv_a = np.abs(v_fs)**2/2.0/np.pi/dur  # approximated PSD
    

    # Linearized Mean pressure field
    rho_a = 1.2  # kg/m**3
    Cd = 0.8
    mup = 0.5*rho_a*Cd*muV**2

    # theoretical PSD
    Sp_t = lambda f: 4.0*(mup/muV)**2*Sv(f)
    # theoretical linearized pressure field
    p_realizations = 0.5*rho_a*Cd*muV**2*(1.0+2.0*beta_realizations/muV)
    # empirical pressure average
    Ep = np.mean(p_realizations)

    p_fs = np.fft.rfft(p_realizations-Ep, nfft, axis=-1)*dt  # real fft
    Sp_a = np.abs(p_fs)**2/2.0/np.pi/dur

    for i,v in enumerate(v_realizations):

        np.savetxt('wind_velocity_realization_{:d}.csv'.format(i),
                   np.vstack([vtm, v]).T,
                   fmt='%15.5f',
                   delimiter=' ')
        np.savetxt('wind_pressure_realization_{:d}.csv'.format(i),
                   np.vstack([vtm, v]).T,
                   fmt='%15.5f',
                   delimiter=' ')
    # plot()
    
def plot():
    
    fig, ax = plt.subplots(1, 1)
    ax.loglog(vfq, Sv(vfq), color='black')
    ax.set(xlim=(0.01, 10.0),
           xlabel=r"$f$ [Hz]",
           ylabel=r"$S_V(f)$ [m/s/s]")
    ax.axis('tight')

    fig.savefig("PSD_wind.png",
                format="png",
                bbox_inches='tight',
                dpi=300)
    plt.close()
    
    fig, ax = plt.subplots(1, 1)
    ax.plot(vtm, beta_realizations.T)
    ax.set(xlim=(0.0, dur),
           xlabel=r"$t$ [s]",
           ylabel=r"$\beta$ [m/s]")
    ax.axis('tight')
    fig.savefig("beta_random_time_realizations.png",
                format="png",
                bbox_inches='tight',
                dpi=300)
    plt.close()
    
    fig, ax = plt.subplots(1, 1)
    ax.loglog(vfq, np.abs(beta_fs.T))
    ax.set(xlabel=r"$f$ [Hz]",
           ylabel=r"$\vert\mathcal{F}(\beta)\vert$ [m]")
    ax.axis('tight')
    fig.savefig("beta_random_frequency_realizations.png",
                format="png",
                bbox_inches='tight',
                dpi=300)
    plt.close()
    
    fig, ax = plt.subplots(1, 1)
    ax.loglog(vfq, Sv_a.T,color='lightgrey',
              linewidth=0.1)
    ax.loglog(vfq, np.mean(Sv_a,0),
              label=r"$S_v^{ap}$",
              color='grey')
    ax.loglog(vfq, Sv(vfq),
              label=r"$S_v^{th}$",
              color='black',
              linestyle='--')
    ax.legend(frameon=False)
    ax.set(xlabel=r"$f$ [Hz]",
           ylabel=r"$S(f)$ [m/s/s]")
    fig.savefig("Sv_theoretical_empirical.png",
                format="png",
                bbox_inches='tight',
                dpi=300)
    plt.close()
    
    fig, ax = plt.subplots(1, 1)
    ax.loglog(vfq, Sp_a.T, color='lightgrey', linewidth=0.5)
    ax.loglog(vfq, np.mean(Sp_a, 0),
               label=r'$S_p^{ap}$', color='grey', linewidth=0.5)
    ax.loglog(vfq, Sp_t(vfq), label=r'$S_p^{th}$', color='black', linestyle='--')
    ax.legend(frameon=False)
    ax.set(xlabel=r"$f$ [Hz]", 
           ylabel=r"$S(f)$  [N$^2$s/m$^4$]")
    plt.close()