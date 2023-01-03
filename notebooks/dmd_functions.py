## -*- coding: utf-8 -*-
#!/usr/bin/env python3
u"""General informations"""
__author__ = "Filippo Gatti"
__license__ = "GPL"
__version__ = "1.0.1"
__Maintainer__ = "Filippo Gatti"
__email__ = "filippo.gatti@centralesupelec.fr"
__status__ = "Beta"

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import dot, multiply, diag, power
from numpy import pi, exp, sin, cos, cosh, tanh, real, imag
from numpy.linalg import inv, eig, pinv
from scipy.linalg import svd, svdvals
from scipy.integrate import odeint, ode, complex_ode
from warnings import warn

def nullspace(A, atol=1e-13, rtol=0):
    # from http://scipy-cookbook.readthedocs.io/items/RankNullspace.html
    A = np.atleast_2d(A)
    u, s, vh = svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns

def check_linear_consistency(X, Y, show_warning=True):
    # tests linear consistency of two matrices (i.e., whenever Xc=0, then Yc=0)
    A = dot(Y, nullspace(X))
    total = A.shape[1]
    z = np.zeros([total, 1])
    fails = 0
    for i in range(total):
        if not np.allclose(z, A[:,i]):
            fails += 1
    if fails > 0 and show_warning:
        warn('linear consistency check failed {} out of {}'.format(fails, total))
    return fails, total

def dmd(X, Y, truncate=None):
    U2,Sig2,Vh2 = svd(X, False) # SVD of input matrix
    r = len(Sig2) if truncate is None else truncate # rank truncation
    U = U2[:,:r]
    Sig = diag(Sig2)[:r,:r]
    V = Vh2.conj().T[:,:r]
    Atil = dot(dot(dot(U.conj().T, Y), V), inv(Sig)) # build A tilde
    mu,W = eig(Atil)
    Phi = dot(dot(dot(Y, V), inv(Sig)), W) # build DMD modes
    return mu, Phi

def check_dmd_result(X, Y, mu, Phi, show_warning=True):
    b = np.allclose(Y, dot(dot(dot(Phi, diag(mu)), pinv(Phi)), X))
    if not b and show_warning:
        warn('dmd result does not satisfy Y=AX')

def modal_time_evolution(mu, Phi, dtm, vtm, ic, r):
    b = dot(pinv(Phi), ic)
    ntm = len(vtm)
    Psi = np.zeros((r, ntm), dtype='complex')
    for i,_t in enumerate(vtm):
        Psi[:,i] = multiply(power(mu, _t/dtm), b)
    Dtil = real(dot(Phi, Psi))
    return Psi, Dtil
