import sympy as sp
from sympy.vector import CoordSys3D
from sympy.plotting import plot
from sympy import besselj as Jb
from sympy import bessely as Yb
from sympy import pi
from sympy import Array, Matrix, Sum, MatrixSymbol
# from sympy import symbols
from sympy.abc import R, h, L, x, n, m, N, M
from sympy.abc import E
from sympy import Function, sin, cos
from sympy.solvers.solveset import linsolve


import pdb
"""_summary_
     (1)Dai, L.; Yang, T.; Du, J.; Li, W. L.; Brennan, M. J. 
     An Exact Series Solution for the Vibration Analysis of Cylindrical Shells
     with Arbitrary Boundary Conditions. Applied Acoustics 2013, 74 (3), 440-449. 
     https://doi.org/10.1016/j.apacoust.2012.09.001.
"""
# Reference system
Ref = CoordSys3D('Oxyz', 
                 vector_names=("ix", "iy", "iz")
                )
C = Ref.create_new('Orθz',
                    transformation='cylindrical',
                    variable_names=("r", "θ", "z"),
                    vector_names=("ir", "iθ", "iz")
                  )

ur = Function("ur")(C.r, C.θ, C.z)
uθ = Function("uθ")(C.r, C.θ, C.z)
uz = Function("uz")(C.r, C.θ, C.z)

N = 12  # number of circumferential modes
M = 10  # number of longitudinal modes

# general symbols and variables
(π, ν, ω, σ, γ) = sp.symbols('π ν ω σ γ', positive=True)
π = pi
# λm = π*m/L # wavelength of the m-th tuned harmonic
λ_s = sp.symbols("λ_0:{:d}".format(M+1))
λm = Matrix(M+1, 1, λ_s)

λm2 = sp.diag(*λ_s)*λm
K = E*h/(1.0-σ**2) # in-plane stifness
D = E*h**3/12.0/(1.0-σ**2) # flexural stiffness
γ = h**2/12.0/R**2


# Boundary conditions stiffnesses
(k1, k2, k3, k4, k5, k6, k7, k8) = sp.symbols('k1 k2 k3 k4 k5 k6 k7 k8')

# α and β vectors
α1 = x*(x/L-1)**2
α2 = (x**2/L)*(x/L-1.0)
β1 = +(9.0*L/4.0/π)*sin(π*x/2.0/L)-(L/12.0/π)*sin(3.0*π*x/2.0/L)
β2 = -(9.0*L/4.0/π)*cos(π*x/2.0/L)-(L/12.0/π)*cos(3.0*π*x/2.0/L)
β3 = +(L/π)**3*sin(π*x/2.0/L)-(L/π)**3/3.0*sin(3.0*π*x/2.0/L)
β4 = -(L/π)**3*cos(π*x/2.0/L)-(L/π)**3/3.0*cos(3.0*π*x/2.0/L)

α = Array([α1, α2])
β = Array([β1, β2, β3, β4])
# Fourier's expansion coefficients

a_s = sp.symbols("a_0:{:d}".format(N+1))
b_s = sp.symbols("b_0:{:d}".format(N+1))
c_s = sp.symbols("c_0:{:d}".format(N+1))
d_s = sp.symbols("d_0:{:d}".format(N+1))
e_s = sp.symbols("e_0:{:d}".format(N+1))
f_s = sp.symbols("f_0:{:d}".format(N+1))
g_s = sp.symbols("g_0:{:d}".format(N+1))
h_s = sp.symbols("h_0:{:d}".format(N+1))
an = Matrix(N+1, 1, a_s)
bn = Matrix(N+1, 1, b_s)
cn = Matrix(N+1, 1, c_s)
dn = Matrix(N+1, 1, d_s)
en = Matrix(N+1, 1, e_s)
fn = Matrix(N+1, 1, f_s)
gn = Matrix(N+1, 1, g_s)
hn = Matrix(N+1, 1, h_s)

# an = ArraySymbol("a", (n,))
Amn = Matrix(M+1, N+1, sp.symbols('A_0:{:d}(0:{:d})'.format(M+1, N+1)))
Bmn = Matrix(M+1, N+1, sp.symbols('B_0:{:d}(0:{:d})'.format(M+1, N+1)))
Cmn = Matrix(M+1, N+1, sp.symbols('C_0:{:d}(0:{:d})'.format(M+1, N+1)))
ΣAn = sp.ones(1,Amn.shape[0])*Amn
ΣBn = sp.ones(1,Bmn.shape[0])*Bmn
ΣCn = sp.ones(1,Cmn.shape[0])*Cmn
eq6a = an-\
     (7.0*σ*L/(3.0*π*R)+3.0*π*R*γ/(4.0*L))*fn-\
     (4.0*σ*L**3/(3.0*π**3*R)+L*γ*R/π)*hn-\
    -((k1/K)*ΣAn-(σ*n/R)*ΣBn-(σ/R)*ΣCn-γ*R*λm2.T*Cmn).T
pdb.set_trace()


    #Σ(Av[m], (m, 0, M))
# lhs = Array([],
#             (,))
# abcdefgh = Array([av, bv, cv, dv, ev, fv, gv, hv], (8*len(av),))
# ABC = Array([Av, Bv, Cv], (3*len(Av),))



u = C.ir*ur+C.iθ*uθ+C.iz*uz

plot(R1(r, 0.01, 1), (r, 0.0, 1000.0))
