import sympy as sp
from sympy.vector import CoordSys3D
from sympy.plotting import plot
from sympy import besselj as Jb
from sympy import bessely as Yb
from sympy import pi
from sympy import Array, Matrix
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


(π, ν, ω, σ, λm, γ) = sp.symbols('π ν ω σ λm γ')
N = 10
M = 10

# (an, bn, cn, dn, en, fn, gn, hn) = sp.symbols('an bn cn dn en fn gn hn')
# (α1, α2, β1, β2, β3, β4) = sp.symbols('α1 α2 β1 β2 β3 β4')

π = pi
λm = π*m/L # wavelength of the m-th tuned harmonic
K = E*h/(1.0-σ**2) # in-plane stifness
D = E*h**3/12.0/(1.0-σ**2) # flexural stiffness
γ = h**2/12.0/R**2


R = CoordSys3D('Oxyz', 
               vector_names=("ix", "iy", "iz")
               )
C = R.create_new('Orθz',
                 transformation='cylindrical',
                 variable_names=("r", "θ", "z"),
                 vector_names=("ir", "iθ", "iz")
                 )

ur = Function("ur")(C.r, C.θ, C.z)
uθ = Function("uθ")(C.r, C.θ, C.z)
uz = Function("uz")(C.r, C.θ, C.z)

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


av = sp.symbols("a_0:{:d}".format(N+1))
bv = sp.symbols("b_0:{:d}".format(N+1))
cv = sp.symbols("c_0:{:d}".format(N+1))
dv = sp.symbols("d_0:{:d}".format(N+1))
ev = sp.symbols("e_0:{:d}".format(N+1))
fv = sp.symbols("f_0:{:d}".format(N+1))
gv = sp.symbols("g_0:{:d}".format(N+1))
hv = sp.symbols("h_0:{:d}".format(N+1))
abcdefgh = Array([av, bv, cv, dv, ev, fv, gv, hv], (8*len(av),))

Av = sp.symbols("A_0:{:d}".format((M+1)*(N+1)))
Bv = sp.symbols("B_0:{:d}".format((M+1)*(N+1)))
Cv = sp.symbols("C_0:{:d}".format((M+1)*(N+1)))

ABC = Array([Av, Bv, Cv], (3*len(Av),))

pdb.set_trace()
# linsolve([x + y + z - 1, 
#           x + y + 2*z - 3], (x, y, z))


u = C.ir*ur+C.iθ*uθ+C.iz*uz

plot(R1(r, 0.01, 1), (r, 0.0, 1000.0))
