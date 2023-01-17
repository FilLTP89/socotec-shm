from sympy.plotting import plot
from sympy import Function, Lambda, plot_parametric
from sympy import besselj as Jν
from sympy import bessely as Yν
from sympy import symbols
from sympy.abc import n, r, z

(θ, ν) = symbols('θ ν')
(ur, uθ) = symbols('ur uθ')
(α1, α2, α23, α4) = symbols('α1 α2 α23 α4')

F1 = 1.0
G1 = 1.0

# R1 = Function("R1")
R1 = Lambda((r, α1, ν), F1*Jν(ν, α1*r) + G1*Yν(ν, α1*r))
import pdb
pdb.set_trace()
plot(R1(r, 0.01, 1), (r, 0.0, 1000.0))
