import sympy as sp
from sympy.vector import CoordSys3D
from sympy.plotting import plot
from sympy import besselj as Jb
from sympy import bessely as Yb
# from sympy import symbols
# from sympy.abc import r, z
from sympy import vectorize, diff, sin, cos

import pdb
     """_summary_
     (1) Mofakhami, M. R.; Toudeshky, H. H.; Hashemi, Sh. H. 
     Finite Cylinder Vibrations with Different End Boundary Conditions. 
     Journal of Sound and Vibration 2006, 297 (1-2), 293-314. 
     https://doi.org/10.1016/j.jsv.2006.03.041.
     """



(ν, ω) = sp.symbols('ν ω')
(α, α1, α2, α23, α4) = sp.symbols('α α1 α2 α23 α4')
(δ, δ1, δ2, δ23, δ4) = sp.symbols('δ δ1 δ2 δ23 δ4')
(M1, M2, M3, M4) = sp.symbols('M1 M2 M3 M4')
(F1, G1, F2, G2, F3, G3, F4, G4) = sp.symbols('F1 G1 F2 G2 F3 G3 F4 G4')


R = CoordSys3D('Oxyz', 
               vector_names=("ix", "iy", "iz")
               )
C = R.create_new('Orθz',
                 transformation='cylindrical',
                 variable_names=("r", "θ", "z"),
                 vector_names=("ir", "iθ", "iz")
                 )

ur = sp.Function("ur")(C.r, C.θ, C.z)
uθ = sp.Function("uθ")(C.r, C.θ, C.z)
uz = sp.Function("uz")(C.r, C.θ, C.z)


A1 = F1*(1.0+M1)*(C.ir+C.iθ)
A2 = F3*(1.0+M3)*(C.ir+C.iθ)
A3 = F4*(1.0+M4)*(C.ir+C.iθ) + \
     δ2/α*(F2*(1.0+M2)-F3*(1.0+M3))*(C.ir-C.iθ)
A4 = sp.conjugate(F1)*(1.0+M1)*(C.ir+C.iθ)
A5 = sp.conjugate(F3)*(1.0+M3)*(C.ir+C.iθ)
A6 = sp.conjugate(F4)*(1.0+M4)*(C.ir+C.iθ) + \
     sp.conjugate(δ2)/sp.conjugate(α2)*(
                                        sp.conjugate(F2)*(1.0+M2)-
                                        sp.conjugate(F3)*(1.0+M3)
                                        )*(C.ir-C.iθ)
B1 = G1*(1.0+M1)*(C.ir+C.iθ)
B2 = G3*(1.0+M3)*(C.ir+C.iθ)
B3 = G4*(1.0+M4)*(C.ir+C.iθ) + \
     δ2/α*(G2*(1.0+M2)-G3*(1.0+M3))*(C.ir-C.iθ)
B4 = sp.conjugate(G1)*(1.0+M1)*(C.ir+C.iθ)
B5 = sp.conjugate(G3)*(1.0+M3)*(C.ir+C.iθ)
B6 = sp.conjugate(G4)*(1.0+M4)*(C.ir+C.iθ) + \
     sp.conjugate(δ2)/sp.conjugate(α2)*(
                                        sp.conjugate(G2)*(1.0+M2) -
                                        sp.conjugate(G3)*(1.0+M3)
                                        )*(C.ir-C.iθ)
R1 = F1*Jb(ν, α1*C.r) + G1*Yb(ν, α1*C.r)
R2 = F2*Jb(ν+1, α23*C.r) + G2*Yb(ν+1, α23*C.r) + \
     (ν/α23/C.r)*(
                  (F3-F2)*Jb(ν, α23*C.r) +
                  (G3-G2)*Yb(ν, α23*C.r)
                  )
R3 = F3*Jb(ν+1, α23*C.r) + G3*Yb(ν+1, α23*C.r) + \
     (ν/α23/C.r)*(
                  (F2-F3)*Jb(ν, α23*C.r) +
                  (G2-G3)*Yb(ν, α23*C.r)
                  )
R4 = F4*Jb(ν, α4*C.r) + G4*Yb(ν, α4*C.r)

T1 = sp.cos(δ1*C.z)*C.ir + sp.sin(δ1*C.z)*C.iθ
T2 = sp.sin(δ23*C.z)*C.ir + sp.cos(δ23*C.z)*C.iθ
T3 = -sp.sin(δ23*C.z)*C.ir + sp.cos(δ23*C.z)*C.iθ
T4 = sp.cos(δ4*C.z)*C.ir + sp.sin(δ4*C.z)*C.iθ

E1 = sp.cos(ν*C.θ)*C.ir + sp.sin(ν*C.θ)*C.iθ
E3 = sp.cos(ν*C.θ)*C.ir + sp.sin(ν*C.θ)*C.iθ
E2 = sp.sin(ν*C.θ)*C.ir + sp.cos(ν*C.θ)*C.iθ
E4 = sp.sin(ν*C.θ)*C.ir + sp.cos(ν*C.θ)*C.iθ

pdb.set_trace()

M4 = M2+M3-1.0
δ1 = δ
δ2 = δ1
δ23 = δ2
δ4 = δ23

α = α2


u = C.ir*ur+C.iθ*uθ+C.iz*uz

plot(R1(r, 0.01, 1), (r, 0.0, 1000.0))
