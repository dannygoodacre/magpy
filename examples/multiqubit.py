from magpy import FunctionProduct as FP, X, Y, timegrid, frob, evolve
from torch import tensor, sin, cos
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

# H(t) = 10sin(t)X1 + 3cos(t)Y2
# rho0 = Y1*X2

# MagPy
H = FP()*10*sin*X() + FP()*3*cos*Y(2)

G = H - sin*X()

# TODO: When a function product is a single function with no coefficient, it should be 'demoted' to just the function.
print(type(list(G._data.keys())[0]))
exit(0)

rho0 = Y()*X(2)
tlist = timegrid(0, 10, 0.5**8)
n_qubits = 2

observables = {'x': lambda u, _: frob(u.matrix(n_qubits), X().matrix(n_qubits)),
               'y': lambda u, _: frob(u.matrix(n_qubits), Y().matrix(n_qubits))}

_, obsvalues, _ = evolve(H, rho0, tlist, n_qubits, observables)

plt.plot(tlist, obsvalues['x'][0])
plt.plot(tlist, obsvalues['y'][0])

plt.show()
