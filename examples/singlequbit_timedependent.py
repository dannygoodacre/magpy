from magpy import FunctionProduct as FP, X, Y, timegrid, frob, evolve
from torch import sin, sqrt
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

# H(t) = 10sin(t)X1 + sqrt(t)Y1
# rho0 = Y1

# MagPy
H = FP()*10*sin*X() + sqrt*Y()
rho0 = Y()
tlist = timegrid(0, 10, 0.5**8)

observables = {'x': lambda u, _: frob(u.matrix(), X().tensor()),
               'y': lambda u, _: frob(u.matrix(), Y().tensor())}

_, obsvalues, _ = evolve(H, rho0, tlist, observables=observables)

plt.plot(tlist, obsvalues['x'][0])
plt.plot(tlist, obsvalues['y'][0])

# QuTiP
H = [[qt.sigmax(), (lambda t: 10 * np.sin(t))], [qt.sigmay(), (lambda t: np.sqrt(t))]]
rho0 = qt.sigmay()
tlist = np.linspace(0, 10, 1_000_000)

result = qt.mesolve(H, rho0, tlist, e_ops=[qt.sigmax(), qt.sigmay()])

plt.plot(tlist, result.expect[0])
plt.plot(tlist, result.expect[1])

plt.show()
