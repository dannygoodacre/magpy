from magpy import FunctionProduct as FP, X, Y, Z, timegrid, frob, evolve
from torch import sin, cos
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

H = sin*X() + cos*Y()

# H(t) = 10*sin(t)*X + sqrt(t)*Y

rho0 = Y()
tlist = timegrid(0, 10, 0.5**6)

observables = {'x': lambda u, _: frob(u.tensor(), X().tensor()),
               'y': lambda u, _: frob(u.tensor(), Y().tensor()),
               'z': lambda u, _: frob(u.tensor(), Z().tensor())}

_, obsvalues, _ = evolve(H, rho0, tlist, observables)

plt.plot(tlist, obsvalues['x'][0])
plt.plot(tlist, obsvalues['y'][0])
plt.plot(tlist, obsvalues['z'][0])
plt.show()
