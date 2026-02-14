from magpy import FunctionProduct as FP, X, Y, Z, timegrid, frob, new_evolve
from torch import sin, sqrt
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

H = FP()*10*sin*X() + sqrt*Y()

# H(t) = 10*sin(t)*X + sqrt(t)*Y

rho0 = Y()
tlist = timegrid(0, 10, 0.5**6)

observables = {'x': lambda u, _: frob(u, X().tensor()),
               'y': lambda u, _: frob(u, Y().tensor()),
               'z': lambda u, _: frob(u, Z().tensor())}

_, obsvalues, _ = new_evolve(H, rho0, tlist, observables=observables)

plt.plot(tlist, obsvalues['x'][0])
# plt.plot(tlist, obsvalues['y'][0])
# plt.plot(tlist, obsvalues['z'][0])
plt.show()
