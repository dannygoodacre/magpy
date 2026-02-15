import torch
from magpy import FunctionProduct as FP, X, Y, Z, I, timegrid, frob, evolve
from torch import sin, cos
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

H = sin*X() + (lambda t: 0.5 * t)*Y()

rho0 = 0.5*(I() + Z())
tlist = timegrid(0, 10, 0.5**6)

observables = {#'x': lambda u, _: frob(u.tensor(), X().tensor()),
            #    'y': lambda u, _: frob(u.tensor(), Y().tensor()),
               'znew': lambda rho, _: rho.expected(Z()),
               'zold': lambda u, _: frob(u.tensor(), Z().tensor())}

_, obsvalues, _ = evolve(H, rho0, tlist, observables)

# plt.plot(tlist, obsvalues['x'][0])
# plt.plot(tlist, obsvalues['y'][0])
plt.plot(tlist, obsvalues['zold'][0])
plt.plot(tlist, obsvalues['znew'][0])
plt.show()
