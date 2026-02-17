import torch
from magpy import FunctionProduct as FP, X, Y, Z, I, timegrid, frob, evolve
from torch import sin, cos
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

H = sin*X() + (lambda t: 0.5 * t)*Y()

rho0 = 0.5*(I() + Z())
tlist = timegrid(0, 10, 0.5**6)

observables = {'x': lambda rho, _: rho.expected(X()),
               'y': lambda rho, _: rho.expected(Y())}

_, obsvalues, _ = evolve(H, rho0, tlist, observables)

plt.plot(tlist, obsvalues['x'][0])
plt.plot(tlist, obsvalues['y'][0])
plt.show()
