from math import sqrt

import torch

from magpy import HamOp, X, Y, Z, I, FunctionProduct as FP, timegrid, frob, new_evolve, PauliString
import matplotlib.pyplot as plt

def f(t): return torch.sin(t)
def g(t): return torch.exp(-t)

# completely independent
H1 = f*Z(0)
H2 = g*Z(1)
H = H1 + H2

print(H.terms)
print(H.coeffs)
print(H.pauli_strings)
exit(0)

rho0 = (0.5*(I(0) + X(0))) * (0.5*(I(1) + X(1)))

tlist = timegrid(0, 10, 0.01)

observables = {
    'x0': lambda rho, t: torch.diagonal(rho.tensor(2) @ X(0).tensor(2), dim1=-2, dim2=-1).sum(-1).real,
    'x1': lambda rho, t: torch.diagonal(rho.tensor(2) @ X(1).tensor(2), dim1=-2, dim2=-1).sum(-1).real
}

_, obsvalue, _ = new_evolve(H, rho0, tlist, observables=observables)

plt.plot(tlist, obsvalue['x0'][0])
plt.plot(tlist, obsvalue['x1'][0])
plt.show()
