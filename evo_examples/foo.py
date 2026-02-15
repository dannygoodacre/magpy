from math import sqrt

import torch

from magpy import HamiltonianOperator, X, Y, Z, I, FunctionProduct as FP, timegrid, frob, evolve, PauliString
import matplotlib.pyplot as plt

def f(t): return torch.sin(t)
def g(t): return torch.exp(-t)

# completely independent
H1 = f*Z(0)
H2 = g*Z(1)
H = H1 + H2

rho0 = (0.5*(I() + X(0))) * (0.5*(I() + X(1)))

tlist = timegrid(0, 10, 0.01)

# observables = {
#     'x0': lambda rho, t: torch.diagonal(rho.tensor(2) @ X(0).tensor(2), dim1=-2, dim2=-1).sum(-1).real,
#     'x1': lambda rho, t: torch.diagonal(rho.tensor(2) @ X(1).tensor(2), dim1=-2, dim2=-1).sum(-1).real
# }

observables = {
    'x0': lambda u, _: frob(u.tensor(), X().tensor(n_qubits=2)),
}

_, obsvalue, _ = evolve(H, rho0, tlist, observables=observables)

plt.plot(tlist, obsvalue['x0'][0])
plt.show()
