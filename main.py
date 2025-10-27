import magpy as mp
from magpy import FunctionProduct as FP
import matplotlib.pyplot as plt
from torch import tensor, sin
import torch

# H = tensor([1, 10, 100])*mp.X()
# H = 3 * mp.X()
H = FP() * 2 * sin * mp.X()
rho0 = mp.Y()
tlist = mp.timegrid(0, 10, 0.5**6)

n_qubits = 2
observables = {'y': lambda u, _: mp.frob(u.matrix(n_qubits=n_qubits), mp.Y().matrix(n_qubits=n_qubits)),
               'x': lambda u, _: mp.frob(u.matrix(n_qubits=n_qubits), mp.X().matrix(n_qubits=n_qubits))}

rho, obsvalues, states = mp.evolve(H, rho0, tlist, n_qubits, observables, True)

print(obsvalues)

plt.plot(tlist, obsvalues['y'][0])
plt.plot(tlist, obsvalues['x'][0])

# plt.plot(tlist, mp.frob(torch.stack([state.matrix(n_qubits=n_qubits) for state in states]), mp.Y().matrix(n_qubits=n_qubits)))
# plt.plot(tlist, mp.frob(torch.stack([state.matrix(n_qubits=n_qubits) for state in states]), mp.X().matrix(n_qubits=n_qubits)))

plt.show()
