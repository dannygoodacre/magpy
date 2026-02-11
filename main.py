import magpy as mp
from magpy import FunctionProduct as FP
import matplotlib.pyplot as plt
from torch import tensor, sin
import torch
import qutip as qt
import numpy as np

# H = tensor([1, 10, 100])*mp.X()
# H = 3 * mp.X()

# The scale factor isn't coming in somewhere
def g(t): return 10 * sin(t)

H = FP()*10*sin*mp.X() + FP()*torch.sqrt*mp.Y()

rho0 = mp.Y()
tlist = mp.timegrid(0, 10, 0.5**6)

n_qubits = 1
observables = {'y': lambda u, _: mp.frob(u.matrix(n_qubits=n_qubits), mp.Y().tensor(n_qubits=n_qubits)),
               'x': lambda u, _: mp.frob(u.matrix(n_qubits=n_qubits), mp.X().tensor(n_qubits=n_qubits))}

rho, obsvalues, states = mp.evolve(H, rho0, tlist, n_qubits, observables, True)

plt.plot(tlist, obsvalues['y'][0])
# plt.plot(tlist, obsvalues['x'][0])

# plt.plot(tlist, mp.frob(torch.stack([state.matrix(n_qubits=n_qubits) for state in states]), mp.Y().matrix(n_qubits=n_qubits)))
# plt.plot(tlist, mp.frob(torch.stack([state.matrix(n_qubits=n_qubits) for state in states]), mp.X().matrix(n_qubits=n_qubits)))

# ---
def f(t): return 10 * np.sin(t)
def g(t): return np.sqrt(t)

H = [[qt.sigmax(), f], [qt.sigmay(), g]]
rho0 = qt.sigmay()

tlist = np.linspace(0, 10, 10000)

result = qt.mesolve(H, rho0, tlist, None, [qt.sigmay(), qt.sigmax()])

plt.plot(tlist, result.expect[0])
# ---

plt.show()
