import magpy as mp
import matplotlib.pyplot as plt
from torch import tensor

# H = tensor([1, 10, 100])*mp.X()
H = 3 * mp.X()
rho0 = mp.Y()
tlist = mp.timegrid(0, 10, 0.5**6)

n_qubits = 2
observables = {'y': lambda u, t: mp.frob(u, mp.Y()(n_qubits)),
               'x': lambda u, t: mp.frob(u, mp.X()(n_qubits))}

rho, obsvalues, states = mp.evolve(H, rho0, tlist, n_qubits, observables, True)

print(obsvalues['y'].shape)
print(obsvalues['x'].shape)

print(states.shape)

plt.plot(tlist, obsvalues['y'][0])
plt.plot(tlist, obsvalues['x'][0])
# plt.plot(tlist, obsvalues['y'][1])
# plt.plot(tlist, obsvalues['y'][2])
plt.show()
