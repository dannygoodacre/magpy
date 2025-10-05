from magpy import X, Y, evolve, timegrid, frob, Z
import torch
import matplotlib.pyplot as plt

H = torch.sin*X() + torch.tensor([1,2], dtype=torch.complex128)*Y() + torch.cos*Z()
rho0 = X()
tlist = timegrid(0, 10, 0.01)
n_qubits = 1
observables = {'y': lambda u, t: frob(u, Y()(n_qubits)),
               'x': lambda u, t: frob(u, X()(n_qubits)),
               'z': lambda u, t: frob(u, X()(n_qubits))}

_, obsvalue, _ = evolve(H, rho0, tlist, observables=observables)

plt.plot(tlist, obsvalue['x'][0])
plt.plot(tlist, obsvalue['x'][1])
plt.plot(tlist, obsvalue['y'][0])
plt.plot(tlist, obsvalue['y'][1])
plt.show()
