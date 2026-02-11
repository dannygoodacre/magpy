from magpy import X, Y, evolve, timegrid, frob, Z
from torch import sin, tensor
import matplotlib.pyplot as plt

H = tensor([1,2])*X() + Y()
rho0 = X()
tlist= timegrid(0, 10, 0.01)
n_qubits = 1
observables = {'y': lambda u, t: frob(u.matrix(), Y().tensor()),
               'x': lambda u, t: frob(u.matrix(), X().tensor())}

_, obsvalue, _ = evolve(H, rho0, tlist, observables=observables)

plt.plot(tlist, obsvalue['x'][0])
plt.plot(tlist, obsvalue['y'][0])
plt.plot(tlist, obsvalue['x'][1])
plt.plot(tlist, obsvalue['y'][1])
plt.show()
