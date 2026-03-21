from magpy import X, Y, evolve, timegrid, frob, Z, set_default_device
from torch import sin
import matplotlib.pyplot as plt

# set_default_device('cuda:0')

H = X() + Y()
rho0 = X()
tlist = timegrid(0, 10, 0.01)
n_qubits = 1
observables = {'y': lambda u, t: frob(u.tensor(), Y().tensor()),
               'x': lambda u, t: frob(u.tensor(), X().tensor())}

_, obsvalue, _ = evolve(H, rho0, tlist, observables=observables)

plt.plot(tlist.cpu(), obsvalue['x'][0].cpu())
plt.plot(tlist.cpu(), obsvalue['y'][0].cpu())
plt.show()
