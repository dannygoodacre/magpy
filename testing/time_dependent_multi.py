from magpy import X, Y, evolve, timegrid, frob, Z
from torch import sin, tensor, stack, cos
import matplotlib.pyplot as plt

# H = (lambda t: stack((sin(t), cos(t))))*X() + [1, 2]*Y()
H = sin*X() + [1,2]*Y()
rho0 = X()
tlist= timegrid(0, 10, 0.05**6)
n_qubits = 1
observables = {'y': lambda u, t: u.expected(Y()),
               'x': lambda u, t: u.expected(X())}

_, obsvalue, _ = evolve(H, rho0, tlist, observables)

plt.plot(tlist, obsvalue['x'][0])
plt.plot(tlist, obsvalue['y'][0])

plt.plot(tlist, obsvalue['x'][1])
plt.plot(tlist, obsvalue['y'][1])

plt.show()
