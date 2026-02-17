from magpy import X, Y, Z, I, evolve, FunctionProduct as FP, timegrid, frob
import matplotlib.pyplot as plt

H = (3,4,5)*Y()
rho0 = X()
tlist = timegrid(0, 5, 0.01)
observables = {'x': lambda u, t: frob(u.tensor(), X().tensor())}

_, obsvalue, intermediate = evolve(H, rho0, tlist, observables=observables, store_intermediate=True)

print(len(intermediate))

for i in range(10):
    print(intermediate[i])

plt.plot(tlist, obsvalue['x'][0])
plt.plot(tlist, obsvalue['x'][1])
plt.plot(tlist, obsvalue['x'][2])
plt.show()
