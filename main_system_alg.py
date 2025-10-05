import magpy as mp
import torch
import matplotlib.pyplot as plt

# mp.set_print_precision(4)

H = 3*mp.X() + 4*mp.Y()
rho0 = mp.Y()
tlist = mp.timegrid(0, 10, 0.5**6)

rho, _, states = mp.evolve_new(H, rho0, tlist)

states = [state.matrix() for state in states]

x = mp.frob(torch.stack(states), mp.Y().matrix())
plt.plot(tlist, x)

# print(H)
# rho, _, states = mp.evolve(H, rho0, tlist, store_intermediate=True)

# x = mp.frob(torch.stack(states), mp.Y().matrix())
# plt.plot(tlist, x)

plt.show()
