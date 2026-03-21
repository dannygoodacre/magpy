from magpy import frob
import torch
import matplotlib.pyplot as plt
import numpy as np

data = torch.load("data.pt")

diff = data["result"] - data["reference"]

print(diff)

errors = torch.linalg.norm(diff, ord="fro", dim=(1,2))

print(errors)

dt_values = 0.5**np.arange(4, 13)
print(dt_values)
plt.loglog(dt_values, errors.numpy(), '-o')

plt.xlabel('step size')
plt.ylabel('Frob')
plt.legend()
plt.show()
