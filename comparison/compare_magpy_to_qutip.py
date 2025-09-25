import magpy as mp
import torch
import matplotlib.pyplot as plt
import numpy as np

ref_k = 20
ref = torch.load('test_data_2/qutip_20.pt')

errors = []
steps = []

for sim_k in range(4, 17):
    sim = torch.load(f'test_data_2/magpy_{sim_k}.pt')[0]
    sim_tlist = mp.timegrid(0, 20, 0.5**sim_k)

    indices = [(2**(ref_k - sim_k)) * i for i in range(len(sim_tlist))]

    ref_ = ref[indices]

    # ref_x = mp.frob(ref_, mp.X().matrix()).real
    # sim_x = mp.frob(sim, mp.X().matrix()).real
    # print(torch.max(ref_x - sim_x))

    errors.append(torch.mean(torch.sqrt(mp.frob(ref_ - sim, ref_ - sim).real)))
    steps.append(0.5**sim_k)

log_errors = np.log10(errors[:9])
log_steps = np.log10(steps[:9])

slope, intercept = np.polyfit(log_steps, log_errors, 1)

print(slope)

plt.loglog(steps[:9], errors[:9])

plt.show()
