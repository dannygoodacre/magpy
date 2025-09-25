import magpy as mp
import torch
from test_system_torch import H, rho0, start, end

for k in range(4, 17):
    print(k)

    tlist = mp.timegrid(start, end, 0.5**k)
    
    states = mp.evolve(H, rho0, tlist)
    
    torch.save(states, f'test_data_2/magpy_{k}.pt')
