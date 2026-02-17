import qutip as qt
import magpy as mp
import torch
from test_system_qutip import H, rho0, start, end

tlist = mp.timegrid(start, end, 0.5**20)

states = torch.stack(tuple(torch.from_numpy(state.full()) for state in qt.mesolve(H, rho0, tlist, progress_bar=True).states))

torch.save(states, 'test_data_2/qutip_20.pt')
