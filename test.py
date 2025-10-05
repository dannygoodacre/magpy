import magpy as mp
import torch

H = 3*mp.X() + 4*mp.Y()

print(H.propagator(0.5).matrix())

print(torch.matrix_exp(-1j * 0.5 * H.matrix()))
