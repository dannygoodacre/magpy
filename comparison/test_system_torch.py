import magpy as mp
import torch

b = 5; c = 2; om = 1

def e(t):
    return b * torch.exp(-(t - 10)**8/ 10**7)

def w(t):
    return torch.exp(1j * c * (t - 10)**2)

def f(t):
    return e(t) * torch.real(w(t))

def g(t):
    return e(t) * torch.imag(w(t))

H = f*mp.X() + g*mp.Y() + om*mp.Z()
rho0 = 0.5*(mp.Id() + mp.X())

start = 0
end = 20
