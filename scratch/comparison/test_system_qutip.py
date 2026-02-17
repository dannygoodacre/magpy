import qutip as qt
import numpy as np

b = 5; c = 2; om = 1

def e(t):
    return b * np.exp(-(t - 10)**8/ 10**7)

def w(t):
    return np.exp(1j * c * (t - 10)**2)

def f(t):
    return e(t) * np.real(w(t))

def g(t):
    return e(t) * np.imag(w(t))

H = [om*qt.sigmaz(), [qt.sigmax(), f], [qt.sigmay(), g]]
rho0 = 0.5*(qt.identity(2) + qt.sigmax())

start = 0
end = 20
