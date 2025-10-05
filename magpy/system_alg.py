import torch
from torch import Tensor
import expsolve as es

from .core import PauliString, HamiltonianOperator


def evolve_new(H, rho0: PauliString, tlist: Tensor, n_qubits: int = None):
    h = tlist[1] - tlist[0]

    u = H.propagator(h)
    uh = u.H

    print(u.matrix())
    print(uh.matrix())
    input()

    stepper = lambda _, __, rho: u * rho * uh

    rho, obsvalues, states = es.solvediffeq(rho0, tlist, stepper, storeintermediate=True)

    return rho, obsvalues, states
