"""This file implements the Magnus expansion to solve the Liouville-von Neumann
equation, using the quadrature formulae described by Iserles et al.

References
----------

.. [1] Magnus, W. (1954), "On the exponential solution of differential
       equations for a linear operator", *Comm. Pure Appl. Math.* 7, 649-673.

.. [2] Iserles, A., Munthe-Kaas, H. Z., Nørsett, S. P. & Zanna, A. (2000),
       "Lie-group methods", *Acta Numerica* 9, 215-365.

"""


from math import sqrt
import torch
from .core import PauliString, Id, HamiltonianOperator
from ._device import _DEVICE_CONTEXT


_KNOTS = torch.tensor([-sqrt(3/5), 0, sqrt(3/5)], dtype=torch.complex128)
_WEIGHTS = torch.tensor([5/9, 8/9, 5/9])


def _update_device():
    global _KNOTS, _WEIGHTS

    _KNOTS = _KNOTS.to(_DEVICE_CONTEXT.device)
    _WEIGHTS = _WEIGHTS.to(_DEVICE_CONTEXT.device)


def evolve(H: HamiltonianOperator, rho0: PauliString, tlist: torch.Tensor, n_qubits: int = None) -> torch.Tensor:
    """Liouville-von Neumann evolution of the density matrix under a given Hamiltonian.
    
    Evolve the density matrix `rho0` using the Hamiltonian `H`.

    The result is the density matrix evaluated at each time point in `tlist`.

    When the Hamiltonian describes a batch system, the respective system's result is 
    accessed by indexing the result accordingly.
    
    The number of qubits `n_qubits` determines the number of qubits to use in each
    batch of the simulation. By default, MagPy will infer this value.

    Examples
    --------
    >>> H = torch.sin*X() + torch.tensor([1,2], dtype=torch.complex128)*Y()
    >>> rho0 = X()
    >>> tlist = timegrid(0, 10, 0.01)
    >>> states = evolve(H, rho0, tlist)
    >>> y1 = frobenius(states[0], Y().matrix())
    >>> y2 = frobenius(states[1], Y().matrix())
    
    >>> H = torch.cosY() + torch.sin*X(2) + Y()
    >>> rho0 = X(2)
    >>> tlist = timegrid(0, 10, 0.01)
    >>> states = evolve(H, rho0, tlist)
    >>> y2 = frobenius(states[0], Y(2).matrix())

    Parameters
    ----------
    H : HamiltonianOperator
        The Hamiltonian operator
    rho0 : PauliString
        The initial density matrix
    tlist : Tensor
        The list of times
    n_qubits : int, optional
        The number of qubits, by default None

    Returns
    -------
    Tensor
        The states of the system(s) over time
    """

    if isinstance(H, PauliString):
        H = HamiltonianOperator([1, H])
        
    if n_qubits is None:
        n_qubits = max(max(p.qubits.keys()) for p in H.pauli_operators())
        
    batch_count = H.batch_count()

    dim = 2 ** n_qubits
        
    if batch_count > 1:
        states = torch.empty((len(tlist), batch_count, dim, dim), dtype=torch.complex128)

    else:
        states = torch.empty((len(tlist), dim, dim), dtype=torch.complex128)

    states[0] = rho0()

    dt = tlist[1] - tlist[0]

    if H.is_constant():
        states = _evolve_time_independent(H, tlist, dt, states)
    else:
        states = _evolve_time_dependent(H, tlist, states, n_qubits)
        
    if batch_count > 1:
        return states.permute(1, 0, 2, 3)
    
    return states.unsqueeze(0)


def _evolve_time_independent(H, tlist, dt, states):
    u = torch.matrix_exp(-1j * dt * H())
    ut = u.transpose(-2, -1).conj()

    for i in range(len(tlist) - 1):
        states[i + 1] = u @ states[i] @ ut

    return states


def _evolve_time_dependent(H, tlist, states, n_qubits):
    step = tlist[1] - tlist[0]

    funcs = H.funcs()
    ops = H.pauli_operators()

    for i in range(len(tlist) - 1):
        knots = tlist[i] + 0.5*step*(1 + _KNOTS)

        first_term = _first_term(H, knots, step, n_qubits)
        second_term = _second_term(funcs, ops, knots, step, n_qubits)

        u = torch.matrix_exp(-1j * (first_term - 0.5*second_term))
        
        ut = u.transpose(-2, -1).conj()
        
        states[i + 1] = u @ states[i] @ ut

    return states
 
    
def _first_term(H, knots, step, n_qubits):
    result = 0

    for f, p in H.unpack_data():
        result += torch.sum(_WEIGHTS * (f(knots) if callable(f) else f)) * p(n_qubits)
 
    return 0.5 * step * result


def _second_term(funcs, ops, knots, step, n_qubits):
    result = 0
    
    n = len(funcs)

    for i in range(n):
        for j in range(i + 1, n):
            if ops[i] == Id() or ops[j] == Id():
                continue

            if callable(funcs[i]):
                fi0 = funcs[i](knots[0])
                fi1 = funcs[i](knots[1])
                fi2 = funcs[i](knots[2])
            
            else:
                fi0 = funcs[i]
                fi1 = funcs[i]
                fi2 = funcs[i]

            if callable(funcs[j]):
                fj0 = funcs[j](knots[0])
                fj1 = funcs[j](knots[1])
                fj2 = funcs[j](knots[2])
            
            else:
                fj0 = funcs[j]
                fj1 = funcs[j]
                fj2 = funcs[j]

            coeff = 2*(fi0*fj1 - fi1*fj0) + (fi0*fj2 - fi2*fj0) + 2*(fi1*fj2 - fi2*fj1)

            op = ops[i]*ops[j] - ops[j]*ops[i]
            
            result += coeff * op(n_qubits)

    return (sqrt(15) / 54) * step**2 * result
