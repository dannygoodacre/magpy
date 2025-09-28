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
from numbers import Number
from typing import Callable

import torch
import expsolve as es

from .core import PauliString, Id, HamiltonianOperator
from ._device import _DEVICE_CONTEXT


_KNOTS = torch.tensor([-sqrt(3/5), 0, sqrt(3/5)], dtype=torch.complex128)
_WEIGHTS = torch.tensor([5/9, 8/9, 5/9])


def _update_device():
    global _KNOTS, _WEIGHTS

    _KNOTS = _KNOTS.to(_DEVICE_CONTEXT.device)
    _WEIGHTS = _WEIGHTS.to(_DEVICE_CONTEXT.device)


def evolve(H: HamiltonianOperator, 
           rho0: PauliString, 
           tlist: torch.Tensor, 
           n_qubits: int = None,
           observables: dict[str, Callable[[torch.Tensor, Number], torch.Tensor]] = {},
           store_intermediate: bool = False) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor | None]:
    """Liouville-von Neumann evolution of the density matrix under a given Hamiltonian.
    
    TODO: Update this docstring.
    
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
        
    >>> H = torch.cos*Y() + torch.sin*X(2) + Y()
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
        n_qubits = max(max(p.qubits.keys()) for p in H.pauli_operators)
        
    batch_count = H.batch_count
        
    if batch_count > 1:
        rho0 = torch.stack([rho0(n_qubits)] * batch_count)
    else:
        rho0 = rho0(n_qubits)

    if H.is_constant():
        h = tlist[1] - tlist[0]

        u = torch.matrix_exp(-1j * h * H(n_qubits=n_qubits))
        ut = u.transpose(-2, -1).conj()

        stepper = lambda _, __, rho: u @ rho @ ut

    else:
        def stepper(t, h, rho):
            knots = t + 0.5*h*(1 + _KNOTS)

            first_term = _first_term(H, knots, h)
            second_term = _second_term(H.funcs, H.pauli_operators, knots, h)
   
            foo = first_term - 0.5*second_term
            # print(foo)
            # input()

            u = torch.matrix_exp(-1j * foo(n_qubits))
            ut = u.transpose(-2, -1).conj()
            
            return u @ rho @ ut

    rho, obsvalues, states = es.solvediffeq(rho0, tlist, stepper, observables, store_intermediate)

    if batch_count > 1:
        return rho, obsvalues, states.permute(1, 0, 2, 3) if store_intermediate else None

    return rho, {k: v[:1, :] for k, v in obsvalues.items()}, states.unsqueeze(0) if store_intermediate else None


def _first_term(H, knots, h):
    result = 0

    for f, p in H.unpack_data():
        result += torch.sum(_WEIGHTS * (f(knots) if callable(f) else f)) * p
 
    return 0.5 * h * result


def _second_term(funcs, ops, knots, h):
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
            
            result += coeff * op

    return (sqrt(15) / 54) * h**2 * result
