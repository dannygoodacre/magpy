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
from torch import Tensor
import expsolve as es

from .core import PauliString, I, HamOp
from ._context import get_device


_KNOTS = torch.tensor([-sqrt(3/5), 0, sqrt(3/5)], dtype=torch.complex128)
_WEIGHTS = torch.tensor([5/9, 8/9, 5/9])


def _update_device():
    global _KNOTS, _WEIGHTS

    _KNOTS = _KNOTS.to(get_device())
    _WEIGHTS = _WEIGHTS.to(get_device())


def new_evolve(H: PauliString | HamOp,
           rho0: PauliString,
           tlist: Tensor,
           n_qubits: int = None,
           observables: dict[str, Callable[[torch.Tensor, Number], torch.Tensor]] = {},
           store_intermediate: bool = False) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor | None]:

    batch_count = H.batch_count

    if isinstance(H, PauliString):
        stepper = _pauli_string_stepper(H, tlist[1] - tlist[0])

    if not H.is_constant():
        if H.is_completely_independent():
            def stepper(t, h, rho):
                knots = t + 0.5*h*(1 + _KNOTS)

                props = [first_term_pauli(unit_pauli, coeff(knots), h).propagator(h) for unit_pauli, coeff in H._data.items()]

                propagator = props[0]

                for prop in props[1:]:
                    propagator *= prop

                pt = propagator.H

                return propagator * rho * pt

    rho, obsvalues, states = es.solvediffeq(torch.ones(batch_count) * rho0,
                                            tlist,
                                            stepper,
                                            observables,
                                            store_intermediate)

    if batch_count > 1:
        return rho, obsvalues, states if store_intermediate else None

    return rho, {k: v[:1, :] for k, v in obsvalues.items()}, states if store_intermediate else None


def _pauli_string_stepper(P: PauliString, h: Tensor):
    u = P.propagator(h)

    ut = u.H

    return (lambda t, h, rho: u * rho * ut)


def first_term_pauli(pauli_unit: PauliString, coeff, h):
    return 0.5 * h * torch.sum(_WEIGHTS * coeff) * pauli_unit


def first_term(H: HamOp, knots, h):
    result = None

    for pauli_unit, coeff in H._data.items():
        if result is None:
            result = __intermediate_first_term(pauli_unit, coeff, knots)

        else:
            result += __intermediate_first_term(pauli_unit, coeff, knots)

    return 0.5 * h * result


def second_term(coeffs, unit_paulis, knots, h):
    result = None

    n = len(coeffs)

    for i in range(n):
        for j in range(i + 1, n):
            if unit_paulis[i] == I() or unit_paulis[j] == I():
                continue



def __intermediate_first_term(pauli_unit: PauliString, coeff, knots):
    return torch.sum(_WEIGHTS * (coeff(knots) if callable(coeff) else coeff)) * pauli_unit
