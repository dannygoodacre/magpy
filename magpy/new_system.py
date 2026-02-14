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
from ._integrate import integral_from_sample, antisymmetric_double_integral_from_sample
from ._utils import commutator, commutes


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

    h = tlist[1] - tlist[0]

    if isinstance(H, PauliString):
        u = H.propagator(h)
        uH = u.H

        def stepper(t, h, rho):
            return u * rho * uH

    if H.is_static and H.is_commuting:
        u = H.static_commuting_propagator(h)
        uH = u.H

        def stepper(t, h, rho):
            return u * rho * uH

    elif H.is_disjoint():
        def stepper(t, h, rho):
            knots = t + 0.5*h*(1 + _KNOTS)

            propagators = [first_term_pauli(unit_pauli, coeff(knots), h).propagator(h) for unit_pauli, coeff in H._data.items()]

            u = propagators[0]

            for p in propagators[1:]:
                u *= p

            return u * rho * u.H

    else:
        def stepper(t, h, rho):
            omega = two_term_magnus_step(H, t, h)

            u = omega.static_commuting_progagator()

            return u * rho * u.H

    rho, obsvalues, states = es.solvediffeq(torch.ones(batch_count) * rho0,
                                            tlist,
                                            stepper,
                                            observables,
                                            store_intermediate)

    if batch_count > 1:
        return rho, obsvalues, states if store_intermediate else None

    return rho, {k: v[:1, :] for k, v in obsvalues.items()}, states if store_intermediate else None


def first_term_pauli(pauli_unit: PauliString, coeff, h):
    return 0.5 * h * torch.sum(_WEIGHTS * coeff) * pauli_unit


def two_term_magnus_step(H: HamOp, t: float, h: float) -> HamOp:
    pauli_strings = H.pauli_strings
    coeffs = H.coeffs

    n = len(pauli_strings)

    knots = t + 0.5*h*(1 + _KNOTS)

    f_nodes = [
        coeff(knots) if callable(coeff) else coeff * torch.ones_like(knots)
        for coeff in coeffs
    ]

    result = HamOp()

    for i in range(n):
        result += integral_from_sample(f_nodes[i], h) * pauli_strings[i]

    if not H.is_commuting():
        for i in range(n):
            for j in range(i + 1, n):
                if commutes(pauli_strings[i], pauli_strings[j]):
                    continue

                result += -0.5 \
                    * antisymmetric_double_integral_from_sample(f_nodes[i], f_nodes[j], h) \
                    * commutator(pauli_strings[i], pauli_strings[j])

    return result
