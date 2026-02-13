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

    if isinstance(H, PauliString):
        stepper = _pauli_string_stepper(H, tlist[1] - tlist[0])

    if not H.is_constant():
        if H.is_completely_independent():
            def stepper(t, h, rho):
                knots = t + 0.5*h*(1 + _KNOTS)

                propagators = [first_term_pauli(unit_pauli, coeff(knots), h).propagator(h) for unit_pauli, coeff in H._data.items()]

                propagator = propagators[0]

                for prop in propagators[1:]:
                    propagator *= prop

                return propagator * rho * propagator.H

        else:
            def stepper(t, h, rho):
                omega = new_magnus_step(H, t, h)

                u = omega.propagator()

                return u * rho * u.H

                # knots = t + 0.5*h*(1 + _KNOTS)

                # first = first_term(H, knots, h)

                # second = second_term(list(H._data.values()), list(H._data.keys()), knots, h)

                # u = (first - 0.5*second).propagator()

                # return u * rho * u.H

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

            if callable(coeffs[i]):
                fi0 = coeffs[i](knots[0])
                fi1 = coeffs[i](knots[1])
                fi2 = coeffs[i](knots[2])

            else:
                fi0 = coeffs[i]
                fi1 = coeffs[i]
                fi2 = coeffs[i]

            if callable(coeffs[j]):
                fj0 = coeffs[j](knots[0])
                fj1 = coeffs[j](knots[1])
                fj2 = coeffs[j](knots[2])

            else:
                fj0 = coeffs[j]
                fj1 = coeffs[j]
                fj2 = coeffs[j]


            coeff = 2*(fi0*fj1 - fi1*fj0) + (fi0*fj2 - fi2*fj0) + 2*(fi1*fj2 - fi2*fj1)

            if result is None:
                result = coeff * (unit_paulis[i]*unit_paulis[j] - unit_paulis[j]*unit_paulis[i])

                continue

            result += coeff * (unit_paulis[i]*unit_paulis[j] - unit_paulis[j]*unit_paulis[i])

    return (sqrt(15) / 54) * h**2 * result


def __intermediate_first_term(pauli_unit: PauliString, coeff, knots):
    return torch.sum(_WEIGHTS * (coeff(knots) if callable(coeff) else coeff)) * pauli_unit


def glq3_integral(f_nodes: Tensor, h: float) -> Tensor:
    """
    Calculates the integral of a function over a step h using GLQ3.
    """

    return 0.5 * h * torch.sum(_WEIGHTS * f_nodes, dim=-1)


def new_first_term(H: HamOp, knots: Tensor, h: float) -> HamOp:
    result = HamOp()

    for pauli_unit, coeff in H._data.items():
        f_nodes = coeff(knots) if callable(coeff) else coeff * torch.ones_like(knots)

        result += glq3_integral(f_nodes, h) * pauli_unit

    return result


def antisymmetric_double_integral(fi, fj, h):
    """
    G_ij using GLQ3 knots.

    fi and fj are tensors of the function values at knots [t0, t1, t2].

    \int_t^{t+h} d\tau_1 \int_t^{\tau_1} d\tau_2 \( f_i(\tau_1)f_j(\tau_2) - f_j(\tau_1)f_i(\tau_2) (\)
    """

    quad_sum = 2*(fi[0]*fj[1] - fi[1]*fj[0]) \
               + (fi[0]*fj[2] - fi[2]*fj[0]) \
               + 2*(fi[1]*fj[2] - fi[2]*fj[1])

    return (sqrt(15) / 108) * h**2 * quad_sum


def new_second_term(H: HamOp, knots: Tensor, h: float):
    operators = list(H._data.keys())
    coeffs = list(H._data.values())

    n = len(operators)

    vals = []

    for coeff in coeffs:
        vals.append(coeff(knots) if callable(coeff) else coeff * torch.ones_like(knots))

    result = HamOp()

    for i in range(n):
        for j in range(i + 1, n):
            if commutes(operators[i], operators[j]):
                continue

            g_ij = antisymmetric_double_integral(vals[i], vals[j], h)

            result += g_ij * commutator(operators[i], operators[j])

    return result

def new_magnus_step(H: HamOp, t: float, h: float) -> HamOp:
    knots = t + 0.5*h*(1 + _KNOTS)

    operators = list(H._data.keys())
    coeffs = list(H._data.values())

    n = len(operators)

    f_nodes = [
        coeff(knots) if callable(coeff) else coeff * torch.ones_like(knots)
        for coeff in coeffs
    ]

    result = HamOp()

    for i in range(n):
        result += glq3_integral(f_nodes[i], h) * operators[i]

    if not H.is_commuting():
        for i in range(n):
            for j in range(i + 1, n):
                if commutes(operators[i], operators[j]):
                    continue

                g_ij = antisymmetric_double_integral(f_nodes[i], f_nodes[j], h)

                result += -0.5 * g_ij * commutator(operators[i], operators[j])

    return result
