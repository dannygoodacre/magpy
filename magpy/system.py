"""High-order geometric integration for Hamiltonian evolution.

This module implements the Magnus expansion to solve the Liouville-von Neumann
equation, using the quadrature formulae described by Iserles et al.

References
----------

.. [1] Magnus, W. (1954), "On the exponential solution of differential
       equations for a linear operator", *Comm. Pure Appl. Math.* 7, 649-673.

.. [2] Iserles, A., Munthe-Kaas, H. Z., Nørsett, S. P. & Zanna, A. (2000),
       "Lie-group methods", *Acta Numerica* 9, 215-365.

"""


from typing import TYPE_CHECKING, Callable

import expsolve as es
import torch

from ._context import get_device
from .core import PauliString, HamOp
from ._glq import _KNOTS_3, _WEIGHTS_3
from ._integrate import integral_from_sample, antisymmetric_double_integral_from_sample
from .linalg import commutes
from ._splitting import s4_step

if TYPE_CHECKING:
    from . import Operator
    from torch import Tensor
    from .core import PauliString


def _update_device():
    global _KNOTS, _WEIGHTS

    _KNOTS = _KNOTS.to(get_device())
    _WEIGHTS = _WEIGHTS.to(get_device())


def evolve(H: Operator,
           rho0: PauliString,
           tlist: Tensor,
           observables: dict[str, Callable[[Tensor, Tensor], Tensor]] = {},
           store_intermediate: bool = False) -> tuple[Operator, dict[str, Tensor], Operator | None]:
    """Liouville-von Neumann evolution of a density operator under a given Hamiltonian.

    Examples
    --------
    >>> H = ...
    >>> rho0 = ...
    >>> tlist = mp.timegrid(0, 10, 0.5**6)
    >>> observables = { ... }
    >>> rhoT, obs_values, _ = mp.evolve(H, rho0, tlist, observables, store_intermediate = False)
    >>> TODO

    Parameters
    ----------
    H : Operator
        Hamiltonian operator
    rho0 : PauliString
        Initial density operator
    tlist : Tensor
        Discretized points in time
    observables : dict[str, Callable[[Tensor, Tensor], Tensor]], optional
        A mapping of labels to functions computing observable quantities,
        by default {}
    store_intermediate : bool, optional
        Whether to store the intermediate states of the evolving system, by default False

    Returns
    -------
    tuple[Operator, dict[str, Operator], Operator | None]
        The final state, the computed observables, and the optional intermediate states
    """

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

    elif H.is_disjoint:
        def stepper(t, h, rho):
            knots = t + 0.5*h*(1 + _KNOTS_3)

            propagators = [
                first_term_pauli(unit_pauli, coeff(knots), h).propagator(h)
                for unit_pauli, coeff in H._data.items()
            ]

            u = propagators[0]

            for p in propagators[1:]:
                u *= p

            return u * rho * u.H

    else:
        def stepper(t, h, rho):
            omega = two_term_magnus_step(H, t, h)

            return s4_step(rho, list(omega.terms), 1)

    rho, obsvalues, states = es.solvediffeq(torch.ones(batch_count) * rho0,
                                            tlist,
                                            stepper,
                                            observables,
                                            store_intermediate)

    if batch_count > 1:
        return rho, obsvalues, states if store_intermediate else None

    return rho, {k: v[:1, :] for k, v in obsvalues.items()}, states if store_intermediate else None


def first_term_pauli(pauli_unit: PauliString, coeff, h):
    return 0.5 * h * torch.sum(_WEIGHTS_3 * coeff) * pauli_unit


def two_term_magnus_step(H: HamOp, t: float, h: float) -> HamOp:
    pauli_strings = H.pauli_strings
    coeffs = H.coeffs

    n = len(pauli_strings)

    knots = t + 0.5*h*(1 + _KNOTS_3)

    f_nodes = [
        coeff(knots) if callable(coeff) else coeff * torch.ones_like(knots)
        for coeff in coeffs
    ]

    result = HamOp()

    for i in range(n):
        result += integral_from_sample(f_nodes[i], h) * pauli_strings[i]

    if not H.is_commuting:
        for i in range(n):
            for j in range(i + 1, n):
                if commutes(pauli_strings[i], pauli_strings[j]):
                    continue

                result += -0.5 \
                    * antisymmetric_double_integral_from_sample(f_nodes[i], f_nodes[j], h) \
                    * (pauli_strings[i]*pauli_strings[j] - pauli_strings[j]*pauli_strings[i])

    return result
