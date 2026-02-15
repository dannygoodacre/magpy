"""High-order splitting methods for Hamiltonian evolution.

This module implements fourth-order Suzuki-Trotter decomposition for the
exponentiation of the Magnus expansion.

References
----------

.. [1] Hatano, N. & Suzuki, M. (2005), "Finding Exponential Product Formulas
       of Higher Orders", *Lecture Notes in Physics* 679, 37-68.

"""


from typing import TYPE_CHECKING



if TYPE_CHECKING:
    from typing import Callable

    from torch import Tensor

    from . import Operator

def s2_step(rho: Operator, terms: list[tuple[Operator, Callable | Tensor]] , scale: float | Tensor):
    """Perform a second-order symmetric Strang split step (S2).

    Parameters
    ----------
    rho : Operator
        Current state
    terms : list[tuple[Operator, Callable | Tensor]]
        Constituent operators and their coefficients
    scale : float | Tensor
        Scaling factor for the step

    Returns
    -------
    Operator
        The evolved state
    """

    for unit_pauli, coeff in terms:
        u = unit_pauli.propagator(coeff * scale * 0.5)

        rho = u * rho * u.H

    for unit_pauli, coeff in reversed(terms):
        u = unit_pauli.propagator(coeff * scale * 0.5)

        rho = u * rho * u.H

    return rho


def s4_step(rho: Operator, terms: list[tuple[Operator, Callable | Tensor]] , scale: float | Tensor):
    """Perform a fourth-order Suzuki-Trotter step (S4).

    Parameters
    ----------
    rho : Operator
        Current state
    terms : list[tuple[Operator, Callable | Tensor]]
        Constituent operators and their coefficients
    scale : float | Tensor
        Scaling factor for the step

    Returns
    -------
    Operator
        The evolved state
    """

    p = 1.0 / (4.0 - 4.0**(1/3))

    rho = s2_step(rho, terms, p * scale)
    rho = s2_step(rho, terms, p * scale)
    rho = s2_step(rho, terms, (1 - 4*p) * scale)
    rho = s2_step(rho, terms, p * scale)
    rho = s2_step(rho, terms, p * scale)

    return rho
