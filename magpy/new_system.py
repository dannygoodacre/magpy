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
        h = tlist[1] - tlist[0]

        u = H.propagator(h)

        ut = u.H

        def stepper(t, h, rho):
            return u * rho * ut

    rho, obsvalues, states = es.solvediffeq(torch.ones(batch_count) * rho0,
                                            tlist,
                                            stepper,
                                            observables,
                                            store_intermediate)

    if batch_count > 1:
        return rho, obsvalues, states if store_intermediate else None

    return rho, {k: v[:1, :] for k, v in obsvalues.items()}, states if store_intermediate else None
