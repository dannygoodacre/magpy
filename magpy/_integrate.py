from math import sqrt

import torch
from torch import Tensor

from ._glq import _KNOTS_3, _WEIGHTS_3

def sample(f, t: float, h: float) -> Tensor:
    # Sample f across the def_ined interval at GLQ3 knots.

    knots = t + 0.5*h*(1 + _KNOTS_3)

    return f(knots) if callable(f) else f * torch.ones_like(knots)


def integral_from_sample(f_nodes: Tensor, h: float) -> Tensor:
    # Calculate the integral of the given function sampled over the given step h using GLQ3.

    return 0.5 * h * torch.sum(_WEIGHTS_3 * f_nodes, dim=-1)


def antisymmetric_double_integral_from_sample(f_i: float, f_j: float, h: float) -> Tensor:
    # f_i and f_j are samples of the function at knots [t0, t1, t2]
    # \int_t^{t+h} d\tau_1 \int_t^{\tau_1} d\tau_2 \( f_i(\tau_1)f_j(\tau_2) - f_j(\tau_1)f_i(\tau_2) (\)

    quad_sum = 2*(f_i[0]*f_j[1] - f_i[1]*f_j[0]) \
               + (f_i[0]*f_j[2] - f_i[2]*f_j[0]) \
               + 2*(f_i[1]*f_j[2] - f_i[2]*f_j[1])

    return (sqrt(15) / 108) * h**2 * quad_sum
