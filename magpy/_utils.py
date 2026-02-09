from numbers import Number

import torch
from torch import Tensor

from ._context import get_print_precision
from ._types import Scalar

def conjugate(x: Number | Tensor):
    try:
        return x.conjugate()

    except AttributeError:
        return x.conj()


def format_number(n: Number | Tensor) -> str:
    if isinstance(n, Number):
        real, imag = n.real, n.imag

        if n == 1:
            return '1'

        if real == 0 and imag == 0:
            return '0'

        if real == 0:
            return f'{imag:.{get_print_precision()}g}j'

        if imag == 0:
            return f'{real:.{get_print_precision()}g}'

        return f'({n:.{get_print_precision()}g})'

    if isinstance(n, Tensor):
        if n.numel() == 1:
            return format_number(n.item())

        return '(' + ', '.join(format_number(x) for x in n.tolist()) + ')'


def tensorize(coeff: Scalar) -> Tensor:
    if isinstance(coeff, Tensor):
        return coeff

    if isinstance(coeff, Number):
        return torch.tensor(coeff)

    return torch.as_tensor(coeff)
