from numbers import Number
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from ._context import get_print_precision

if TYPE_CHECKING:
    from .types import Scalar


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
        return coeff.to(torch.complex128)

    if isinstance(coeff, Number):
        return torch.tensor(coeff, dtype=torch.complex128)

    return torch.as_tensor(coeff, dtype=torch.complex128)
