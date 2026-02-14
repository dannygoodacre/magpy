from numbers import Number

import torch
from torch import Tensor

from .core.pauli_string import PauliString
from .types import SCALAR_TYPES

from ._context import get_print_precision


def conjugate(x: Number | Tensor):
    try:
        return x.conjugate()

    except AttributeError:
        return x.conj()


def commutator(a: PauliString, b: PauliString) -> bool:
    return a*b - b*a


def commutes(a: PauliString, b: PauliString) -> bool:
    # TODO: Document
    # symplectic inner product: (Z1 & X2) ^ (X1 & Z2)

    check = (a._z_mask & b._x_mask) ^ (a._x_mask & b._z_mask)

    return check.bit_count() % 2 == 0


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


def tensorize(coeff: SCALAR_TYPES) -> Tensor:
    if isinstance(coeff, Tensor):
        return coeff.to(torch.complex128)

    if isinstance(coeff, Number):
        return torch.tensor(coeff, dtype=torch.complex128)

    return torch.as_tensor(coeff, dtype=torch.complex128)
