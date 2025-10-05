from numbers import Number

from torch import Tensor

from ._context import get_print_precision


def conjugate(x: Number | Tensor):
    try:
        return x.conjugate()

    except AttributeError:
        return x.conj()


def format_number(n: Number) -> str:
    real, imag = n.real, n.imag

    if real == 0 and imag == 0:
        return '0'
    
    if real == 0:
        return f'{imag:.{get_print_precision()}g}j'

    if imag == 0:
        return f'{real:.{get_print_precision()}g}'
    
    return f'{n:.{get_print_precision()}g}'


def format_tensor(tensor: Tensor) -> str:
    if tensor.numel() == 1:
        return format_number(tensor.item())

    return '(' + ', '.join(format_number(n) for n in tensor.tolist()) + ')'
