from numbers import Number

from torch import Tensor

from ._context import get_print_precision


def format_number(number: Number) -> str:
    return f'{number:.{get_print_precision()}g}'

def format_tensor(tensor: Tensor) -> str:
    if tensor.numel() == 1:
        return format_number(tensor.item())

    return '(' + ', '.join(format_number(n) for n in tensor.tolist()) + ')'
