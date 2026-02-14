from __future__ import annotations
from numbers import Number
from typing import Callable

import torch
from torch import Tensor

from ..types import SCALAR_TYPES
from .._utils import format_number, tensorize


class FunctionProduct:

    def __init__(self, *args):
        self._functions: dict[Callable, int] = {}
        self._scale: Tensor = torch.tensor(1, dtype=torch.complex128)

        for arg in args:
            if isinstance(arg, FunctionProduct):
                self._scale *= arg._scale

                for f, power in arg._functions.items():
                    self._functions[f] = self._functions.get(f, 0) + power

            elif isinstance(arg, SCALAR_TYPES):
                self._scale *= tensorize(arg)

            elif callable(arg):
                self._functions[arg] = self._functions.get(arg, 0) + 1

    def __call__(self, *args, **kwargs):
        out = self._scale

        for f, power in self._functions.items():
            out = out * f(*args, **kwargs) ** power

        return out

    def __hash__(self):
        return hash((self._scale, frozenset(self._functions.items())))

    def __eq__(self, other):
        if not isinstance(other, FunctionProduct):
            return False

        return self._scale == other._scale and self._functions == other._functions

    def __mul__(self, other):
        if hasattr(other, '_x_mask') or hasattr(other, '_data'):
            return other.__rmul__(self)

        result = FunctionProduct(self)

        if isinstance(other, FunctionProduct):
            result._scale *= other._scale

            for f, p in other._functions.items():
                result._functions[f] = result._functions.get(f, 0) + p

        elif isinstance(other, SCALAR_TYPES):
            scale_val = torch.as_tensor(other, dtype=torch.complex128) \
                if isinstance(other, (list, tuple)) else other

            result._scale *= scale_val

        elif callable(other):
            result._functions[other] = result._functions.get(other, 0) + 1

        else:
            return NotImplemented

        return result

    def __neg__(self):
        result = FunctionProduct()

        result._functions = dict(self._functions)

        result._scale = -self._scale

        return result

    def __str__(self):
        if not self._functions:
            return format_number(self._scale)

        parts = []

        if isinstance(self._scale, Tensor) or self._scale != 1.0:
            parts.append(format_number(self._scale))

        for f, power in self._functions.items():

            name = getattr(f, '__name__', 'f')

            if name == '<lambda>':
                # TODO: Find a better way to handle this. I don't like it.
                name = 'λ'

            parts.append(f'{name}^{power}' if power > 1 else name)

        return '*'.join(parts)

    __rmul__ = __mul__

    @property
    def is_empty(self) -> bool:
        return not self._functions

    @property
    def is_negative(self) -> bool:
        if isinstance(self._scale, int | float):
            return self._scale < 0

        if isinstance(self._scale, complex):
            return self._scale.real < 0 and self._scale.imag == 0

        if torch.is_tensor(self._scale):
            return torch.all(self._scale < 0)

        return False

    @property
    def __name__(self):
        return '*'.join(f.__name__ for f in self._functions)

    @property
    def scale(self) -> Number | Tensor:
        return self._scale
