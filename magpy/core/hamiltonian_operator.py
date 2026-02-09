from numbers import Number
from typing import Callable

import torch
from torch import Tensor

from .pauli_string import PauliString
from .._utils import format_number


class HamiltonianOperator:
    def __init__(self, *args):
        self._data = {}

        for arg in args:
            if isinstance(arg, tuple):
                scale, pauli_string = arg

                self.__add_term(pauli_string, scale)

            elif isinstance(arg, PauliString):
                self.__add_term(arg)

    def __add__(self, other: PauliString | HamiltonianOperator):
        return HamiltonianOperator(self, other)

    def __call__(self, *args, **kwargs):
        result = HamiltonianOperator()

        for operator, coeff in self._data.items():
            result.__add_term(operator, coeff(*args, **kwargs) if callable(coeff) else coeff)

        return result

    def __mul__(self, other):
        if isinstance(other, Number | torch.Tensor):
            return self.__scalar_mul(other)

        if isinstance(other, PauliString):
            result = HamiltonianOperator()

            for pauli_string, coeff in self._data.items():
                result.__add_term(pauli_string * other, coeff)

            return result

    def __neg__(self):
        return -1 * HamiltonianOperator(self)

    def __rmul__(self, other):
        return self.__scalar_mul(other)

    def __str__(self):
        if not self._data:
            return '0'

        items = list(self._data.items())

        label_parts = []

        for i, (term, coeff) in enumerate(items):
            c_str = format_number(coeff) if isinstance(coeff, Number) else str(coeff)

            p_str = str(term)

            is_negative = False

            if isinstance(coeff, (int, float)):
                is_negative = coeff < 0

            elif torch.is_tensor(coeff) and coeff.ndim == 0:
                is_negative = coeff.item().real < 0 if coeff.is_complex() else coeff.item() < 0

            elif isinstance(coeff, complex):
                is_negative = coeff.real < 0 and coeff.imag == 0

            if i == 0:
                label_parts.append(f'{c_str}*{p_str}')

            else:
                if is_negative:
                    label_parts.append(f' - {c_str.lstrip('-')}*{p_str}')

                else:
                    label_parts.append(f' + {c_str}*{p_str}')

        return ''.join(label_parts)

    def __sub__(self, other):
        return HamiltonianOperator(self, -other)

    def __add_term(self, pauli_string: PauliString, scale: Number | torch.Tensor | Callable = 1):
        operator = pauli_string.as_unit_operator()

        coeff = scale * pauli_string.coeff

        if operator in self._data:
            self._data[operator] += coeff

        else:
            self._data[operator] = coeff

        if self._data[operator] == 0:
            del self._data[operator]

    def __scalar_mul(self, other):
        result = HamiltonianOperator()

        result._data = {pauli_string: coeff * other for pauli_string, coeff in self._data.items()}

        return result
