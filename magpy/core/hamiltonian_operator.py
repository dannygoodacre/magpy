from numbers import Number
from typing import Callable

import torch
from torch import Tensor

from .pauli_string import PauliString
from .function_product import FunctionProduct

from .._types import Scalar
from .._utils import commutes, format_number, tensorize


class HamOp:
    def __init__(self, *args):
        self._data = {}

        for arg in args:
            if isinstance(arg, tuple):
                scale, pauli_string = arg

                self.__add_term(pauli_string, scale)

            elif isinstance(arg, PauliString):
                self.__add_term(arg)

    def __add__(self, other: PauliString | HamOp):
        return HamOp(self, other)

    def __call__(self, *args, **kwargs):
        result = HamOp()

        for operator, coeff in self._data.items():
            result.__add_term(operator, coeff(*args, **kwargs) if callable(coeff) else coeff)

        return result

    def __copy__(self):
        return self.copy()

    def __mul__(self, other):
        if isinstance(other, Scalar):
            return self.__scalar_mul(other)

        if isinstance(other, PauliString):
            other = HamOp(other)

        if isinstance(other, HamOp):
            result = HamOp()

            for p1, c1 in self._data.items():
                for p2, c2 in other._data.items():
                    result.__add_term(p1 * p2, c1 * c2)

            return result

        return NotImplemented

    def __neg__(self):
        return -1 * HamOp(self)

    def __pow__(self, n: int):
        return self.power(n)

    def __rmul__(self, other):
        if isinstance(other, Scalar):
            return self.__scalar_mul(other)

        if isinstance(other, PauliString):
            return HamOp(other) * self

        if isinstance(other, HamOp):
            return other * self

    def __str__(self):
        if not self._data:
            return '0'

        items = list(self._data.items())
        label_parts = []

        for i, (term, coeff) in enumerate(items):
            p_str = str(term)
            is_negative = False
            is_unity = False

            if callable(coeff):
                c_str = str(coeff)

            else:
                c_str = format_number(coeff)

                if not torch.is_tensor(coeff) or coeff.ndim == 0:
                    val = coeff.item() if torch.is_tensor(coeff) else coeff
                    if val == 1:
                        is_unity = True

                    elif val == -1:
                        is_unity = True
                        is_negative = True

                    elif val == 0:
                        continue

                if isinstance(coeff, (int, float)):
                    is_negative = coeff < 0

                elif isinstance(coeff, complex):
                    is_negative = coeff.real < 0 and coeff.imag == 0

                elif torch.is_tensor(coeff) and coeff.ndim == 0:
                    val = coeff.item()
                    is_negative = val.real < 0 if coeff.is_complex() else val < 0

            if is_unity:
                display_c = ""

            else:
                display_c = f"{c_str.lstrip('-') if is_negative and i > 0 else c_str}*"

            if i == 0:
                prefix = "-" if is_negative and is_unity else ""
                label_parts.append(f"{prefix}{display_c}{p_str}")

            else:
                sign = " - " if is_negative else " + "
                label_parts.append(f"{sign}{display_c}{p_str}")

        return ''.join(label_parts)

    def __sub__(self, other):
        return HamOp(self, -other)

    def copy(self) -> HamOp:
        result = HamOp()
        result._data = self._data.copy()
        result._n_qubits = self._n_qubits

        return

    def power(self, n: int) -> PauliString | HamOp:
        if n == 0:
            from .pauli_string import I

            return I()

        if n == 1:
            return self

        result = None
        base = self

        while n > 0:
            if n % 2 == 1:
                result = base if result is None else result * base
            base = base * base
            n //= 2

        return result

    def tensor(self, n_qubits: int = None) -> Tensor:
        if n_qubits is None:
            n_qubits = self.n_qubits

        if not self._data:
            return torch.zeros((1,1), dtype=torch.complex128)

        result = None

        for pauli_unit, coeff in self._data.items():
            matrix = pauli_unit.tensor(n_qubits)

            c = torch.as_tensor(coeff)

            if c.ndim > 0:
                c = c.view(*c.shape, 1, 1)

            if result is None:
                result = c * matrix

                continue

            result = result + c * matrix

        return result

    @property
    def is_commuting(self) -> bool:
        operators = list(self._data.keys())

        for i in range(len(operators)):
            for j in range(i + 1, len(operators)):
                if not commutes(operators[i], operators[j]):
                    return False

        return True

    @property
    def is_constant(self) -> bool:
        return not any(callable(coeff) for coeff in self._data.values())

    @property
    def n_qubits(self) -> int:
        return max(p.n_qubits for p in self._data.keys())

    def __add_term(self, pauli_string: PauliString, scale: Number | Tensor | Callable = 1):
        operator = pauli_string.as_unit_operator()

        if callable(scale):
            coeff = FunctionProduct(scale, pauli_string.coeff)
        else:
            coeff = scale * pauli_string.coeff

        if operator in self._data:
            self._data[operator] += coeff

        else:
            self._data[operator] = coeff

        current_val = self._data[operator]

        if callable(current_val):
            return

        if torch.is_tensor(current_val):
            if current_val.ndim == 0 and current_val == 0:
                del self._data[operator]

        elif current_val == 0:
            del self._data[operator]

    def __scalar_mul(self, other):
        other = tensorize(other)

        result = HamOp()

        result._data = {
            pauli_string: coeff * other
            for pauli_string, coeff in self._data.items()
        }

        return result
