from functools import reduce
from numbers import Number
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .function_product import FunctionProduct
from .linalg import kron
from .._utils import format_number

if TYPE_CHECKING:
    from .hamiltonian_operator import HamiltonianOperator


def X(index: int = 0) -> PauliString:
    return PauliString(1 << index, 0, 1.0, index + 1)

def Y(index: int = 0) -> PauliString:
    return PauliString(1 << index, 1 << index, 1.0, index + 1)

def Z(index: int = 0) -> PauliString:
    return PauliString(0, 1 << index, 1.0, index + 1)

def I(index: int = 0) -> PauliString:
    return PauliString(0, 0, 1.0, index + 1)


class PauliString:

    _X_MATRIX = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)

    _Y_MATRIX = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)

    _Z_MATRIX = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)

    _I_MATRIX = torch.eye(2, dtype=torch.complex128)

    def __init__(self, x_mask: int, z_mask: int, coeff: complex | torch.Tensor = 1.0, n_qubits: int = None):
        self._x_mask = x_mask
        self._z_mask = z_mask
        self._coeff = coeff

        if n_qubits is None:
            self._n_qubits = max(x_mask.bit_length(), z_mask.bit_length(), 1)

        else:
            self._n_qubits = n_qubits

    def __add__(self, other) -> HamiltonianOperator:
        if isinstance(other, PauliString):
            from .hamiltonian_operator import HamiltonianOperator

            return HamiltonianOperator(self, other)

    def __call__(self, *args, **kwargs) -> Tensor:
        a = self._coeff(*args, **kwargs) if callable(self._coeff) else self._coeff

        return a * self.as_unit_operator()

    def __hash__(self):
        return hash((self._x_mask, self._z_mask, self._n_qubits))

    def __eq__(self, other):
        if not isinstance(other, PauliString):
            return False

        return self._x_mask == other._x_mask \
            and self._z_mask == other._z_mask \
            and self._n_qubits == other._n_qubits

    def __mul__(self, other):
        if isinstance(other, Number | torch.Tensor):
            coeff = self._coeff * other

            return PauliString(self._x_mask, self._z_mask, coeff)

        if callable(other):
            coeff = other if self._coeff == 1 else FunctionProduct(self._coeff, other)

            return PauliString(self._x_mask, self._z_mask, coeff)

        if isinstance(other, PauliString):
            x_mask = self._x_mask ^ other._x_mask
            z_mask = self._z_mask ^ other._z_mask

            if callable(self._coeff):
                coeff = FunctionProduct(self._coeff, other._coeff)

            else:
                coeff = self._coeff * other._coeff

            return PauliString(x_mask, z_mask, coeff * PauliString.__phase_factor(self, other))

        from .hamiltonian_operator import HamiltonianOperator

        if isinstance(other, HamiltonianOperator):
            return other.__rmul__(self)

        return NotImplemented

    def __neg__(self):
        return self.__mul__(-1)

    def __rmul__(self, other):
        if isinstance(other, Number | torch.Tensor) or callable(other):
            return self.__mul__(other)

        from .hamiltonian_operator import HamiltonianOperator

        if isinstance(other, HamiltonianOperator):
            return other.__mul__(self)

        return self.__mul__(other)

    def __str__(self):
        from .._context import get_print_identities

        label = []

        for i in range(self._n_qubits):
            x = (self._x_mask >> i) & 1
            z = (self._z_mask >> i) & 1

            if x and z:
                label.append(f'Y{i}')

            elif x:
                label.append(f'X{i}')

            elif z:
                label.append(f'Z{i}')

            elif get_print_identities():
                label.append(f'I{i}')

        if not label:
            label.append('I')

        operator_string = '*'.join(label)

        is_one = False

        if torch.is_tensor(self._coeff):
            is_one = torch.all(self._coeff == 1)

        else:
            is_one = self._coeff == 1

        if is_one:
            return operator_string

        if isinstance(self._coeff, FunctionProduct):
            return f'{format_number(self._coeff._scale)}*{self._coeff.__name__}*{operator_string}'

        if callable(self._coeff):
            return f'{self._coeff.__name__}*{operator_string}'

        return f'{format_number(self._coeff)}*{operator_string}'

    def __sub__(self, other):
        from .hamiltonian_operator import HamiltonianOperator

        return HamiltonianOperator(self, -other)

    def as_unit_operator(self) -> PauliString:
        return PauliString(self._x_mask, self._z_mask, n_qubits = self._n_qubits)

    def matrix(self, n_qubits: int = None) -> torch.Tensor:
        if n_qubits is None:
            n_qubits = self.n_qubits

        ops = []

        for i in range(n_qubits):
            ops.append(self.__single_qubit_matrix(i))

        return self._coeff * reduce(torch.kron, ops)

    def propagator(self, h: Tensor = torch.tensor(1, dtype=torch.complex128), t: Tensor = None) -> PauliString | HamiltonianOperator:
        # expm( -i * h * H(t))

        if self.is_constant:
            v = torch.as_tensor(self._coeff * h)

            return torch.cos(v)*I() - 1j*torch.sin(v)*self.as_unit_operator()

        s = FunctionProduct() * h * (lambda *args: torch.sin(self._coeff(*args)))
        c = FunctionProduct() * h * (lambda *args: torch.cos(self._coeff(*args)))

        from .hamiltonian_operator import HamiltonianOperator

        return HamiltonianOperator(
            (c, I()),
            (-1j * s, self.as_unit_operator())
        )

    @property
    def coeff(self) -> int | torch.Tensor:
        return self._coeff

    @property
    def is_constant(self) -> bool:
        return not callable(self._coeff)

    @property
    def n_qubits(self) -> int:
        return max(self._x_mask.bit_length(), self._z_mask.bit_length(), 1)

    def __single_qubit_matrix(self, i: int) -> torch.Tensor:
        x = (self._x_mask >> i) & 1
        z = (self._z_mask >> i) & 1

        if x and z:
            return self._Y_MATRIX

        elif x:
            return self._X_MATRIX

        elif z:
            return self._Z_MATRIX

        return self._I_MATRIX

    @staticmethod
    def from_label(label: str, coeff: complex | torch.Tensor = 1.0):
        x_mask = 0
        z_mask = 0

        for i, char in enumerate(reversed(label.upper())):
            if char == 'X':
                x_mask |= (1 << i)

            elif char == 'Y':
                x_mask |= (1 << i)
                z_mask |= (1 << i)

            elif char == 'Z':
                z_mask |= (1 << i)

            elif char != 'I':
                raise ValueError(f"Invalid Pauli character: {char}")

        return PauliString(x_mask, z_mask, coeff, len(label))

    @staticmethod
    def __phase_factor(a: PauliString, b: PauliString):
        x1, z1 = a._x_mask, a._z_mask
        x2, z2 = b._x_mask, b._z_mask

        num_qubits = max(x1.bit_length(), z1.bit_length(), x2.bit_length(), z2.bit_length())

        p = 0

        for i in range(num_qubits):
            b1_x = (x1 >> i) & 1
            b1_z = (z1 >> i) & 1
            b2_x = (x2 >> i) & 1
            b2_z = (z2 >> i) & 1

            if b1_x == b1_z:
                if b1_x == 1:
                    if b2_x == 1 and b2_z == 0:
                        p -= 1

                    elif b2_x == 0 and b2_z == 1:
                        p += 1

            elif b1_x == 1:
                if b2_x == 1 and b2_z == 1:
                    p += 1

                elif b2_x == 0 and b2_z == 1:
                    p -= 1

            elif b1_z == 1:
                if b2_x == 1 and b2_z == 0:
                    p += 1

                elif b2_x == 1 and b2_z == 1:
                    p -= 1

        return 1j ** (p % 4)
