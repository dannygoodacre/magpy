from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .linalg import kron

from .._types import Scalar

if TYPE_CHECKING:
    from .hamiltonian_operator import HamOp


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

    def __init__(self, x_mask: int, z_mask: int, coeff: complex | Tensor = 1.0, n_qubits: int = None):
        from .._utils import tensorize

        self._x_mask = x_mask
        self._z_mask = z_mask

        self._coeff = tensorize(coeff)

        if n_qubits is None:
            self._n_qubits = max(x_mask.bit_length(), z_mask.bit_length(), 1)

        else:
            self._n_qubits = n_qubits

    def __add__(self, other: PauliString | HamOp) -> HamOp:
        from .hamiltonian_operator import HamOp

        return HamOp(self, other)

    def __copy__(self):
        return self.copy()

    def __hash__(self):
        return hash((self._x_mask, self._z_mask, self._n_qubits))

    def __eq__(self, other):
        if not isinstance(other, PauliString):
            return False

        return self._x_mask == other._x_mask \
            and self._z_mask == other._z_mask \
            and self._n_qubits == other._n_qubits

    def __mul__(self, other: Scalar | PauliString | HamOp) -> PauliString | HamOp:
        from .._utils import tensorize

        if isinstance(other, Scalar):
            return PauliString(
                self._x_mask,
                self._z_mask,
                self._coeff * tensorize(other)
            )

        if isinstance(other, PauliString):
            return PauliString(
                self._x_mask ^ other._x_mask,
                self._z_mask ^ other._z_mask,
                self._coeff * other._coeff * PauliString.__phase_factor(self, other)
            )

        from .hamiltonian_operator import HamOp

        if isinstance(other, HamOp):
            return other.__rmul__(self)

        if callable(other):
            return HamOp((other, self))

        return NotImplemented

    def __neg__(self) -> PauliString:
        return PauliString(
            self._x_mask,
            self._z_mask,
            -self._coeff
        )

    def __pow__(self, n: int) -> PauliString:
        return self.power(n)

    def __rmul__(self, other: Scalar | PauliString | HamOp) -> PauliString | HamOp:
        if isinstance(other, Scalar | PauliString):
            return self * other

        from .hamiltonian_operator import HamOp

        if isinstance(other, HamOp):
            return other * self

        if callable(other):
            return HamOp((other, self))

    def __str__(self):
        from .._context import get_print_identities
        from .._utils import format_number

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

        operator_string = '*'.join(label) if label else 'I'

        if isinstance(self._coeff, torch.Tensor):
            if torch.all(self._coeff == 1):
                return operator_string

            return f"{format_number(self._coeff)}*{operator_string}"

        if self._coeff == 1:
            return operator_string

        return f"{format_number(self._coeff)}*{operator_string}"

    def __sub__(self, other: PauliString | HamOp) -> HamOp:
        from .hamiltonian_operator import HamOp

        return HamOp(self, -other)

    def as_unit_operator(self) -> PauliString:
        return PauliString(self._x_mask, self._z_mask, n_qubits = self._n_qubits)

    def copy(self) -> PauliString:
        return PauliString(self._x_mask, self._z_mask, self._coeff, self._n_qubits)

    def power(self, n: int) -> PauliString:
        if n == 0:
            return I()

        if n == 1:
            return self

        coeff = self._coeff ** n

        if n % 2 == 0:
            return coeff * I()

        return self.copy()

    def tensor(self, n_qubits: int = None) -> Tensor:
        if n_qubits is None:
            n_qubits = self.n_qubits

        matrix = kron(*[self.__single_qubit_matrix(i) for i in range(n_qubits)])

        scale = self._coeff

        if scale.ndim > 0:
            scale = scale.view(*scale.shape, 1, 1)

        return scale * matrix

    def propagator(self, h: float = 1.0) -> HamOp:
        """(t: expm(-i * h * t)"""

        v = h * self._coeff

        return torch.cos(v)*I() - 1j*torch.sin(v)*self.as_unit_operator()

    @property
    def coeff(self) -> Tensor:
        return self._coeff

    @property
    def n_qubits(self) -> int:
        return max(self._x_mask.bit_length(), self._z_mask.bit_length(), 1)

    def __single_qubit_matrix(self, i: int) -> Tensor:
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
    def from_label(label: str, coeff: complex | Tensor = 1.0):
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
                raise ValueError(f"Invalid character: {char}")

        return PauliString(x_mask, z_mask, coeff, len(label))

    @staticmethod
    def __phase_factor(a: PauliString, b: PauliString):
        x1, z1 = a._x_mask, a._z_mask
        x2, z2 = b._x_mask, b._z_mask

        n_qubits = max(x1.bit_length(), z1.bit_length(), x2.bit_length(), z2.bit_length())

        p = 0

        for i in range(n_qubits):
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
