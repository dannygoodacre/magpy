from __future__ import annotations
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .hamiltonian_operator import HamOp
from .linalg import kron
from ..types import Scalar

if TYPE_CHECKING:
    from ..types import Operator


def X(index: int = 0) -> PauliString:
    return PauliString(1 << index, 0)

def Y(index: int = 0) -> PauliString:
    return PauliString(1 << index, 1 << index)

def Z(index: int = 0) -> PauliString:
    return PauliString(0, 1 << index)

def I(index: int = 0) -> PauliString:
    return PauliString(0, 0)


class PauliString:
    _X_MATRIX = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)

    _Y_MATRIX = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)

    _Z_MATRIX = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)

    _I_MATRIX = torch.eye(2, dtype=torch.complex128)

    def __init__(self, x_mask: int, z_mask: int, coeff: complex | Tensor = 1.0):
        from .._utils import tensorize

        self._x_mask: int = x_mask
        self._z_mask: int = z_mask

        self._coeff: Scalar = tensorize(coeff)

    def __add__(self, other: Operator) -> Operator:
        result = HamOp(self, other)

        return result.simplify()

    def __call__(self, t: Tensor = None, **kwargs) -> PauliString:
        return self

    def __copy__(self):
        return self.copy()

    def __hash__(self):
        return hash((self._x_mask, self._z_mask, self.n_qubits))

    def __eq__(self, other):
        if not isinstance(other, PauliString):
            return False

        return self._x_mask == other._x_mask \
            and self._z_mask == other._z_mask \
            and self.n_qubits == other.n_qubits

    def __mul__(self, other: Scalar | Operator) -> Operator:
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

    def __rmul__(self, other: Scalar | Operator) -> Operator:
        from ..types import Scalar
        if isinstance(other, Scalar | PauliString):
            return self * other

        if isinstance(other, HamOp):
            return other * self

        if callable(other):
            return HamOp((other, self))

    def __str__(self):
        from .._context import get_print_identities
        from .._utils import format_number

        label = []

        for i in range(self.n_qubits):
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

    def __sub__(self, other: Operator) -> Operator:
        return HamOp(self, -other)

    def as_unit_operator(self) -> PauliString:
        """Create a copy of the operator with unit coefficient.

        Returns
        -------
        PauliString
            A new instance with the same qubit structure and unit coefficient.
        """
        return PauliString(self._x_mask, self._z_mask)

    def copy(self) -> PauliString:
        """Create a shallow copy of the operator.

        Returns
        -------
        PauliString
            A new instance containing the same terms as `self`.
        """

        return PauliString(self._x_mask, self._z_mask, self._coeff)

    def power(self, n: int) -> PauliString:
        """Compute the n-th power of the operator.

        Parameters
        ----------
        n : int
            A non-negative integer.

        Returns
        -------
        PauliString
            The resulting operator.
        """

        if n == 0:
            return I()

        if n == 1:
            return self

        coeff = self._coeff ** n

        if n % 2 == 0:
            return coeff * I()

        return self.copy()

    def tensor(self, t: Tensor = None, n_qubits: int = None) -> Tensor:
        """Calculate the dense matrix representation of the operator.

        If the operator contains time-dependent coefficients, then the `t` parameter is used to
        evaluate them. If `t` is a tensor of multiple values, then the resulting matrix will be
        batched accordingly.

        Parameters
        ----------
        t : Tensor, optional
            Time value(s) at which to evaluate the operator, by default None
        n_qubits : int, optional
            Total number of qubits, by default None

        Returns
        -------
        Tensor
            A complex-valued, dense matrix of shape `(2^n, 2^n)` if `t` is a scalar or None.
            If `t` or the operator is batched, returns shape `(batch_count, 2^n, 2^n)`.
        """

        if n_qubits is None:
            n_qubits = self.n_qubits

        result = torch.tensor([[1.0]], dtype=torch.complex128)

        for i in range(n_qubits):
            xi = (self._x_mask >> i) & 1
            zi = (self._z_mask >> i) & 1

            if xi == 0 and zi == 0:
                m = torch.eye(2, dtype=torch.complex128)

            elif xi == 1 and zi == 0:
                m = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.complex128)

            elif xi == 0 and zi == 1:
                m = torch.tensor([[1., 0.], [0., -1.]], dtype=torch.complex128)

            else:
                m = torch.tensor([[0., -1j], [1j, 0.]], dtype=torch.complex128)

            result = torch.kron(result, m)

        scale = self._coeff

        if torch.is_tensor(scale) and scale.ndim > 0:
            scale = scale.view(-1, 1, 1)

        return scale * result

    def propagator(self, h: Tensor = torch.tensor(1, dtype=torch.complex128)) -> HamOp:
        """Compute the unitary propagator exp(-i * P * h) of the operator.

        Parameters
        ----------
        h : Tensor, optional
            The scaling factor for the exponent, by default torch.tensor(1.0, dtype=torch.complex128)

        Returns
        -------
        HamOp
            The resultant operator
        """

        v = h * self._coeff

        return torch.cos(v)*I() - 1j*torch.sin(v)*self.as_unit_operator()

    @property
    def coeff(self) -> Tensor:
        """The coefficient of the operator."""

        return self._coeff

    @property
    def batch_count(self) -> int:
        """The number of parallel simulations represented by the operator."""

        try:
            return self.coeff.shape[0] if isinstance(self.coeff, Tensor) else 1

        except IndexError:
            return 1

    @property
    def H(self) -> PauliString:
        """The Hermititan adjoint."""

        return PauliString(self._x_mask, self._z_mask, self._coeff.conj())

    @property
    def n_qubits(self) -> int:
        """The total number of qubits the operatorupon."""

        return max(self._x_mask.bit_length(), self._z_mask.bit_length(), 1)

    @property
    def qubit_map(self) -> dict[int, str]:
        mapping = {}

        max_bits = max(self._x_mask.bit_length(), self._z_mask.bit_length())

        for i in range(max_bits):
            x = (self._x_mask >> i) & 1
            z = (self._z_mask >> i) & 1

            if x and z:
                mapping[i] = 'Y'
            elif x:
                mapping[i] = 'X'
            elif z:
                mapping[i] = 'Z'

        return mapping

    @property
    def shape(self) -> tuple[int, int, int]:
        """The dimensions of the dense tensor representation.

        Returns
        -------
        tuple[int, int, int]
            `(batch_count, 2^n_qubits, 2^n_qubits)`
        """
        return (self.batch_count, self.n_qubits**2, self.n_qubits**2)

    @property
    def weight(self) -> int:
        """The number of non-identity qubits."""

        active_qubits = self._x_mask | self._z_mask

        return active_qubits.bit_count()

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
    def from_label(label: str, coeff: Scalar = 1.0) -> PauliString:
        """Create a PauliString from a string label (e.g., 'XIYZ').

        The mapping from characters to qubits is big-endian.

        Parameters
        ----------
        label : str
            A string of 'I', 'X', 'Y', or 'Z'.
        coeff : Scalar, optional
            A scalar coefficient, by default 1.0

        Returns
        -------
        PauliString
            The resultant operator.

        Raises
        ------
        ValueError
            If `label` contains invalid characters.
        """

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
                raise ValueError(
                    f"Invalid character '{char}' at position {i} in label '{label}'. "
                    "Labels must only contain 'I', 'X', 'Y', or 'Z'."
                )

        return PauliString(x_mask, z_mask, coeff, len(label))

    @staticmethod
    def __phase_factor(a: PauliString, b: PauliString) -> complex:
        """Calculate the phase factor accumulated from the product of two PauliStrings.

        Parameters
        ----------
        a : PauliString
            The left-hand operand.
        b : PauliString
            The right-hand operand.

        Returns
        -------
        complex
            The resultant phase factor.
        """

        x1, z1 = a._x_mask, a._z_mask
        x2, z2 = b._x_mask, b._z_mask

        pos = (x1 & ~z1 & x2 & z2).bit_count()
        pos += (x1 & z1 & ~x2 & z2).bit_count()
        pos += (~x1 & z1 & x2 & ~z2).bit_count()

        neg = (x1 & z1 & x2 & ~z2).bit_count()
        neg += (~x1 & z1 & x2 & z2).bit_count()
        neg += (x1 & ~z1 & ~x2 & z2).bit_count()

        return 1j ** ((pos - neg) % 4)
