from __future__ import annotations

from typing import Callable, TYPE_CHECKING

import torch
from torch import Tensor

from .._registry import is_type, register
from ..types import Scalar

if TYPE_CHECKING:
    from .pauli_string import PauliString
    from .. import Operator

@register
class HamiltonianOperator:
    def __init__(self, *args):
        """Initialize a Hamiltonian operator from PauliString instances, tuples, or other HamOp instances."""

        self._data: dict[PauliString, Scalar | Callable] = {}

        for arg in args:
            if isinstance(arg, tuple):
                scale, pauli_string = arg

                self._add_term(pauli_string, scale)

            elif is_type(arg, 'PauliString'):
                self._add_term(arg)

            elif isinstance(arg, HamiltonianOperator):
                for pauli_string, coeff in arg._data.items():
                    self._add_term(pauli_string, coeff)

    def __add__(self, other: Operator) -> Operator:
        result = HamiltonianOperator(self, other)

        return result.simplify()

    def __call__(self, t: Tensor = None, **kwargs) -> Operator:
        """Evaluate the operator at the given parameters.

        Returns
        -------
        Operator
            The resultant operator.
        """

        result = HamiltonianOperator()

        for operator, coeff in self._data.items():
            result._add_term(operator, coeff(t, **kwargs) if callable(coeff) else coeff)

        return result.simplify()

    def __copy__(self):
        return self.copy()

    def __mul__(self, other: Scalar | Operator) -> Operator:
        if isinstance(other, Scalar):
            return self.__scalar_mul(other)

        if is_type(other, 'PauliString'):
            other = HamiltonianOperator(other)

        if isinstance(other, HamiltonianOperator):
            result = HamiltonianOperator()

            for p1, c1 in self._data.items():
                for p2, c2 in other._data.items():
                    result._add_term(p1 * p2, c1 * c2)

            return result

        return NotImplemented

    def __neg__(self):
        return self.__scalar_mul(-1)

    def __pow__(self, n: int):
        return self.power(n)

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        if isinstance(other, Scalar):
            return self.__scalar_mul(other)

        if is_type(other, 'PauliString'):
            return HamiltonianOperator(other) * self

        if isinstance(other, HamiltonianOperator):
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
                from .._utils import format_number

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
        return HamiltonianOperator(self, -other)

    def copy(self) -> HamiltonianOperator:
        """Create a shallow copy of the operator.

        Returns
        -------
        HamOp
            A new instance containing the same terms as `self`.
        """

        result = HamiltonianOperator()
        result._data = self._data.copy()

        return result

    def power(self, n: int) -> Operator:
        """Compute the n-th power of the operator.

        Parameters
        ----------
        n : int
            A non-negative integer.

        Returns
        -------
        Operator
            The resulting operator.
        """

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

    def static_commuting_propagator(self, h: Tensor = torch.tensor(1, dtype=torch.complex128)) -> HamiltonianOperator:
        """Compute the unitary propagator exp(-i * H * h) for a static, commuting operator.

        Parameters
        ----------
        h : float | Tensor, optional
            The scaling factor for the exponent, by default 1.0

        Returns
        -------
        HamOp
            The resultant operator.
        """

        if not self.is_static or not self.is_commuting:
            return NotImplemented

        result = None

        for pauli_string, coeff in self._data.items():
            if result is None:
                result = (coeff * pauli_string).propagator(h)

                continue

            result *= (coeff * pauli_string).propagator(h)

        return result

    def simplify(self) -> HamiltonianOperator | PauliString:
        """Reduce the operator to a PauliString if it contains only one static term.

        Returns
        -------
        HamOp | PauliString
            The single PauliString if possible, otherwise `self`.
        """

        if len(self._data) == 1 and self.is_static:
            pauli_unit, coeff = next(iter(self._data.items()))

            return coeff * pauli_unit

        return self

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

        op = self(t) if t is not None else self

        if n_qubits is None:
            n_qubits = op.n_qubits

        if not op._data:
            return torch.zeros((1,1), dtype=torch.complex128)

        result = None

        for pauli_unit, coeff in op._data.items():
            matrix = pauli_unit.tensor(n_qubits=n_qubits)

            c = torch.as_tensor(coeff)

            if c.ndim > 0:
                c = c.view(*c.shape, 1, 1)

            if result is None:
                result = c * matrix

                continue

            result = result + c*matrix

        return result

    @property
    def batch_count(self) -> int:
        """The number of parallel simulations represented by the operator."""

        batch = 1
        for coeff in self._data.values():

            val = coeff.scale if is_type(coeff, 'FunctionProduct') else coeff

            if isinstance(val, Tensor):
                if val.ndim > 0:
                    batch = max(batch, val.shape[0])

            elif isinstance(val, (list, tuple)):
                batch = max(batch, len(val))

        return batch

    @property
    def coeffs(self) -> tuple[Tensor | Callable]:
        """The constituent coefficients."""

        return tuple(coeff for coeff in self._data.values())

    @property
    def H(self) -> HamiltonianOperator:
        """The Hermititan adjoint."""

        result = None
        for pauli_unit, coeff in self._data.items():
            if result is None:
                result = pauli_unit * coeff.conj()

                continue

            result += pauli_unit * coeff.conj()

        return result

    @property
    def is_commuting(self) -> bool:
        """True if the constiuent terms of the operator mutually commute."""

        if len(self._data) <= 1:
            return True

        from ..linalg import commutes

        operators = list(self._data.keys())

        for i in range(len(operators)):
            for j in range(i + 1, len(operators)):
                if not commutes(operators[i], operators[j]):
                    return False

        return True

    @property
    def is_disjoint(self) -> bool:
        """True if the system is completely independent.

        Each qubit is affected by at most one type of Pauli operator.
        """

        if not self.is_local:
            return False

        active_axes: dict[int, str] = {}

        for pauli_unit in self._data.keys():
            for qubit_index, pauli_type in pauli_unit.qubit_map.items():
                if qubit_index in active_axes and active_axes[qubit_index] != pauli_type:
                    return False

            active_axes[qubit_index] = pauli_type

        return True

    @property
    def is_local(self):
        """True if every term acts on at most one qubit."""

        return not any(p.weight > 1 for p in self._data.keys())

    @property
    def is_static(self) -> bool:
        """True if all coefficients are constant."""

        return not any(callable(coeff) for coeff in self._data.values())

    @property
    def n_qubits(self) -> int:
        """The total number of qubits the operatorupon."""

        if not self._data:
            return 0

        return max(p.n_qubits for p in self._data.keys())

    @property
    def pauli_strings(self) -> tuple[PauliString]:
        """The PauliString components."""

        return tuple(p for p in self._data.keys())

    @property
    def shape(self) -> tuple[int, int, int]:
        """The dimensions of the dense tensor representation.

        Returns
        -------
        tuple[int, int, int]
            `(batch_count, 2^n_qubits, 2^n_qubits)`
        """
        dim = 2 ** self.n_qubits

        return (self.batch_count, dim, dim)

    @property
    def terms(self) -> tuple[tuple[PauliString, Tensor | Callable]]:
        """The (PauliString, coefficient) pairs constituting the operator."""

        return tuple((operator, coeff) for operator, coeff in self._data.items())

    def _add_term(self, pauli_string: PauliString, scale: Scalar = 1):
        operator = pauli_string.as_unit_operator()

        if callable(scale):
            from .function_product import FunctionProduct

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
        from .._utils import tensorize

        other = tensorize(other)

        result = HamiltonianOperator()

        result._data = {
            pauli_string: coeff * other
            for pauli_string, coeff in self._data.items()
        }

        return result
