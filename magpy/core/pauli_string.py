from __future__ import annotations
from numbers import Number

import torch
from torch import Tensor

import magpy as mp
from .._context import get_device
from .._utils import format_number, format_tensor


def X(*args):
    """Multi-qubit operator formed of Pauli X operators."""
    return PauliString(x=args if args else 1)


def Y(*args):
    """Multi-qubit operator formed of Pauli Y operators."""
    return PauliString(y=args if args else 1)


def Z(*args):
    """Multi-qubit operator formed of Pauli Z operators."""
    return PauliString(z=args if args else 1)


def Id():
    """The identity operator."""
    return PauliString()


class PauliString:
    """A multi-qubit Pauli operator.

    A representation of the tensor product of single qubit Pauli operators.
    Identity operators are inferred from the gaps in the indices in the
    internal dictionary.
    
    A batch operator may be created by providing a tensor coefficient.

    Examples
    --------
    >>> a = 3 * X()
    >>> a
    3*X1
    >>> a.matrix()
    tensor([[0.+0.j, 3.+0.j],
            [3.+0.j, 0.+0.j]], dtype=torch.complex128)

    >>> b = torch.tensor([1,2]) * Y()
    >>> b
    tensor([1, 2])*Y1
    >>> b.matrix()
    tensor([[[0.+0.j, 1.+0.j],
            [1.+0.j, 0.+0.j]],
            [[0.+0.j, 2.+0.j],
            [2.+0.j, 0.+0.j]]], dtype=torch.complex128)
    """

    _matrices = {
        'X': torch.tensor([[0, 1], [1, 0]]).to(get_device()),
        'Y': torch.tensor([[0, -1j], [1j, 0]]).to(get_device()),
        'Z': torch.tensor([[1, 0], [0, -1]]).to(get_device())
    }

    def __init__(self, x=None, y=None, z=None, scale=1):
        """A multi-qubit Pauli operator.

        Parameters
        ----------
        x, y, z : set[int], optional
            Position of the Pauli qubits in the operator, by default None
        scale : int | Tensor, optional
            Constant coefficient, by default 1
        """

        self._qubits = {}
        self._scale = scale

        for qubit, label in zip([x, y, z], ['X', 'Y', 'Z']):
            try:
                self._qubits |= {n: label for n in qubit}

            except TypeError:
                if qubit is not None:
                    self._qubits[qubit] = label

    def __add__(self, other):
        try:
            if self._qubits != other._qubits:
                return mp.HamiltonianOperator((1, self), (1, other))

            scale = self._scale + other._scale

            if torch.is_tensor(scale) and torch.all(scale.eq(0)) or isinstance(scale, Number) and scale == 0:
                return 0

            out = PauliString()
            out._scale = scale
            out._qubits = self._qubits

            return out

        except AttributeError:
            return other + self

    def __eq__(self, other):
        try:
            return self._qubits == other._qubits and self._scale == other._scale

        except AttributeError:
            return False

    def __mul__(self, other):
        result = PauliString()
        result._scale = self._scale
        result._qubits = dict(self._qubits)

        if isinstance(other, mp.HamiltonianOperator):
            return mp.HamiltonianOperator([1, result]) * other

        if isinstance(other, Number | torch.Tensor):
            result._scale *= other

            return result
        
        if callable(other):
            try:
                result._scale *= other._scale

                other._scale = 1

            except AttributeError:
                pass

            return mp.HamiltonianOperator([other, result])

        result._scale = result._scale * other._scale
        result._qubits |= other._qubits

        for n in list(set(self._qubits.keys() & other._qubits.keys())):
            if self._qubits[n] == other._qubits[n]:
                del result._qubits[n]

            else:
                products = {
                    ('X', 'Y'): (1, 'Z'), ('Y', 'X'): (-1, 'Z'),
                    ('Y', 'Z'): (1, 'X'), ('Z', 'Y'): (-1, 'X'),
                    ('Z', 'X'): (1, 'Y'), ('X', 'Z'): (-1, 'Y'),
                }
                
                phase, qubit = products[(self._qubits[n], other._qubits[n])]
                
                result._scale *= 1j * phase

                result._qubits[n] = qubit

        return result

    def __neg__(self):
        result = PauliString()
        result._qubits = self._qubits
        result._scale = -self._scale

        return result

    def __radd__(self, other):
        if other == 0:
            return self

    def __repr__(self):
        result = ''

        if isinstance(self._scale, torch.Tensor):
            result += format_tensor(self._scale) + '*'

        elif self._scale != 1:
            result += format_number(self._scale) + '*'

        if self._qubits.items():
            result += '*'.join(q[1] + str(q[0]) for q in sorted(self._qubits.items()))

        else:
            result += 'Id'

        return result

    __rmul__ = __mul__

    def __rsub__(self, other):
        return -self + other
    
    __str__ = __repr__

    def __sub__(self, other):
        return self + -other

    @property
    def as_single_qubit(self) -> PauliString:
        """The Pauli string acting on the first qubit only.
        
        If it acts on multiple qubits, this is `None`.
        """

        if not self._qubits:
            return Id()
        
        if len(self._qubits) > 1:
            return None
        
        (qubit, op), = self._qubits.items()

        self._target_qubit = qubit

        match op:
            case 'X':
                return X()
            case 'Y':
                return Y()
            case 'Z':
                return Z()

    @property
    def batch_count(self) -> int:
        """The number of parallel operators represented by the Pauli string."""

        return self._scale.shape[0] if isinstance(self._scale, torch.Tensor) else 1
    
    @property
    def H(self) -> PauliString:
        """The conjugate transpose of the operator."""
        result = PauliString()
        result._scale = self._scale.conjugate()
        result._qubits = dict(self._qubits)
        
        return result

    @property
    def is_single_qubit(self) -> bool:
        """Whether the Pauli string acts on a single qubit.
        
        If false, then `as_single_qubit` is `None`.
        """

        return not self._qubits or len(self._qubits) == 1

    @property
    def n_qubits(self) -> int:
        """The number of qubits acted on by the Pauli string."""

        return max(self._qubits) if self._qubits else 1
    
    @property
    def operator(self) -> PauliString:
        """The Pauli string with no constant coefficient."""

        result = PauliString()
        result._qubits = self._qubits

        return result

    @property
    def scale(self) -> Number | Tensor:
        """The constant coefficient of the Pauli string."""

        return self._scale
    
    @property
    def shape(self) -> tuple[Number, Number]:
        return (self.n_qubits**2, self.n_qubits**2)
    
    @property
    def target_qubit(self) -> int:
        """The qubit position acted on by the Pauli string when it is a single qubit.
        
        If the Pauli string acts on multiple qubits, this is `None`."""

        return next(iter(self.__qubits)) if len(self._qubits) == 1 else None

    def matrix(self, n_qubits: int = None) -> Tensor:
        """The matrix representation of the Pauli string.

        If `n_qubits` is not specified, then it is determined to be the largest qubit position internally.

        Parameters
        ----------
        n_qubits : int, optional
            The number of qubits over which to evaluate the matrix, by default None

        Returns
        -------
        Tensor
            The matrix representation.
        """

        if n_qubits is None:
            n_qubits = self.n_qubits

        qubits = [torch.eye(2) for _ in range(n_qubits)]

        for index, qubit in self._qubits.items():
            qubits[index - 1] = PauliString._matrices[qubit]

        try:
            scale = self._scale.view(len(self._scale), 1, 1)

        except (AttributeError, TypeError):
            scale = self._scale

        return scale * mp.kron(*qubits).type(torch.complex128)

    def propagator(self, h: Tensor = torch.tensor(1, dtype=torch.complex128)) -> PauliString | mp.HamiltonianOperator:
        """The exponential of `-i * h * P`."""

        v = self.scale * h

        return torch.cos(v)*Id() - 1j*torch.sin(v)*self.operator

    def qutip(self, n_qubits: int = None):
        """The Pauli string as a `QObj` instance."""

        import qutip as qt

        states = {
            'X' : qt.sigmax(),
            'Y' : qt.sigmay(),
            'Z' : qt.sigmaz(),
            'Id' : qt.qeye(2)
        }

        n_qubits = (self.n_qubits if n_qubits is None else n_qubits) + 1

        if not self._qubits:
            return qt.eye(2 ** n_qubits)

        return self._scale * qt.tensor(states.get(self._qubits.get(i, 'Id')) for i in range(1, n_qubits))

    @staticmethod
    def _collect(pauli_strings: list[PauliString]):
        unique_qubit_counts = {}

        try:
            for pauli_string in pauli_strings:
                qubits = tuple(pauli_string._qubits.items())

                unique_qubit_counts[qubits] = unique_qubit_counts.get(qubits, 0) + pauli_string._scale

        except TypeError:
            return pauli_strings

        result = []

        for qubits, count in unique_qubit_counts.items():
            pauli_string = PauliString()

            pauli_string._qubits = dict(qubits)
            pauli_string._scale = count

            result.append(pauli_string)

        return result[0] if len(result) == 1 else result

    @staticmethod
    def _update_device():
        PauliString._matrices['X'] = PauliString._matrices['X'].to(get_device())
        PauliString._matrices['Y'] = PauliString._matrices['Y'].to(get_device())
        PauliString._matrices['Z'] = PauliString._matrices['Z'].to(get_device())
