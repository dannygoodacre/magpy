from numbers import Number
from copy import copy, deepcopy
import torch
import magpy as mp
from .._device import _DEVICE_CONTEXT


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

    Attributes
    ----------
    qubits : dict
        The qubits and their indices in the operator
    scale : Number | Tensor
        Constant coefficient
    """

    _matrices = {
        'X': torch.tensor([[0, 1], [1, 0]]).to(_DEVICE_CONTEXT.device),
        'Y': torch.tensor([[0, -1j], [1j, 0]]).to(_DEVICE_CONTEXT.device),
        'Z': torch.tensor([[1, 0], [0, -1]]).to(_DEVICE_CONTEXT.device)
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

        self.qubits = {}
        self.scale = scale

        for q, label in zip([x, y, z], ["X", "Y", "Z"]):
            try:
                self.qubits |= {n: label for n in q}

            except TypeError:
                if q is not None:
                    self.qubits[q] = label

    def __add__(self, other):
        try:
            if self.qubits == other.qubits:
                scale = self.scale + other.scale

                if torch.is_tensor(scale) and torch.all(scale.eq(0)) or isinstance(scale, Number) and scale == 0:
                    return 0

                out = PauliString()
                out.scale = scale
                out.qubits = self.qubits

                return out

            return mp.HamiltonianOperator([1, self], [1, other])

        except AttributeError:
            # other is HOp.
            return other + self
        
    def __call__(self, n_qubits=None) -> torch.Tensor:
        if n_qubits is None:
            n_qubits = self.n_qubits

        qubits = [torch.eye(2) for _ in range(n_qubits)]

        for index, qubit in self.qubits.items():
            qubits[index - 1] = PauliString._matrices[qubit]

        try:
            scale = self.scale.view(len(self.scale), 1, 1)
        except (AttributeError, TypeError):
            scale = self.scale

        return scale * mp.kron(*qubits).type(torch.complex128)

    def __eq__(self, other):
        return self.qubits == other.qubits and self.scale == other.scale

    def __mul__(self, other):
        if isinstance(other, mp.PauliString):
            return self.__mul_pauli_string(other)

        if isinstance(other, mp.HamiltonianOperator):
            return self.__mul_hamiltonian_operator(other)

        if isinstance(other, Number | torch.Tensor):
            return self.__mul_scalar(other)

        try:
            # FunctionProduct.
            self *= other.scale
            other.scale = 1

        except AttributeError:
            # Other type of function.
            pass

        return mp.HamiltonianOperator([other, self])

    def __neg__(self):
        out = PauliString()
        out.scale = -self.scale
        out.qubits = self.qubits

        return out
    
    def __radd__(self, other):
        if other == 0:
            return self

    def __rmul__(self, other):
        if isinstance(other, Number | torch.Tensor):
            return self.__mul_scalar(other)
        
        return self * other
    
    def __repr__(self):
        out = ""

        if isinstance(self.scale, torch.Tensor) or self.scale != 1:
            out += str(self.scale) + "*"

        if self.qubits.items():
            out += "*".join(q[1] + str(q[0]) for q in sorted(self.qubits.items()))
        else:
            out += "Id"

        return out

    def __rsub__(self, other):
        return -self + other

    def __str__(self):
        return repr(self)

    def __sub__(self, other):
        return self + -other
    
    def as_single_qubit(self):
        if len(self.qubits) == 0:
            return Id()

        if len(self.qubits) > 1:
            raise RuntimeError(
                'Cannot convert to single qubit')

        (_, op), = self.qubits.items()

        match op:
            case 'X':
                return X()
            case 'Y':
                return Y()
            case 'Z':
                return Z()

    def matrix(self, n_qubits: int = None) -> torch.Tensor:
        """The matrix representation of the Pauli string."""

        return self(n_qubits)
    
    def expm(self):
        "The exponential of the Pauli string."
        
        a = torch.tensor(self.scale)
        
        return torch.cosh(a)*Id() + (torch.sinh(a) / a)*self
    
    @property
    def batch_count(self) -> int:
        """The number of parallel qubits in the batch Pauli string."""
        
        return self.scale.shape[0] if isinstance(self.scale, torch.Tensor) else 1
 
    @property
    def n_qubits(self) -> int:
        return max(self.qubits) if self.qubits else 1
    
    @property
    def operator(self):
        result = copy(self)
        result.scale = 1

        return result

    def qutip(self, n_qubits=None):
        """Convert to QuTiP form."""

        import qutip as qt

        states = {
            'X' : qt.sigmax(),
            'Y' : qt.sigmay(),
            'Z' : qt.sigmaz(),
            'Id' : qt.qeye(2)
        }

        n_qubits = (self.n_qubits if n_qubits is None else n_qubits) + 1

        if not self.qubits:
            return qt.eye(2 ** n_qubits)

        return self.scale * qt.tensor(states.get(self.qubits.get(i, 'Id')) for i in range(1, n_qubits))

    def __mul_pauli_string(self, other):
        result = PauliString()
        result.scale = self.scale * other.scale
        result.qubits = self.qubits | other.qubits

        for n in list(set(self.qubits.keys() & other.qubits.keys())):
            if self.qubits[n] == other.qubits[n]:
                del result.qubits[n]

            else:
                phase, qubit = PauliString.__char_compose(self.qubits[n], other.qubits[n])

                result.scale *= 1j * phase
                result.qubits[n] = qubit

        return result

    def __mul_hamiltonian_operator(self, other):
        return mp.HamiltonianOperator([1, self]) * other

    def __mul_scalar(self, other):

        result = deepcopy(self)
        result.scale *= other

        return result

    @staticmethod
    def collect(arr):
        """Group PauliStrings in list which have the same qubit structure.

        Parameters
        ----------
        arr : list[PauliString]
            PauliString instances

        Returns
        -------
        list[PauliString]|PauliString
            Collected instances or single instance
        """

        unique_qubit_counts = {}

        try:
            # list[PauliString].
            for pauli_string in arr:
                qubits = tuple(pauli_string.qubits.items())

                unique_qubit_counts[qubits] = unique_qubit_counts.get(qubits, 0) + pauli_string.scale

        except TypeError:
            # PauliString.
            return arr

        result = []

        for qubits, count in unique_qubit_counts.items():
            pauli_string = PauliString()

            pauli_string.qubits = dict(qubits)
            pauli_string.scale = count

            result.append(pauli_string)

        return result[0] if len(result) == 1 else result

    @staticmethod
    def __char_compose(a, b):
        products = {
            ('X', 'Y'): (1, 'Z'), ('Y', 'X'): (-1, 'Z'),
            ('Y', 'Z'): (1, 'X'), ('Z', 'Y'): (-1, 'X'),
            ('Z', 'X'): (1, 'Y'), ('X', 'Z'): (-1, 'Y'),
        }

        return products[(a,  b)]
 
    @staticmethod
    def _update_device():
        PauliString._matrices['X'] = PauliString._matrices['X'].to(_DEVICE_CONTEXT.device)
        PauliString._matrices['Y'] = PauliString._matrices['Y'].to(_DEVICE_CONTEXT.device)
        PauliString._matrices['Z'] = PauliString._matrices['Z'].to(_DEVICE_CONTEXT.device)
