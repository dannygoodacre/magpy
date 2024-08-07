from numbers import Number
from copy import deepcopy
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

    A representation of a tensor product of single qubit Pauli operators.
    Identity operators are inferred from the gaps in the indices in the
    internal dictionary.

    Attributes
    ----------
    qubits : dict
        The qubits and their indices in the operator
    scale : Number
        Scalar coefficient
    """

    matrices = {
        'X': torch.tensor([[0, 1], [1, 0]]),
        'Y': torch.tensor([[0, -1j], [1j, 0]]),
        'Z': torch.tensor([[1, 0], [0, -1]])
    }

    def __init__(self, x=None, y=None, z=None, scale=1):
        """A multi-qubit Pauli operator.

        Parameters
        ----------
        x, y, z : set[int], optional
            Position of the Pauli qubits in the operator, by default None
        scale : int, optional
            Scalar coefficient, by default 1
        """

        self.qubits = {}
        self.scale = scale

        for q, label in zip([x, y, z], ["X", "Y", "Z"]):
            try:
                self.qubits |= {n: label for n in q}
            except TypeError:
                if q is not None:
                    self.qubits[q] = label

    def __eq__(self, other):
        return self.qubits == other.qubits and self.scale == other.scale

    def __mul__(self, other):
        if isinstance(other, mp.PauliString):
            return self.__mul_ps(other)

        if isinstance(other, mp.HamiltonianOperator):
            return self.__mul_hop(other)

        if isinstance(other, Number | torch.Tensor):
            return self.__mul_num(other)

        try:
            # other is FunctionProduct.
            self *= other.scale
            other.scale = 1
        except AttributeError:
            # other is another type of function
            pass

        return mp.HamiltonianOperator([other, self])

    __rmul__ = __mul__ # This may have to be changed

    def __add__(self, other):
        try:
            if self.qubits == other.qubits:
                scale = self.scale + other.scale
                if torch.is_tensor(scale) and torch.all(scale.eq(0)) or isinstance(scale, Number) and scale == 0:
                    return 0

                out = PauliString(scale=scale)
                out.qubits = self.qubits
                return out

            return mp.HamiltonianOperator([1, self], [1, other])

        except AttributeError:
            # other is HOp
            return other + self

    def __neg__(self):
        s = PauliString(scale=-self.scale)
        s.qubits = self.qubits
        return s

    def __sub__(self, other):
        return self + -other

    def __repr__(self):
        out = ""

        if isinstance(self.scale, torch.Tensor) or self.scale != 1:
            out += str(self.scale) + "*"

        if self.qubits.items():
            out += "*".join(q[1] + str(q[0]) for q in sorted(self.qubits.items()))
        else:
            out += "Id"

        return out

    def __call__(self, n=None):
        if n is None:
            n = max(self.qubits)

        qubits = n * [torch.eye(2)]
        for index, qubit in self.qubits.items():
            qubits[index - 1] = PauliString.matrices[qubit]

        scale = self.scale.view(len(self.scale), 1, 1).to(_DEVICE_CONTEXT.device) \
            if isinstance(self.scale, torch.Tensor) else self.scale
        return scale * mp.kron(*qubits).type(torch.complex128).to(_DEVICE_CONTEXT.device)

    def __mul_ps(self, other):
        # Right multiply by PauliString
        out = PauliString()
        out.scale = self.scale * other.scale
        out.qubits = self.qubits | other.qubits

        for n in list(set(self.qubits.keys() & other.qubits.keys())):
            if self.qubits[n] == other.qubits[n]:
                del out.qubits[n]
            else:
                scale, spin = PauliString.__pauli_mul(self.qubits[n], other.qubits[n])
                out.scale *= 1j * scale
                out.qubits[n] = spin

        return out

    def __mul_hop(self, other):
        # Right multiply by Hamiltonian
        return mp.HamiltonianOperator([1, self]) * other

    def __mul_num(self, other):
        # Multiply by scalar
        out = deepcopy(self)
        out.scale *= other

        return out

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

        counts = {}  # Number of occurrences of each unique PauliString.
        out = []

        try:
            for ps in arr:
                scale = ps.scale
                ps = tuple(ps.qubits.items())

                try:
                    counts[ps] += scale
                except KeyError:
                    counts[ps] = scale
        except TypeError:  # arr is single PauliString.
            return arr

        for ps, c in counts.items():
            a = PauliString()
            a.qubits = dict(ps)
            a.scale = c
            out.append(a)

        return out[0] if len(out) == 1 else out

    @staticmethod
    def __e_ijk(i, j, k):
        # Levi-Civita symbol.
        return int((i - j) * (j - k) * (k - i) / 2)

    @staticmethod
    def __pauli_mul(a, b):
        # Composition of two Pauli qubits.
        if a == b:
            return None
        c = "XYZ".replace(a, "").replace(b, "")
        return PauliString.__e_ijk(ord(a), ord(b), ord(c)), c
