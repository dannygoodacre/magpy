from itertools import chain
from copy import deepcopy
from numbers import Number
import torch
import magpy as mp
from .._device import _DEVICE_CONTEXT

class HamiltonianOperator:
    """A Hamiltonian operator.

    A representation of a Hamiltonian operator, formed of a sum of Pauli
    operators with functional coefficients.

    Attributes
    ----------
    data : dict
        Functional coefficients and their PauliStrings
    """

    def __init__(self, *pairs):
        """A Hamiltonian operator.

        Parameters
        ----------
        *pairs : tuple
            Pairs of functions and PauliStrings
        """

        self.data = {}

        for pair in pairs:
            try:
                # Move any constant coefficients to the corresponding PauliString.
                if pair[0].scale != 1:
                    pair[1] *= pair[0].scale
                    pair[0].scale = 1
            except AttributeError:
                if isinstance(pair[0], Number):
                    pair[1] *= pair[0]
                    pair[0] = 1

            try:
                self.data[pair[0]].append(pair[1])
            except KeyError:
                self.data[pair[0]] = pair[1]
            except AttributeError:
                self.data[pair[0]] = [self.data[pair[0]], pair[1]]

        HamiltonianOperator.__simplify(self.data)

    def __eq__(self, other):
        return self.data == other.data

    def __mul__(self, other):
        out = HamiltonianOperator()

        try:
            self_data = HamiltonianOperator.__expand(self.data)
            other_data = HamiltonianOperator.__expand(other.data)

            for p in self_data:
                for q in other_data:
                    out += mp.FunctionProduct() * p[0] * q[0] * p[1] * q[1]

            return out

        except AttributeError:
            out.data = deepcopy(self.data)

            if isinstance(other, Number | mp.PauliString):
                for coeff in out.data:
                    try:
                        for i in range(len(out.data[coeff])):
                            out.data[coeff][i] *= other
                    except TypeError:
                        out.data[coeff] *= other

            else:
                # other is FunctionProduct or other type of function.
                for coeff in list(out.data):
                    out.data[mp.FunctionProduct(coeff, other)] = out.data.pop(coeff)

            return out

    __rmul__ = __mul__

    def __add__(self, other):
        out = HamiltonianOperator()

        try:
            out.data = self.data | other.data
        except AttributeError:
            # other is PauliString; add it to constants.
            out.data = self.data.copy()

            try:
                out.data[1].append(other)
            except KeyError:
                out.data[1] = other
            except AttributeError:
                out.data[1] = [out.data[1], other]
        else:
            # other is HamiltionianOperator.
            for coeff in list(set(self.data.keys() & other.data.keys())):
                out.data[coeff] = []

                try:
                    out.data[coeff].extend(self.data[coeff])
                except TypeError:
                    out.data[coeff].append(self.data[coeff])

                try:
                    out.data[coeff].extend(other.data[coeff])
                except TypeError:
                    out.data[coeff].append(other.data[coeff])

        HamiltonianOperator.__simplify(out.data)
        return out

    def __neg__(self):
        out = HamiltonianOperator()
        out.data = deepcopy(self.data)
        return -1 * out

    def __sub__(self, other):
        out = self + -other
        return out

    def __repr__(self):
        return '{' + ', '.join((str(f) if isinstance(f, Number)
                                else f.__name__) + ': ' + str(q) for f, q in self.data.items()) + '}'

    def __call__(self, t=None, n=None):
        if n is None:
            pauli_strings = chain.from_iterable(p if isinstance(p, list) else [p] for p in self.data.values())
            n = max(max(p.qubits) for p in pauli_strings)

        if self.is_constant():
            try:
                return sum(p(n).type(torch.complex128) for p in self.data[1])
            except TypeError:
                return self.data[1](n).type(torch.complex128)

        if t is None:
            raise ValueError(
                "Hamiltonian is not constant. A value of t is required.")

        out = 0
        for coeff, ps in self.data.items():
            try:
                out += coeff(torch.tensor(t)) * ps(n).type(torch.complex128)
            except TypeError:
                if isinstance(coeff, Number):
                    out += coeff * ps(n).type(torch.complex128)
                else:
                    for p in ps:
                        out += coeff(torch.tensor(t)) * p(n).type(torch.complex128)

        return out.to(_DEVICE_CONTEXT.device)

    def is_constant(self):
        "Return true if the Hamiltonian is time-independent."
        for coeff in self.data:
            if not isinstance(coeff, Number):
                return False
        return True

    def is_interacting(self):
        """Return true if the Hamiltonian's qubits are interacting."""
        for ps in self.data.values():
            try:
                if len(ps.qubits) != 1:
                    return True
            except AttributeError:
                for p in ps:
                    if len(p.qubits) != 1:
                        return True
        return False

    def unpack_data(self):
        """All function-operator pairs in H, unpacking those with shared functions."""
        return [(k, v) for k, items in self.data.items() for v in (items if isinstance(items, list) else [items])]

    def funcs(self):
        """All functions in H."""
        # return list(self.data.keys())
        return [u[0] for u in self.unpack_data()]

    def pauli_operators(self):
        """All Pauli operators in H."""
        return [u[1] for u in self.unpack_data()]

    @staticmethod
    def __simplify(arrs):
        # Collect all PauliStrings in all lists in arrs.
        for coeff in arrs:
            arrs[coeff] = mp.PauliString.collect(arrs[coeff])

    @staticmethod
    def __expand(data):
        # Expand all functions and lists of qubits into pairs of functions with single qubits.
        expanded_data = []
        for pair in data.items():
            try:
                for qubit in pair[1]:
                    expanded_data.append((pair[0], qubit))
            except TypeError:
                expanded_data.append(pair)
        return expanded_data
