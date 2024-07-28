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

        self.data = HamiltonianOperator.__simplify_and_sort_data(self.data)

    def __eq__(self, other):
        return self.data == other.data

    def __mul__(self, other):
        out = HamiltonianOperator()

        try:
            out = sum((p[0]*p[1]*q[0]*q[1] for q in other.unpack_data() for p in self.unpack_data()), out)

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

        out.data = HamiltonianOperator.__simplify_and_sort_data(out.data)
        return out

    __rmul__ = __mul__

    def __add__(self, other):
        out = HamiltonianOperator()

        try:
            out.data = self.data | other.data
        except AttributeError:
            # other is PauliString; add it to constants.
            out.data = deepcopy(self.data)

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

        out.data = HamiltonianOperator.__simplify_and_sort_data(out.data)
        return out

    def __neg__(self):
        out = HamiltonianOperator()
        out.data = deepcopy(self.data)
        return -1 * out

    def __sub__(self, other):
        out = self + -other
        return out

    def __repr__(self):
        return str(self.data)

    def __str__(self):
        out = ""
        for f, p in self.data.items():
            try:
                p_str = str(p)
                scale_pos = p_str.find('*')

                if isinstance(p.scale, torch.Tensor) or p.scale != 1:
                    out += p_str[:scale_pos] + '*'

                out = HamiltonianOperator.__add_coeff_to_str(out, f)
                out += p_str[scale_pos + 1:] if scale_pos > 0 else p_str

            except AttributeError:
                out = HamiltonianOperator.__add_coeff_to_str(out, f)

                out += '(' if f != 1 else ""
                out += " + ".join([str(q) for q in p])
                out += ')' if f != 1 else ""

            out += " + "

        return out[:-3]

    def __call__(self, t=None, n_qubits=None):
        if n_qubits is None:
            n_qubits = max(max(p.qubits) if p.qubits else 0 for p in self.pauli_operators())

        if self.is_constant():
            return self.__call_time_independent(n_qubits)

        if t is None:
            raise ValueError(
                "Hamiltonian is not constant. A value of t is required.")

        # Convert input to tensor.
        try:
            t = t.clone().detach()
        except AttributeError:
            t = torch.tensor(t)

        return self.__call_time_dependent(t, n_qubits)

    def is_constant(self):
        "Return true if the Hamiltonian is time-independent."
        for coeff in self.data:
            if not isinstance(coeff, Number | torch.Tensor):
                try:
                    if not coeff.is_empty():
                        return False
                except AttributeError:
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

    def __call_time_independent(self, n_qubits):
        # Return matrix representation when constant.
        out = 0
        for p in self.pauli_operators():
            try:
                p_val = p(n_qubits)

                # If p is a batch, repeat the current value to agree with its shape.
                if out.dim() == 2 and p_val.dim() == 3:
                    out = out.repeat(len(p_val), 1, 1)

            except AttributeError:
                pass

            out += p(n_qubits)

        return out

    def __call_time_dependent(self, t, n_qubits):
        out = 0
        for coeff, p in self.unpack_data():
            p_val = p(n_qubits).to(_DEVICE_CONTEXT.device)

            # Evaluate coefficient if it's a function.
            try:
                coeff = coeff(t).to(_DEVICE_CONTEXT.device)
            except TypeError:
                pass

            # Evaluate next term in data.
            next_term = 0
            try:
                next_term = coeff.reshape(-1,1,1) * p_val
            except AttributeError:
                next_term = coeff * p_val

            # If p is a batch, repeat the current value to agree with its shape.
            try:
                if out.dim() == 2 and next_term.dim() == 3:
                    out = out.repeat(len(next_term), 1, 1)
            except AttributeError:
                pass

            out += next_term

        return out

    @staticmethod
    def __simplify_data(arrs):
        # Collect all PauliStrings in all lists in arrs.
        for coeff in arrs:
            arrs[coeff] = mp.PauliString.collect(arrs[coeff])

    @staticmethod
    def __add_coeff_to_str(out, f):
        try:
            out += f.__name__ + '*'
        except AttributeError:
            try:
                if f != 1:
                    out += str(f) + '*'
            except (RuntimeError, AttributeError):
                out += str(f) + '*'

        return out

    @staticmethod
    def __sort_data(data):
        # Move all constant keys to the start of the dictionary.
        const_keys = []
        other_keys = []

        for key in data:
            (const_keys if isinstance(key, Number | torch.Tensor) else other_keys).append(key)

        return dict((key, data[key]) for key in const_keys + other_keys)

    @staticmethod
    def __simplify_and_sort_data(data):
        HamiltonianOperator.__simplify_data(data)
        return HamiltonianOperator.__sort_data(data)
