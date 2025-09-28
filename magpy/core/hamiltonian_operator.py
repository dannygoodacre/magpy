from copy import deepcopy
from numbers import Number
import torch
import magpy as mp
from .._device import _DEVICE_CONTEXT


class HamiltonianOperator:
    """A Hamiltonian operator.

    A representation of a Hamiltonian operator, formed of a sum of Pauli
    operators with constant or function coefficients.

    A batch operator may be created by providing at least one tensor 
    coefficient.

    Examples
    --------
    >>> H = torch.tensor([1,2])*X() + torch.sin*Y()
    >>> H
    tensor([1, 2])*X1 + sin*Y1
    >>> H(torch.tensor(1))
    tensor([[[0.+0.0000j, 1.-0.8415j],
            [1.+0.8415j, 0.+0.0000j]],
            [[0.+0.0000j, 2.-0.8415j],
            [2.+0.8415j, 0.+0.0000j]]], dtype=torch.complex128)

    >>> G = torch.square*X()
    >>> G
    square*X1
    >>> G(torch.tensor(3))
    tensor([[0.+0.j, 9.+0.j],
            [9.+0.j, 0.+0.j]], dtype=torch.complex128)
 
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

        for coeff, pauli_string in pairs:
            try:
                # Move any constant coefficients to the corresponding PauliString.
                if coeff.scale != 1:
                    pauli_string *= coeff.scale
                    coeff.scale = 1

            except AttributeError:
                if isinstance(coeff, Number):
                    pauli_string *= coeff
                    coeff = 1

            try:
                self.data[coeff].append(pauli_string)

            except KeyError:
                self.data[coeff] = pauli_string

            except AttributeError:
                self.data[coeff] = [self.data[coeff], pauli_string]

        self.data = HamiltonianOperator.__simplify_and_sort_data(self.data)

    def __add__(self, other):
        out = HamiltonianOperator()

        try:
            out.data = self.data | other.data

        except AttributeError:
            # PauliString; add it to constants.
            out.data = deepcopy(self.data)

            try:
                out.data[1].append(other)

            except KeyError:
                out.data[1] = other

            except AttributeError:
                out.data[1] = [out.data[1], other]

        else:
            # HamiltionianOperator.
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

    def __call__(self, t=None):
        result = 0

        # TODO: Use new unpack.
        for f, p in self.unpack():
            result += (f(t) if callable(f) else f) * p
            
        return result

    def matrix(self, t=None, n_qubits=None) -> torch.Tensor:
        if n_qubits is None:
            n_qubits = max(max(p.qubits) if p.qubits else 0 for p in self.pauli_operators())

        if self.is_constant():
            return self.__call_time_independent(n_qubits)

        if t is None:
            raise ValueError(
                "Hamiltonian is not constant. A value of t is required.")

        return self.__call_time_dependent(t, n_qubits)

    def __eq__(self, other):
        """Two HamiltonianOperators are equal if they have the same keys in their data
        dictionaries and the corresponding values are equal. If the values are lists,
        they must contain the same elements regardless of order.
        """

        if not isinstance(other, HamiltonianOperator):
            return False

        if set(self.data.keys()) != set(other.data.keys()):
            return False

        for key in self.data:
            self_val = self.data[key]
            other_val = other.data[key]

            if isinstance(self_val, list) and isinstance(other_val, list):
                if len(self_val) != len(other_val):
                    return False

                self_set = {str(ps) for ps in self_val}
                other_set = {str(ps) for ps in other_val}

                if self_set != other_set:
                    return False

            elif isinstance(self_val, list) and not isinstance(other_val, list):
                if len(self_val) != 1 or self_val[0] != other_val:
                    return False

            elif not isinstance(self_val, list) and isinstance(other_val, list):
                if len(other_val) != 1 or self_val != other_val[0]:
                    return False

            elif self_val != other_val:
                return False

        return True
    
    def __mul__(self, other):
        out = HamiltonianOperator()

        try:
            out = sum((p[0]*p[1]*q[0]*q[1] for q in other.unpack() for p in self.unpack()), out)

        except AttributeError:
            out.data = deepcopy(self.data)

            if isinstance(other, Number | torch.Tensor | mp.PauliString):
                for coeff in out.data:
                    try:
                        for i in range(len(out.data[coeff])):
                            out.data[coeff][i] *= other

                    except TypeError:
                        out.data[coeff] *= other

            else:
                # FunctionProduct or other type of function.
                for coeff in list(out.data):
                    out.data[mp.FunctionProduct(coeff, other)] = out.data.pop(coeff)

        out.data = HamiltonianOperator.__simplify_and_sort_data(out.data)

        return out
    
    def __neg__(self):
        out = HamiltonianOperator()
        out.data = deepcopy(self.data)

        return -1 * out
    
    def __repr__(self):
        result = ""

        for f, p in self.data.items():
            try:
                p_str = str(p)

                scale_pos = p_str.find('*') if p_str[0].isnumeric() or p_str.startswith("tensor") else -1

                if isinstance(p.scale, torch.Tensor) or p.scale != 1:
                    result += p_str[:scale_pos] + '*'

                result = HamiltonianOperator.__add_coeff_to_str(result, f)
                
                result += p_str[scale_pos + 1:] if scale_pos > 0 else p_str

            except AttributeError:
                result = HamiltonianOperator.__add_coeff_to_str(result, f)

                result += '(' if f != 1 else ""
                result += " + ".join([str(q) for q in p])
                result += ')' if f != 1 else ""

            result += " + "

        return result[:-3]

    __rmul__ = __mul__

    def __sub__(self, other):
        return self + -other

    def is_constant(self):
        """Return true if the Hamiltonian is time-independent."""

        for coeff in self.data:
            if not isinstance(coeff, Number | torch.Tensor):
                try:
                    if not coeff.is_empty():
                        return False

                except AttributeError:
                    return False

        return True
    
    def is_interacting(self) -> bool:
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

    def qutip(self):
        """Convert to QuTiP form."""

        if self.is_constant():
            return self.__qutip_constant(self.unpack())
        
        constant_components = [state for state in self.unpack() if isinstance(state[0], Number | torch.Tensor)]
        
        time_dependent_components = [state for state in self.unpack() if state not in constant_components]

        return [self.__qutip_constant(constant_components)] + [[c[1].qutip(self.n_qubits), c[0]] for c in time_dependent_components]
    
    def unpack(self, t=None, unit_ops=False):
        """All function-operator pairs in H."""
        
        if not t:
            if unit_ops:
                return [(f if callable(f) else op.scale, op.operator) for f, ops in self.data.items() for op in (ops if isinstance(ops, list) else [ops])]
    
            return [(f, op) for f, ops in self.data.items() for op in (ops if isinstance(ops, list) else [ops])]
        
        if unit_ops:
            return [(f(t) if callable(f) else op.scale, op.operator) for f, ops in self.data.items() for op in (ops if isinstance(ops, list) else [ops])]
        
        return [(f(t) if callable(f) else f, op) for f, ops in self.data.items() for op in (ops if isinstance(ops, list) else [ops])]


    @property
    def batch_count(self):
        return max(p.batch_count for p in self.pauli_operators)
    
    def propagator(self, t=None):
        """The exponential of -i * H(t)."""

        if t is None and not self.is_constant():
            raise ValueError(
                'Hamiltonian is not constant. A value of t is required.')
        
        if self.n_qubits == 1:
            hop = self(t)
            coeffs = torch.tensor(hop.coeffs(unit_ops=True))
            ops = hop.pauli_operators(unit_ops=True)
            
            norm = torch.norm(coeffs)
            
            unit_coeffs = coeffs / norm

            return torch.cos(norm)*mp.Id() - 1j*torch.sin(norm)*sum(c*op for c, op in zip(unit_coeffs, ops))

    def coeffs(self, t=None, unit_ops=False):
        """All coefficients in H."""

        if t:
            return [f(t) if callable(f) else f for f, _ in self.unpack(t, unit_ops)]

        return [f for f, _ in self.unpack(t, unit_ops)]

    def pauli_operators(self, unit_ops=False):
        """All Pauli operators in H."""

        return [op for _, op in self.unpack(unit_ops)]

    @property
    def n_qubits(self):
        """Number of qubits in operator."""
        return max(p.n_qubits for p in self.pauli_operators())

    def __call_time_independent(self, n_qubits):
        """Calculate the matrix representation of the operator."""

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
        """Evaluate the operator at the given time."""
        result = 0

        # TODO: Use new unpack.
        for coeff, p in self.unpack():
            p_val = p(n_qubits).to(_DEVICE_CONTEXT.device)

            try:
                coeff = coeff(t).to(_DEVICE_CONTEXT.device)
            except TypeError:
                pass

            next_term = 0

            try:
                next_term = coeff.reshape(-1,1,1) * p_val

            except AttributeError:
                next_term = coeff * p_val

            # If p is a batch, repeat the current value to agree with its shape.
            try:
                if result.dim() == 2 and next_term.dim() == 3:
                    result = result.repeat(len(next_term), 1, 1)

            except AttributeError:
                pass

            try:
                result += next_term

            except RuntimeError:
                result = result.repeat(next_term.shape[0], 1, 1) + next_term

        return result.squeeze()

    @staticmethod
    def __add_coeff_to_str(result, f):
        try:
            result += f.__name__ + '*'

        except AttributeError:
            try:
                if f != 1:
                    result += str(f) + '*'

            except (RuntimeError, AttributeError):
                result += str(f) + '*'

        return result

    @staticmethod
    def __simplify_data(arrs):
        """Collect all PauliStrings in all lists in"""
        for coeff in arrs:
            arrs[coeff] = mp.PauliString.collect(arrs[coeff])

    @staticmethod
    def __sort_data(data):
        """Move all constant keys to the start of the dictionary."""

        const_keys = []
        other_keys = []

        for key in data:
            (const_keys if isinstance(key, Number | torch.Tensor) else other_keys).append(key)

        return dict((key, data[key]) for key in const_keys + other_keys)

    @staticmethod
    def __simplify_and_sort_data(data):
        HamiltonianOperator.__simplify_data(data)

        return HamiltonianOperator.__sort_data(data)

    @staticmethod
    def __qutip_constant(states):
        if not states:
            return 0

        n = max(max(states[i][1].qubits) for i in range(len(states)))

        return sum(state[0] * state[1].qutip(n) for state in states)
