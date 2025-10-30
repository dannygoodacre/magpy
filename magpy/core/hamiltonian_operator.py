from __future__ import annotations
from numbers import Number
from copy import deepcopy

import torch
from torch import Tensor

import magpy as mp
from .._utils import conjugate

class HamiltonianOperator:

    def __init__(self, *pairs: tuple[Number | Tensor, mp.PauliString]) -> HamiltonianOperator:
        self._data = {}
        
        for coeff, pauli_string in pairs:
            try:
                if coeff._scale != 1:
                    pauli_string *= coeff._scale
                    coeff._scale = 1

            except AttributeError:
                if isinstance(coeff, Number):
                    pauli_string *= coeff
                    coeff = 1

            try:
                self._data[coeff].append(pauli_string)

            except KeyError:
                self._data[coeff] = pauli_string

            except AttributeError:
                self._data[coeff] = [self._data[coeff], pauli_string]

            self.__simplify_and_sort_data()

    def __add__(self, other):
        result = HamiltonianOperator()
        
        try:
            result._data = self._data | other._data

        except AttributeError:
            result._data = dict(self._data)
            
            try:
                result._data[1].append(other)

            except KeyError:
                result._data[1] = other

            except AttributeError:
                result._data[1] = [result._data[1], other]
                
        else:
            for coeff in list(set(self._data.keys() & other._data.keys())):
                result._data[coeff] = []

                try:
                    result._data[coeff].extend(self._data[coeff])

                except TypeError:
                    result._data[coeff].append(self._data[coeff])

                try:
                    result._data[coeff].extend(other._data[coeff])

                except TypeError:
                    result._data[coeff].append(other._data[coeff])

        result.__simplify_and_sort_data()

        return result
    
    def __call__(self, t: Tensor = None) -> HamiltonianOperator:
        return sum(coeff * pauli_operator for coeff, pauli_operator in self.unpack(t, unit_ops=False))
    
    def __eq__(self, other) -> bool:
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
        result = HamiltonianOperator()

        try:
            result = sum((self_coeff * deepcopy(self_op) * other_coeff * deepcopy(other_op)
                          for other_coeff, other_op in other.unpack(unit_ops=False)
                          for self_coeff, self_op in self.unpack(unit_ops=False)), result)

        except AttributeError:
            result._data = deepcopy(self._data)

            if isinstance(other, Number | Tensor | mp.PauliString):
                for coeff in result._data:
                    try:
                        for i in range(len(result._data[coeff])):
                            result._data[coeff][i] *= other

                    except TypeError:
                        result._data[coeff] *= other

            else:
                for coeff in list(result._data):
                    result._data[mp.FunctionProduct(coeff, other)] = result._data.pop(coeff)

        result.__simplify_and_sort_data()

        return result
    
    def __neg__(self):
        result = HamiltonianOperator()

        result._data = deepcopy(self._data)
        
        return -1 * result

    # TODO: Do we want this to be fully mathematical-looking? I.e. no `*`` and appending `(t)` to function names?
    # TODO: Do we want to group by X, Y, Z, Id?
    def __repr__(self) -> str:
        def coeff_to_str(coeff):
            if isinstance(coeff, mp.FunctionProduct):
                return str(coeff) + '*'
            
            if isinstance(coeff, Number):
                return '' if coeff == 1 else str(coeff) + '*'
            
            if isinstance(coeff, Tensor):
                return str(tuple(coeff.tolist())) + '*'
            
            return coeff.__name__ + '*'

        result = []

        for f, p in self._data.items():
            try:
                p_str = str(p)

                scale_pos = p_str.find('*')

                part = ''

                if isinstance(p.scale, Tensor) or p.scale != 1:
                    part += p_str[:scale_pos] + '*'

                part += coeff_to_str(f) + (p_str[scale_pos + 1:] if scale_pos > 0 else p_str)

            except AttributeError:
                part = coeff_to_str(f)
                
                if f != 1:
                    part += '('
                
                part += ' + '.join([str(q) for q in p])

                if f != 1:
                    part += ')'

            result.append(part)

        return ' + '.join(result)

    __rmul__ = __mul__
    
    __str__ = __repr__

    def __sub__(self, other):
        return self + -other

    @property
    def batch_count(self) -> int:
        """The number of parallel operators represented by the operator."""

        return max(op.batch_count for op in self.pauli_operators())

    @property
    def H(self) -> HamiltonianOperator:
        """The conjugate transpose of the operator."""
        result = HamiltonianOperator()
        result._data = deepcopy(self._data)

        for pauli_operator in result._data.values():
            try:
                pauli_operator._scale = conjugate(pauli_operator._scale)
            
            except AttributeError:
                for op in pauli_operator:
                    op._scale = conjugate(op._scale)

        return result

    @property
    def n_qubits(self) -> int:
        """The number of qubits upon which the operator acts."""

        return max(op.n_qubits for op in self.pauli_operators())

    @property
    def shape(self) -> tuple[Number, Number]:
        return (self.batch_count, self.n_qubits**2, self.n_qubits**2)

    def coeffs(self, t: Tensor = None, unit_ops: bool = False) -> list:
        """All coefficients in the operator.

        Parameters
        ----------
        t : Tensor, optional
            The value at which to evaluate the coefficients, by default None
        unit_ops : bool, optional
            Whether to scale the coefficients such that the corresponding 
            operators have unit coefficients, by default False

        Returns
        -------
        list
            The coefficients in the operator
        """

        coeffs = [coeff if callable(coeff) or isinstance(coeff, Tensor) 
                  else torch.tensor(coeff) 
                  for (coeff, _) in self.unpack(t, unit_ops)]

        # if (self.is_constant() and t is None) or self.batch_count == 1:
        #     return coeffs

        if self.batch_count == 1 or (t is None and not self.is_constant()):
            return coeffs
        
        for i, coeff in enumerate(coeffs):
            if coeff.ndim == 0 or coeff.shape[0] != self.batch_count:
                coeffs[i] = coeff.repeat(self.batch_count)
        
        return coeffs
            
    
    def is_constant(self) -> bool:
        """Whether the operator is time-independent."""
        
        for coeff in self._data:
            if not isinstance(coeff, Number | Tensor):
                try:
                    if not coeff.is_empty:
                        return False
                    
                except AttributeError:
                    return False
                
        return True
    
    def is_interacting(self) -> bool:
        """Whether the operator's qubits are interacting."""

        for operator in self._data.values():
            try:
                if len(operator.qubits()) != 1:
                    return True

            except AttributeError:
                for op in operator:
                    if len(op.qubits) != 1:
                        return True

        return False

    def matrix(self, t: Tensor = None, n_qubits: int = None) -> Tensor:
        """The matrix representation of the operator, evaluated at `t` where
        applicable, as an operator with `n_qubit` qubits.

        Parameters
        ----------
        t : Tensor, optional
            The value at which to evaluate the operator, by default None
        n_qubits : int, optional
            The number of qubits in the operator, by default None

        Returns
        -------
        Tensor
            The matrix representation of the operator
        """

        if n_qubits is None:
            n_qubits = self.n_qubits
            
        result = 0

        for (coeff, op) in self.unpack(t, unit_ops=False, as_matrices=True, n_qubits=n_qubits):
            next_term = 0

            try:
                next_term = coeff.reshape(-1, 1, 1) * op
            
            except AttributeError:
                next_term = coeff * op

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

    def pauli_operators(self, unit_ops: bool = False, as_matrices: bool = False, n_qubits: int = None) -> list:
        """All Pauli operators in the operator.

        Parameters
        ----------
        unit_ops : bool, optional
            Whether to scale the coefficients such that the corresponding operators have unit coefficients, by default False
        as_matrices : bool, optional
            Whether to return the matrix representation of the operator, by default False
        n_qubits : int, optional
            The number of qubits in the operator, by default None

        Returns
        -------
        list
            The Pauli operators
        """

        return [op for (_, op) in self.unpack(unit_ops=unit_ops, as_matrices=as_matrices, n_qubits=n_qubits)]

    def propagator(self, h: Tensor = torch.tensor(1, dtype=torch.complex128), t: Tensor = None) -> HamiltonianOperator:
        """The exponential of `-i * h * H(t)`"""

        if t is None and not self.is_constant():
            raise ValueError('Operator is not constant. A value of t is required.')

        # TODO: Multi-qubit operators.
        if self.n_qubits == 1:
            op = self(t)

            try:
                coeffs = torch.stack(op.coeffs(unit_ops=True)).to(torch.complex128)

                pauli_ops = op.pauli_operators(unit_ops=True)

                norm = torch.norm(coeffs, dim=0)

                if torch.all(norm == 0):
                    return torch.exp(-1j * h * coeffs[0]) * mp.Id()

                H_normalized = sum(c * p for c, p in zip(coeffs / norm, pauli_ops))

                return (torch.cos(h * norm) * mp.Id()
                        - 1j * torch.sin(h * norm) * H_normalized)

            except AttributeError:
                return op.propagator(h, t)

    # TODO: What about NumPy interoperability?
    def qutip(self):
        """The operator as a `QObj` instance."""

        def constant(states):
            if not states:
                return 0
            
            n = max(max(states[i][1]._qubits) for i in range(len(states)))

            return sum(coeff * op.qutip(n) for (coeff, op) in states)

        if self.is_constant():
            return constant(self.unpack())
        
        constant_components = [state for state in self.unpack() if isinstance(state[0], Number | torch.Tensor)]
        
        time_dependent_components = [state for state in self.unpack() if state not in constant_components]
        
        return [constant(constant_components)] + [[op.qutip(self.n_qubits), coeff] for (coeff, op) in time_dependent_components]

    def unpack(self, t: Tensor = None, unit_ops: bool = True, as_matrices: bool = False, n_qubits: int = None) -> list[tuple]:
        """All coefficient-operator pairs in the operator.

        Parameters
        ----------
        t : Tensor, optional
            The value at which to evaluate the operator, by default None
        unit_ops : bool, optional
            Whether to scale the coefficients such that the corresponding operators have unit coefficients, by default False
        as_matrices : bool, optional
            Whether to return the matrix representation of the operator, by default False
        n_qubits : int, optional
            The number of qubits in the operator, by default None

        Returns
        -------
        list[tuple]
            The coefficient-operator pairs
        """
        
        result = []

        for coeff, pauli_strings in self._data.items():
            for pauli_string in (pauli_strings if isinstance(pauli_strings, list) else [pauli_strings]):
                scale = getattr(pauli_string, 'scale', 1)
                is_one = isinstance(scale, Number) and scale == 1

                if t is not None:
                    if callable(coeff):
                        value = coeff(t) if is_one or not unit_ops else scale * coeff(t)

                    else:
                        value = coeff if not unit_ops else scale * coeff

                else:
                    if callable(coeff):
                        value = coeff if is_one or not unit_ops else scale * mp.FunctionProduct(coeff)

                    else:
                        value = coeff if not unit_ops else scale * coeff

                if n_qubits:
                    operator = pauli_string.operator.matrix(n_qubits) if unit_ops else pauli_string.matrix(n_qubits)

                elif as_matrices:
                    operator = pauli_string.operator.matrix(self.n_qubits) if unit_ops else pauli_string.matrix(self.n_qubits)

                else:
                    operator = pauli_string.operator if unit_ops else pauli_string

                result.append((value, operator))

        return result

    def __simplify_and_sort_data(self):
        for coeff in self._data:
            self._data[coeff] = mp.PauliString._collect(self._data[coeff])

        const_coeffs = []
        other_coeffs = []

        for coeff in self._data:
            if isinstance(coeff, Number | Tensor):
                const_coeffs.append(coeff)

            else:
                other_coeffs.append(coeff)

        return dict((coeff, self._data[coeff]) for coeff in const_coeffs + other_coeffs)
