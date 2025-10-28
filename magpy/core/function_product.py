from numbers import Number

from torch import Tensor

import magpy as mp
from .._utils import format_number, format_tensor


class FunctionProduct:
    """A product of functions.

    Example
    -------
    >>> f = FP() * torch.square * torch.sin
    >>> f
    square*sin
    >>> f(torch.tensor(2, dtype=torch.complex128))
    tensor(3.6372+0.j, dtype=torch.complex128)

    Attributes
    ----------
    funcs: dict
        The functions and their exponents in the product
    scale: Number | Tensor
        Constant coefficient
    """

    def __init__(self, *funcs):
        """A product of functions.

        Parameters
        ----------
        *funcs : tuple
            Functions
        """

        self._funcs = {}
        self._scale = 1

        for f in funcs:
            try:
                self._funcs = self.__merge(f.funcs)
                self._scale *= f.scale

            except AttributeError:
                try:
                    self._scale *= f

                except TypeError:
                    self._funcs = FunctionProduct.__add(self, f)

    def __call__(self, arg):
        out = 1

        for f in self._funcs:
                out *= f(arg)

        return out * self._scale

    def __eq__(self, other):
        try:
            return self._funcs == other.funcs and self._scale == other.scale

        except:
            return False

    def __hash__(self):
        return hash(tuple(self._funcs)) + hash(self._scale)

    def __mul__(self, other):
        if isinstance(other, mp.PauliString | mp.HamiltonianOperator):
            return other * self

        result = FunctionProduct()
        result._funcs = dict(self._funcs)
        result._scale = self._scale

        if isinstance(other, Number | Tensor):
            result._scale *= other

        else:
            try:
                result._scale *= other.scale
                result._funcs = self.__merge(other.funcs)

            except AttributeError:
                result._funcs = FunctionProduct.__add(result, other)

        return result

    def __neg__(self):
        result = FunctionProduct()
        result._funcs = dict(self._funcs)
        result._scale = -self._scale

        return result

    def __repr__(self):
        if not self._funcs:
            if isinstance(self._scale, Tensor):
                return format_tensor(self._scale)
            
            return format_number(self._scale)

        result = ''

        if isinstance(self._scale, Tensor):
            result += format_tensor(self._scale) + '*'

        elif self._scale != 1:
            result += format_number(self._scale) + '*'

        if self._funcs.items():
            result += '*'.join(f.__name__ + (f'^{n}' if n > 1 else '') for f, n in self._funcs.items())
        
        return result

    __rmul__ = __mul__
    
    __str__ = __repr__
    
    @property
    def scale(self):
        return self._scale

    @property
    def is_empty(self):
        """Whether the function product contains no functions."""

        return not self._funcs

    def __merge(self, funcs):
        """Merge dictionary into internal function dictionary."""

        return {f: self._funcs.get(f, 0) + funcs.get(f, 0) for f in set(self._funcs) | set(funcs)}

    @staticmethod
    def __add(fp, f):
        """Add function to internal function dictionary."""

        try:
            fp._funcs[f] += 1

        except KeyError:
            fp._funcs[f] = 1

        return fp._funcs
