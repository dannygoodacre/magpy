from numbers import Number
from copy import deepcopy

import torch
import magpy as mp


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

        self.funcs = {}
        self.scale = 1

        for f in funcs:
            try:
                # FunctionProduct.
                self.funcs = self.__merge(f.funcs)
                self.scale *= f.scale

            except AttributeError:
                try:
                    # Scalar.
                    self.scale *= f

                except TypeError:
                    # other type of function.
                    self.funcs = FunctionProduct.__add(self, f)

    def __call__(self, arg):
        out = 1

        for f in self.funcs:
                out *= f(arg)

        return out * self.scale

    def __eq__(self, other):
        return self.funcs == other.funcs and self.scale == other.scale
    
    def __hash__(self):
        return hash(tuple(self.funcs)) + hash(self.scale)

    def __mul__(self, other):
        if isinstance(other, mp.PauliString | mp.HamiltonianOperator):
            return other * self

        out = FunctionProduct()
        out.funcs = deepcopy(self.funcs)
        out.scale = deepcopy(self.scale)

        if isinstance(other, Number | torch.Tensor):
            out.scale *= other

        else:
            try:
                out.scale *= other.scale
                out.funcs = self.__merge(other.funcs)

            except AttributeError:
                out.funcs = FunctionProduct.__add(out, other)

        return out

    def __neg__(self):
        return -1 * self

    def __repr__(self):
        return (str(self.scale) + "*" if isinstance(self.scale, torch.Tensor) or self.scale != 1 else "") \
            + '*'.join([f.__name__ + (f"^{str(n)}" if n > 1 else "") for f, n in self.funcs.items()])

    __rmul__ = __mul__

    def __str__(self):
        return repr(self)

    def is_empty(self):
        """Return true if function product contains no functions, else false."""

        return not self.funcs

    def __merge(self, funcs):
        """Merge dictionary into internal function dictionary."""

        return {f: self.funcs.get(f, 0) + funcs.get(f, 0) for f in set(self.funcs) | set(funcs)}

    @staticmethod
    def __add(fp, f):
        """Add function to internal function dictionary."""

        try:
            fp.funcs[f] += 1

        except KeyError:
            fp.funcs[f] = 1

        return fp.funcs
