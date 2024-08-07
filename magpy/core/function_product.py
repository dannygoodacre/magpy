from numbers import Number
from copy import deepcopy
import torch
import magpy as mp


class FunctionProduct:
    """A product of scalar-to-scalar functions.

    A representation of a product of scalar-to-scalar functions, which may
    be from different packages. The exponent of each function in the product
    is determined by the function's corresponding value in the internal
    dictionary.

    Attributes
    ----------
    funcs: dict
        The functions and their exponents in the product
    scale: Number
        Scalar coefficient
    """

    def __init__(self, *funcs):
        """A product of scalar-to-scalar functions.

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
                self.funcs = self.__merge_funcs(f.funcs)
                self.scale *= f.scale
            except AttributeError:
                try:
                    # Scalar.
                    self.scale *= f
                except TypeError:
                    # Other type of function.
                    self.funcs = FunctionProduct.__add_func(self, f)

    def __eq__(self, other):
        return self.funcs == other.funcs and self.scale == other.scale

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
                out.funcs = self.__merge_funcs(other.funcs)
            except AttributeError:
                out.funcs = FunctionProduct.__add_func(out, other)

        return out

    __rmul__ = __mul__

    def __neg__(self):
        return -1 * self

    def __call__(self, arg):
        out = 1
        for f in self.funcs:
            try:
                out *= f(arg.clone().detach())
            except AttributeError:
                out *= f(torch.tensor(arg))

        return out * self.scale

    def __repr__(self):
        return f"{str(self.scale)}*{str(self.funcs)}"

    def __str__(self):
        return (str(self.scale) + "*" if isinstance(self.scale, torch.Tensor) or self.scale != 1 else "") \
            + '*'.join([f.__name__ + (f"^{str(n)}" if n > 1 else "") for f, n in self.funcs.items()])

    def __hash__(self):
        return hash(tuple(self.funcs)) + hash(self.scale)

    def is_empty(self):
        """Return true if function product contains no functions.
        """
        return not self.funcs

    def __merge_funcs(self, funcs):
        # Combine funcs dict with own funcs dict, summing values with shared keys.
        return {f: self.funcs.get(f, 0) + funcs.get(f, 0) for f in set(self.funcs) | set(funcs)}

    @staticmethod
    def __add_func(out, f):
        # Add function to funcs dict, adding new key or incrementing existing value accordingly.
        try:
            out.funcs[f] += 1
        except KeyError:
            out.funcs[f] = 1
        return out.funcs
