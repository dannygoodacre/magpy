from .core.function_product import FunctionProduct
from .core.hamiltonian_operator import HamiltonianOperator
from .core.pauli_string import PauliString, X, Y, Z, I
from .linalg import commutes, frob, msqrt_herm, timegrid, uhlmann
from .system import evolve


__all__ = [
    'FunctionProduct',
    'HamiltonianOperator',
    'PauliString', 'X', 'Y', 'Z', 'I',
    'commutes', 'frob', 'msqrt_herm', 'timegrid', 'uhlmann',
    'evolve'
]


type Operator = PauliString | HamiltonianOperator


def set_default_device(device: str):
    """Set the default device for `magpy` and `torch` tensors. The default is `cpu`."""

    from torch import set_default_device
    from ._context import _CONTEXT
    from .system import _update_device

    _CONTEXT.device = device

    set_default_device(_CONTEXT.device)

    PauliString._update_device()

    _update_device()


def set_print_precision(precision: int):
    """Set the number of digits of precision for floating point output."""

    from torch import set_printoptions
    from ._context import _CONTEXT

    _CONTEXT.print_precision = precision

    set_printoptions(precision)


def set_print_identities(arg: bool):
    """Whether to include the identity operators explicitly when displaying operators."""
    from._context import _CONTEXT

    _CONTEXT.print_identities = arg
