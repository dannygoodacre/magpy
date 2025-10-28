from .core.function_product import FunctionProduct
from .core.hamiltonian_operator import HamiltonianOperator
from .core.linalg import frob, kron, timegrid, msqrt_herm, uhlmann
from .core.pauli_string import PauliString, X, Y, Z, Id

from .system import evolve

__all__ = [
    'PauliString', 'X', 'Y', 'Z', 'Id', 
    'FunctionProduct', 
    'HamiltonianOperator', 
    'frob', 'kron', 'msqrt_herm', 'timegrid', 'uhlmann'
    'evolve'
]

def set_default_device(device: str):
    """Set the default device for `magpy` and `torch` tensors. The default is `cpu`.

    Parameters
    ----------
    device : str
        Device name
    """

    from torch import set_default_device
    from ._context import _CONTEXT
    from .system import _update_device

    _CONTEXT.device = device

    set_default_device(_CONTEXT.device)

    PauliString._update_device()

    _update_device()

def set_print_precision(precision: int):
    """Set the number of digits of precision for floating point output.

    Parameters
    ----------
    precision : int
        Number of digits
    """
    
    from torch import set_printoptions
    from ._context import _CONTEXT

    _CONTEXT.print_precision = precision

    set_printoptions(precision)
