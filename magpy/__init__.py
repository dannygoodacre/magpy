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

def set_default_device(device):
    """Set the default device for `magpy` and `torch` tensors. The default is `cpu`.

    Parameters
    ----------
    device : str
        Device name
    """

    from torch import set_default_device
    from ._device import _DEVICE_CONTEXT
    from .system import _update_device

    _DEVICE_CONTEXT.device = device

    set_default_device(_DEVICE_CONTEXT.device)

    PauliString._update_device()

    _update_device()
