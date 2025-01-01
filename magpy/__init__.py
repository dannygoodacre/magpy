from .core import PauliString, X, Y, Z, Id, FunctionProduct, HamiltonianOperator, kron, frobenius, timegrid
from .solver import evolve
from torch import set_default_device

__all__ = [
    'PauliString', 'X', 'Y', 'Z', 'Id', 'FunctionProduct', 'HamiltonianOperator', 'kron', 'frobenius', 'timegrid',
    'evolve'
]


def set_device(device):
    """Set the device to use when evaluating MagPy objects and
    performing calculations. The default is `cpu`.

    Parameters
    ----------
    device : str
        Device name
    """

    from ._device import _DEVICE_CONTEXT
    from .solver.gauss_legendre_quadrature import _update_device
    _DEVICE_CONTEXT.device = device
    set_default_device(_DEVICE_CONTEXT.device)
    _update_device()
